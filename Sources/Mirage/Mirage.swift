//
//  Mirage.swift
//  Swift facade over the C ABI exposed by `CMirage`.
//
//  Public surface intentionally kept small. The `Engine` actor owns the
//  C++ context for its lifetime, so loading the (multi-GB) weights happens
//  exactly once per `Engine` instance. Generation calls are serialized
//  via the actor isolation — necessary because stable-diffusion.cpp is
//  not safe to drive from multiple threads concurrently against the same
//  context.
//

import Foundation
import CoreGraphics
import ImageIO
import CMirage

// MARK: - Top-level namespace

/// Kiln-Image: a multi-model, on-device diffusion image generator for
/// iOS / macOS / visionOS. Backed by `stable-diffusion.cpp` + ggml-metal.
public enum Mirage {

    /// The engine version reported by the embedded native library.
    public static var nativeVersion: String {
        String(cString: mirage_version())
    }

    /// Install a global progress callback fired once per denoising step.
    /// `step` is 1-indexed, `total` is the configured `steps`, `elapsed` is
    /// seconds since the previous step (the first call also includes any
    /// per-graph warm-up time). Invoked on the engine's worker thread —
    /// hop to your UI actor before touching view state. Pass `nil` to clear.
    public static func setProgressCallback(_ cb: (@Sendable (_ step: Int, _ total: Int, _ elapsed: TimeInterval) -> Void)?) {
        progressLock.lock()
        progressClosure = cb
        progressLock.unlock()

        if cb != nil {
            mirage_set_progress_callback({ step, total, time, _ in
                progressLock.lock()
                let c = progressClosure
                progressLock.unlock()
                c?(Int(step), Int(total), TimeInterval(time))
            }, nil)
        } else {
            mirage_set_progress_callback(nil, nil)
        }
    }
}

// Closure storage for the global progress callback. Protected by `progressLock`
// because sd.cpp invokes the C trampoline from its sampler thread while UI code
// updates the closure from the main actor.
nonisolated(unsafe) private var progressClosure: (@Sendable (Int, Int, TimeInterval) -> Void)?
private let progressLock = NSLock()

// MARK: - Model

/// File-system locations for the three model files Kiln-Image needs to
/// load. Some pipelines only use two of these — pass nil for the rest.
///
/// Reference: see [stable-diffusion.cpp docs/z_image.md](https://github.com/leejet/stable-diffusion.cpp/blob/master/docs/z_image.md)
/// for which files go where for each supported model family.
public struct ModelFiles: Sendable {
    /// Diffusion transformer weights, usually `.gguf` or `.safetensors`.
    public var diffusionModel: URL
    /// VAE encoder/decoder weights, e.g. FLUX's `ae.safetensors`.
    public var vae: URL?
    /// LLM-style text encoder weights, e.g. `Qwen3-4B-Instruct-2507-Q4_K_M.gguf`
    /// for Z-Image. sd.cpp binds this to its `llm` tensor prefix.
    public var textEncoder: URL?
    /// T5-family text encoder weights, e.g. `umt5-xxl-encoder-Q4_K_M.gguf` for
    /// Wan video (or T5-XXL for SD3/Flux). Distinct from `textEncoder` because
    /// sd.cpp binds T5 weights to a different tensor prefix than LLM encoders.
    public var t5Encoder: URL?
    /// Optional tiny autoencoder (TAESD/TAEHV — e.g. `taew2_2.safetensors`
    /// for Wan 2.2). When set, sd.cpp decodes latents through it instead of
    /// the full VAE: a ~23 MB file replacing 1.4 GB, and seconds of decode
    /// instead of minutes, at the cost of softer texture.
    public var taeDecoder: URL?

    public init(diffusionModel: URL, vae: URL? = nil, textEncoder: URL? = nil,
                t5Encoder: URL? = nil, taeDecoder: URL? = nil) {
        self.diffusionModel = diffusionModel
        self.vae = vae
        self.textEncoder = textEncoder
        self.t5Encoder = t5Encoder
        self.taeDecoder = taeDecoder
    }
}

// MARK: - Generation request

/// Inputs to one image-generation call. Field defaults are tuned for
/// Z-Image-Turbo; tweak `cfgScale` and `steps` for other model families.
public struct GenerationRequest: Sendable {
    /// User-facing prompt. UTF-8.
    public var prompt: String
    /// Optional negative prompt. Empty when nil.
    public var negativePrompt: String?
    /// Output width in pixels. Must be a multiple of 8.
    public var width: Int = 1024
    /// Output height in pixels. Must be a multiple of 8.
    public var height: Int = 1024
    /// Sampling steps. Turbo models: 8-9. Full models: 20-50.
    public var steps: Int = 9
    /// Classifier-free guidance scale. Turbo models distill CFG into the
    /// weights and use 1.0; full SDXL etc. use 5-9.
    public var cfgScale: Float = 1.0
    /// RNG seed. Pass nil for a random seed.
    public var seed: Int64? = nil

    public init(
        prompt: String,
        negativePrompt: String? = nil,
        width: Int = 1024,
        height: Int = 1024,
        steps: Int = 9,
        cfgScale: Float = 1.0,
        seed: Int64? = nil
    ) {
        self.prompt = prompt
        self.negativePrompt = negativePrompt
        self.width = width
        self.height = height
        self.steps = steps
        self.cfgScale = cfgScale
        self.seed = seed
    }
}

// MARK: - Video generation request

/// Inputs to one video-generation call. Defaults follow the upstream Wan 2.2
/// TI2V 5B recipe (euler / cfg 6.0 / flow-shift 3.0).
public struct VideoGenerationRequest: Sendable {
    /// User-facing prompt. UTF-8.
    public var prompt: String
    /// Optional negative prompt. Wan responds strongly to its upstream
    /// default negative prompt — pass it through unless the caller overrides.
    public var negativePrompt: String?
    /// Frame width in pixels. Multiple of 16 (Wan's VAE compresses 16×).
    public var width: Int = 480
    /// Frame height in pixels. Multiple of 16.
    public var height: Int = 320
    /// Frame count. Wan requires `4n + 1` (13, 17, 21, …, 33); illegal
    /// counts are rounded down to the nearest legal one by the engine.
    public var frames: Int = 13
    /// Sampling steps. Wan 2.2 5B: 15-25.
    public var steps: Int = 20
    /// Classifier-free guidance scale. Wan 2.2 TI2V 5B: 6.0.
    public var cfgScale: Float = 6.0
    /// Flow-matching shift. Wan: 3.0.
    public var flowShift: Float = 3.0
    /// RNG seed. Pass nil for a random seed.
    public var seed: Int64? = nil
    /// Optional still image to animate (image-to-video). Nil = text-to-video.
    public var initImage: CGImage? = nil
    /// Tile the VAE decode to cap peak memory. Costs a little speed;
    /// recommended anywhere memory is tight (i.e. every iPhone).
    public var vaeTiling: Bool = true
    /// Latent-space tile edge when `vaeTiling` is on. 0 keeps the engine
    /// default (32). Smaller tiles cut the decode's peak allocation —
    /// needed on iPhone, where the default-tile decode buffer exceeds the
    /// per-app memory ceiling at Wan's native resolutions.
    public var vaeTileSize: Int = 0

    public init(
        prompt: String,
        negativePrompt: String? = nil,
        width: Int = 480,
        height: Int = 320,
        frames: Int = 13,
        steps: Int = 20,
        cfgScale: Float = 6.0,
        flowShift: Float = 3.0,
        seed: Int64? = nil,
        initImage: CGImage? = nil,
        vaeTiling: Bool = true,
        vaeTileSize: Int = 0
    ) {
        self.prompt = prompt
        self.negativePrompt = negativePrompt
        self.width = width
        self.height = height
        self.frames = frames
        self.steps = steps
        self.cfgScale = cfgScale
        self.flowShift = flowShift
        self.seed = seed
        self.initImage = initImage
        self.vaeTiling = vaeTiling
        self.vaeTileSize = vaeTileSize
    }
}

/// One generated clip: ordered RGB frames plus their geometry. Feed the
/// frames to `AVAssetWriter` (or any encoder) to produce a movie file.
public struct GeneratedVideo: Sendable {
    public let width: Int
    public let height: Int
    public let frames: [CGImage]

    public init(width: Int, height: Int, frames: [CGImage]) {
        self.width = width
        self.height = height
        self.frames = frames
    }
}

// MARK: - Errors

public enum MirageError: Error, CustomStringConvertible, Sendable {
    /// `mirage_ctx_create` returned NULL. The associated string is the
    /// last-error text from the native side.
    case modelLoadFailed(String)
    /// `mirage_generate` returned NULL.
    case generationFailed(String)
    /// The native-side image was unrecognisable (wrong channel count, zero
    /// dims, etc.).
    case invalidNativeImage(String)
    /// `CGImage` construction from the pixel buffer failed.
    case cgImageCreationFailed
    /// The loaded model cannot generate video (image-only checkpoint).
    case videoUnsupported

    public var description: String {
        switch self {
        case .modelLoadFailed(let s):     return "Kiln-Image: model load failed — \(s)"
        case .generationFailed(let s):    return "Kiln-Image: generation failed — \(s)"
        case .invalidNativeImage(let s):  return "Kiln-Image: native image invalid — \(s)"
        case .cgImageCreationFailed:      return "Kiln-Image: failed to build CGImage from pixel buffer"
        case .videoUnsupported:           return "Kiln-Image: loaded model does not support video generation"
        }
    }
}

// MARK: - Engine

/// Owns the loaded model weights for the lifetime of the actor. Create one
/// engine per model you want to generate against; loading weights is
/// expensive (multi-GB read + GPU upload). Generation calls are serialized
/// by the actor — the underlying C++ context is not thread-safe.
public actor Engine {

    /// Raw pointer to the C++ engine context. Lifetime is the actor's; freed
    /// in `deinit`. Held as `OpaquePointer` because Swift imports the
    /// `mirage_ctx*` typedef that way.
    private let ctx: OpaquePointer

    /// Load a model into a new engine context. Throws if the native side
    /// fails to load the weights (bad path, incompatible quantization, …).
    public init(models: ModelFiles) throws {
        // Convert URLs → C strings. Hold onto the Swift String copies until
        // after the call so the C strings remain valid.
        let diffusion = models.diffusionModel.path
        let vae = models.vae?.path
        let llm = models.textEncoder?.path
        let t5 = models.t5Encoder?.path
        let tae = models.taeDecoder?.path

        func withOptionalCString<T>(_ s: String?, _ body: (UnsafePointer<CChar>?) -> T) -> T {
            if let s = s { return s.withCString { body($0) } }
            return body(nil)
        }

        let result: OpaquePointer? = diffusion.withCString { dPtr in
            withOptionalCString(vae) { vPtr in
                withOptionalCString(llm) { lPtr in
                    withOptionalCString(t5) { tPtr in
                        withOptionalCString(tae) { taePtr in
                            var paths = mirage_model_paths(
                                diffusion_model_path: dPtr,
                                vae_path: vPtr,
                                llm_path: lPtr,
                                t5xxl_path: tPtr,
                                taesd_path: taePtr
                            )
                            return withUnsafePointer(to: &paths) { pp -> OpaquePointer? in
                                mirage_ctx_create(pp)
                            }
                        }
                    }
                }
            }
        }

        guard let ctx = result else {
            throw MirageError.modelLoadFailed(Self.lastNativeError())
        }
        self.ctx = ctx
    }

    deinit {
        mirage_ctx_free(ctx)
    }

    /// Generate one image. The returned `CGImage` is detached from the
    /// native buffer — Kiln frees the C-side buffer before this call
    /// returns.
    public func generate(_ request: GenerationRequest) async throws -> CGImage {
        let promptHolder = request.prompt
        let negHolder = request.negativePrompt ?? ""

        let imgPtr: UnsafeMutablePointer<mirage_image>? = promptHolder.withCString { pPtr in
            negHolder.withCString { nPtr in
                var params = mirage_gen_params(
                    prompt: pPtr,
                    negative_prompt: request.negativePrompt == nil ? nil : nPtr,
                    width: Int32(request.width),
                    height: Int32(request.height),
                    steps: Int32(request.steps),
                    cfg_scale: request.cfgScale,
                    seed: request.seed ?? -1,
                    batch_size: 1
                )
                return withUnsafePointer(to: &params) { pp in
                    mirage_generate(self.ctx, pp)
                }
            }
        }

        guard let imgPtr = imgPtr else {
            throw MirageError.generationFailed(Self.lastNativeError())
        }
        defer { mirage_free_image(imgPtr) }

        let img = imgPtr.pointee
        guard img.width > 0, img.height > 0, img.channels == 4 || img.channels == 3 else {
            throw MirageError.invalidNativeImage(
                "got width=\(img.width) height=\(img.height) channels=\(img.channels)"
            )
        }

        guard let cg = Self.makeCGImage(from: img) else {
            throw MirageError.cgImageCreationFailed
        }
        return cg
    }

    /// True if the loaded checkpoint can generate video (Wan / SVD family).
    public var supportsVideo: Bool {
        mirage_supports_video(ctx)
    }

    /// Generate one video clip. Frames are detached from the native buffer —
    /// the C-side allocation is freed before this returns. Progress callbacks
    /// (see `Mirage.setProgressCallback`) fire once per denoising step.
    public func generateVideo(_ request: VideoGenerationRequest) async throws -> GeneratedVideo {
        guard mirage_supports_video(ctx) else {
            throw MirageError.videoUnsupported
        }

        let promptHolder = request.prompt
        let negHolder = request.negativePrompt ?? ""

        // Flatten the optional init image to a tightly-packed RGB8 buffer.
        var initPixels: [UInt8]? = nil
        var initW: Int32 = 0
        var initH: Int32 = 0
        if let cg = request.initImage {
            if let rgb = Self.makeRGB8(from: cg) {
                initPixels = rgb
                initW = Int32(cg.width)
                initH = Int32(cg.height)
            }
        }

        let videoPtr: UnsafeMutablePointer<mirage_video>? = promptHolder.withCString { pPtr in
            negHolder.withCString { nPtr -> UnsafeMutablePointer<mirage_video>? in
                func run(_ ip: UnsafePointer<UInt8>?) -> UnsafeMutablePointer<mirage_video>? {
                    var params = mirage_video_params(
                        prompt: pPtr,
                        negative_prompt: request.negativePrompt == nil ? nil : nPtr,
                        width: Int32(request.width),
                        height: Int32(request.height),
                        frames: Int32(request.frames),
                        steps: Int32(request.steps),
                        cfg_scale: request.cfgScale,
                        flow_shift: request.flowShift,
                        seed: request.seed ?? -1,
                        init_image_pixels: ip,
                        init_image_width: initW,
                        init_image_height: initH,
                        vae_tiling: request.vaeTiling,
                        vae_tile_size: Int32(request.vaeTileSize)
                    )
                    return withUnsafePointer(to: &params) { pp in
                        mirage_generate_video(self.ctx, pp)
                    }
                }
                if let initPixels {
                    return initPixels.withUnsafeBufferPointer { run($0.baseAddress) }
                }
                return run(nil)
            }
        }

        guard let videoPtr = videoPtr else {
            throw MirageError.generationFailed(Self.lastNativeError())
        }
        defer { mirage_free_video(videoPtr) }

        let v = videoPtr.pointee
        guard v.width > 0, v.height > 0, v.frame_count > 0,
              v.channels == 3 || v.channels == 4 else {
            throw MirageError.invalidNativeImage(
                "got width=\(v.width) height=\(v.height) channels=\(v.channels) frames=\(v.frame_count)"
            )
        }

        let frameBytes = Int(v.width) * Int(v.height) * Int(v.channels)
        var frames: [CGImage] = []
        frames.reserveCapacity(Int(v.frame_count))
        for i in 0..<Int(v.frame_count) {
            let frame = mirage_image(
                width: v.width, height: v.height, channels: v.channels,
                pixels: v.pixels.advanced(by: frameBytes * i)
            )
            guard let cg = Self.makeCGImage(from: frame) else {
                throw MirageError.cgImageCreationFailed
            }
            frames.append(cg)
        }
        return GeneratedVideo(width: Int(v.width), height: Int(v.height), frames: frames)
    }

    // MARK: Private

    private static func lastNativeError() -> String {
        guard let cstr = mirage_last_error() else { return "(no error message)" }
        let s = String(cString: cstr)
        return s.isEmpty ? "(no error message)" : s
    }

    /// Render any CGImage into a tightly-packed RGB8 buffer (no alpha, no
    /// row padding) — the layout sd.cpp expects for init images.
    private static func makeRGB8(from cg: CGImage) -> [UInt8]? {
        let w = cg.width, h = cg.height
        var rgba = [UInt8](repeating: 0, count: w * h * 4)
        let ok: Bool = rgba.withUnsafeMutableBytes { buf in
            guard let ctx = CGContext(
                data: buf.baseAddress, width: w, height: h,
                bitsPerComponent: 8, bytesPerRow: w * 4,
                space: CGColorSpaceCreateDeviceRGB(),
                bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue
            ) else { return false }
            ctx.draw(cg, in: CGRect(x: 0, y: 0, width: w, height: h))
            return true
        }
        guard ok else { return nil }
        var rgb = [UInt8](repeating: 0, count: w * h * 3)
        for i in 0..<(w * h) {
            rgb[i * 3 + 0] = rgba[i * 4 + 0]
            rgb[i * 3 + 1] = rgba[i * 4 + 1]
            rgb[i * 3 + 2] = rgba[i * 4 + 2]
        }
        return rgb
    }

    private static func makeCGImage(from img: mirage_image) -> CGImage? {
        let w = Int(img.width)
        let h = Int(img.height)
        let c = Int(img.channels)
        let bytesPerRow = w * c
        let total = bytesPerRow * h

        // Copy out of the native buffer; the caller frees it via `defer`
        // above. The pixel data needs to outlive this scope, so make our
        // own CFData backing the CGImage.
        guard let data = CFDataCreate(nil, img.pixels, total) else { return nil }
        guard let provider = CGDataProvider(data: data) else { return nil }

        let colorSpace = CGColorSpaceCreateDeviceRGB()
        let bitmapInfo: CGBitmapInfo = c == 4
            ? CGBitmapInfo(rawValue: CGImageAlphaInfo.last.rawValue)
            : CGBitmapInfo(rawValue: CGImageAlphaInfo.none.rawValue)

        return CGImage(
            width: w,
            height: h,
            bitsPerComponent: 8,
            bitsPerPixel: c * 8,
            bytesPerRow: bytesPerRow,
            space: colorSpace,
            bitmapInfo: bitmapInfo,
            provider: provider,
            decode: nil,
            shouldInterpolate: false,
            intent: .defaultIntent
        )
    }
}

// MARK: - CGImage helpers

public extension CGImage {
    /// Encode to PNG `Data`. Convenience for callers that just want bytes
    /// to write to disk.
    func pngData() -> Data? {
        let cf = CFDataCreateMutable(nil, 0)!
        guard let dest = CGImageDestinationCreateWithData(cf, "public.png" as CFString, 1, nil) else {
            return nil
        }
        CGImageDestinationAddImage(dest, self, nil)
        guard CGImageDestinationFinalize(dest) else { return nil }
        return cf as Data
    }
}

// MARK: - Upscaler

/// ESRGAN-family super-resolution, wrapping the native `mirage_upscaler`.
/// Used as the optional second pass after video/image generation: frames
/// come out of the diffusion pipeline at the model's native resolution and
/// this inflates them 2-4x with learned detail.
///
/// Thread-safety mirrors `Engine`: create/use/free from any single thread
/// or actor; calls are internally synchronous.
public final class Upscaler {

    private let handle: OpaquePointer

    /// The model's native scale factor (4 for RealESRGAN_x4plus).
    public var nativeFactor: Int { 4 }

    /// Loads an ESRGAN-class model (.pth / .safetensors).
    /// - Parameter tileSize: caps peak memory during the conv pass;
    ///   0 uses the library default. Use ~128-256 on iPhone.
    public init(modelURL: URL, tileSize: Int = 0) throws {
        let up: OpaquePointer? = modelURL.path.withCString { cPath in
            mirage_upscaler_create(cPath, Int32(tileSize))
        }
        guard let up else {
            throw MirageError.modelLoadFailed(Self.lastError())
        }
        handle = up
    }

    deinit {
        mirage_upscaler_free(handle)
    }

    /// Upscales one frame by `factor`. The ESRGAN always renders at its
    /// native factor (4x for RealESRGAN_x4plus); smaller requested factors
    /// are produced by Lanczos-downscaling that output — supersampling,
    /// which reads cleaner than a native low-factor model would.
    public func upscale(_ image: CGImage, factor: Int = 2) throws -> CGImage {
        let w = image.width, h = image.height
        var rgba = [UInt8](repeating: 0, count: w * h * 4)
        let drawn: Bool = rgba.withUnsafeMutableBytes { buf in
            guard let ctx = CGContext(
                data: buf.baseAddress, width: w, height: h,
                bitsPerComponent: 8, bytesPerRow: w * 4,
                space: CGColorSpaceCreateDeviceRGB(),
                bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue
            ) else { return false }
            ctx.draw(image, in: CGRect(x: 0, y: 0, width: w, height: h))
            return true
        }
        guard drawn else { throw MirageError.cgImageCreationFailed }

        guard let out = rgba.withUnsafeBufferPointer({ buf in
            mirage_upscale(handle, buf.baseAddress, Int32(w), Int32(h), Int32(factor))
        }) else {
            throw MirageError.generationFailed(Self.lastError())
        }
        defer { mirage_free_image(out) }

        guard let cg = Self.cgImage(from: out.pointee) else {
            throw MirageError.cgImageCreationFailed
        }
        // Downscale native-factor output to the requested size when needed.
        let targetW = image.width * factor
        let targetH = image.height * factor
        if cg.width != targetW || cg.height != targetH,
           let scaled = Self.resized(cg, width: targetW, height: targetH) {
            return scaled
        }
        return cg
    }

    private static func resized(_ cg: CGImage, width: Int, height: Int) -> CGImage? {
        guard let ctx = CGContext(
            data: nil, width: width, height: height,
            bitsPerComponent: 8, bytesPerRow: width * 4,
            space: CGColorSpaceCreateDeviceRGB(),
            bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue
        ) else { return nil }
        ctx.interpolationQuality = .high
        ctx.draw(cg, in: CGRect(x: 0, y: 0, width: width, height: height))
        return ctx.makeImage()
    }

    // MARK: Private

    private static func lastError() -> String {
        guard let cstr = mirage_last_error() else { return "(no error message)" }
        let s = String(cString: cstr)
        return s.isEmpty ? "(no error message)" : s
    }

    private static func cgImage(from img: mirage_image) -> CGImage? {
        let w = Int(img.width), h = Int(img.height), c = Int(img.channels)
        let bytes = w * h * c
        guard let pixels = img.pixels else { return nil }
        let data = Data(bytes: pixels, count: bytes)
        guard let provider = CGDataProvider(data: data as CFData) else { return nil }
        return CGImage(
            width: w, height: h,
            bitsPerComponent: 8, bitsPerPixel: 8 * c, bytesPerRow: w * c,
            space: CGColorSpaceCreateDeviceRGB(),
            bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.noneSkipLast.rawValue),
            provider: provider, decode: nil,
            shouldInterpolate: false, intent: .defaultIntent
        )
    }
}

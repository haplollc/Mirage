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
    /// Text encoder weights, e.g. `Qwen3-4B-Instruct-2507-Q4_K_M.gguf` for
    /// Z-Image, T5-XXL for SD3, etc.
    public var textEncoder: URL?

    public init(diffusionModel: URL, vae: URL? = nil, textEncoder: URL? = nil) {
        self.diffusionModel = diffusionModel
        self.vae = vae
        self.textEncoder = textEncoder
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

    public var description: String {
        switch self {
        case .modelLoadFailed(let s):     return "Kiln-Image: model load failed — \(s)"
        case .generationFailed(let s):    return "Kiln-Image: generation failed — \(s)"
        case .invalidNativeImage(let s):  return "Kiln-Image: native image invalid — \(s)"
        case .cgImageCreationFailed:      return "Kiln-Image: failed to build CGImage from pixel buffer"
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

        func withOptionalCString<T>(_ s: String?, _ body: (UnsafePointer<CChar>?) -> T) -> T {
            if let s = s { return s.withCString { body($0) } }
            return body(nil)
        }

        let result: OpaquePointer? = diffusion.withCString { dPtr in
            withOptionalCString(vae) { vPtr in
                withOptionalCString(llm) { lPtr in
                    var paths = mirage_model_paths(
                        diffusion_model_path: dPtr,
                        vae_path: vPtr,
                        llm_path: lPtr
                    )
                    return withUnsafePointer(to: &paths) { pp -> OpaquePointer? in
                        mirage_ctx_create(pp)
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

    // MARK: Private

    private static func lastNativeError() -> String {
        guard let cstr = mirage_last_error() else { return "(no error message)" }
        let s = String(cString: cstr)
        return s.isEmpty ? "(no error message)" : s
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

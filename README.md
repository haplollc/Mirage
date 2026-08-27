<div align="center">

# 🎨 Mirage

### A one-stop on-device diffusion image-generation engine for iOS, macOS, and visionOS.

<sub>Swift package · Metal-accelerated · GGUF & safetensors · drop-in for any sd.cpp-compatible model</sub>

[![Swift](https://img.shields.io/badge/Swift-5.9+-orange.svg)](https://swift.org)
[![Platforms](https://img.shields.io/badge/platforms-iOS%2017+%20|%20macOS%2014+%20|%20visionOS-blue.svg)](https://swift.org)
[![SPM](https://img.shields.io/badge/SPM-compatible-brightgreen.svg)](https://swift.org/package-manager/)
[![License: MIT](https://img.shields.io/badge/license-MIT-lightgrey.svg)](LICENSE)

<table>
<tr>
<td align="center">
  <img src="https://raw.githubusercontent.com/haplollc/Mirage/main/Resources/sample-apple.png" width="220" alt="red apple, 256² Z-Image-Turbo, 4 steps, 28s" /><br/>
  <sub><i>"a single red apple on a white background"</i><br/>256² · 4 steps · <b>28s</b> · M-series Mac · Z-Image-Turbo Q3_K_M</sub>
</td>
<td align="center">
  <img src="https://raw.githubusercontent.com/haplollc/Mirage/main/Resources/sample-puppy.png" width="220" alt="golden retriever puppy in wildflowers, 1024² Z-Image-Turbo, 9 steps, 7.5min" /><br/>
  <sub><i>"a photorealistic golden retriever puppy in a sunlit field of wildflowers"</i><br/>1024² · 9 steps · <b>7.5min</b> · M-series Mac · Z-Image-Turbo Q3_K_M</sub>
</td>
</tr>
</table>

</div>

---

## Why Mirage?

Apple's `ml-stable-diffusion` is great for the **specific** Stable Diffusion checkpoints Apple converted to Core ML — and stops there. Every new diffusion model (Flux, Z-Image, Qwen-Image, ERNIE-Image, Chroma, …) requires its own custom Core ML conversion that takes Apple weeks to publish, if it happens at all.

Mirage takes a different approach: **embed [`stable-diffusion.cpp`](https://github.com/leejet/stable-diffusion.cpp) + `ggml-metal`** into a clean Swift package. Anything sd.cpp can load, Mirage can run. No Core ML conversion required.

```swift
import Mirage

let engine = try Engine(models: ModelFiles(
    diffusionModel: zImageTurboGGUF,
    vae: fluxVAE,
    textEncoder: qwen3GGUF
))
let image = try await engine.generate(.init(prompt: "..."))
```

That's the whole public surface.

---

## Supported model families

Every model below works through the same `Engine` — only the file inputs change.

| Family | Architecture | Example | Status |
|---|---|---|---|
| **Stable Diffusion 1.x / 2.x** | UNet (latent diffusion) | `sd-v1-5.gguf` | ✅ |
| **SDXL / SDXL-Turbo** | UNet (latent diffusion, 2-stage) | `sd-xl-base-1.0.gguf` | ✅ |
| **SD3 / SD3.5** | MMDiT | `sd3.5-medium.gguf` | ✅ |
| **FLUX.1 schnell / dev** | DiT (rectified flow) | `flux1-schnell-Q4_K.gguf` | ✅ |
| **[Chroma1-HD](https://huggingface.co/lodestones/Chroma1-HD)** | FLUX-derived (8B params) | `chroma1-hd.gguf` | ✅ |
| **Qwen-Image** | DiT (1.1B) | `qwen-image-2512.gguf` | ✅ |
| **[ERNIE-Image-Turbo](https://huggingface.co/unsloth/ERNIE-Image-Turbo-GGUF)** | DiT (Turbo-distilled) | `ernie-image-turbo.gguf` | ✅ |
| **[Z-Image-Turbo](https://huggingface.co/Tongyi-MAI/Z-Image-Turbo)** | S3-DiT (6B, Turbo, 8 steps) | `z-image-turbo-Q3_K_M.gguf` | ✅ |

Mirrored, mobile-friendly bundles ship on Hugging Face — each repo includes the diffusion weights, the right text encoder, the right VAE, and a `README.md` with copy-pastable `Engine(...)` snippets:

- 🐶 [`jc-builds/Z-Image-Turbo-iOS`](https://huggingface.co/jc-builds/Z-Image-Turbo-iOS) — 6 B params, 9-step turbo, **6.5 GB**
- 📜 [`jc-builds/ERNIE-Image-Turbo-iOS`](https://huggingface.co/jc-builds/ERNIE-Image-Turbo-iOS) — 8 B params, best-in-class text rendering, **5.9 GB**
- 🎨 [`jc-builds/Chroma1-HD-iOS`](https://huggingface.co/jc-builds/Chroma1-HD-iOS) — 8.9 B FLUX-derived, T5-XXL, **14.5 GB** (iPhone 17 Pro / Mac only)

---

## Install

Add to your `Package.swift`:

```swift
dependencies: [
    .package(url: "https://github.com/haplollc/Mirage.git", from: "0.1.0"),
],
targets: [
    .target(name: "MyApp", dependencies: ["Mirage"]),
]
```

Or in Xcode: **File ▸ Add Package Dependencies…** → `https://github.com/haplollc/Mirage`

The package consumes a prebuilt `sdcpp.xcframework` (Apple Silicon + iOS device + iOS simulator) as a SPM binary target — **no cmake / ninja / clang++ wrangling required on consumer machines**. The framework itself is not in git: fresh checkouts grab it from the latest `sdcpp-*` GitHub release with one command:

```bash
./Scripts/fetch-xcframework.sh   # requires the gh CLI, ~30 s
```

---

## Quick start

```swift
import SwiftUI
import Mirage

struct ImageGenScreen: View {
    @State private var prompt = "a cute corgi astronaut on Mars, photorealistic"
    @State private var output: CGImage?
    @State private var status = "Tap to generate."

    let engine: Engine

    init() throws {
        // 1. Download a model bundle from huggingface.co/HaploApps once
        //    (use Hugging Face's download helpers, or Haplo's model manager).
        let models = ModelFiles(
            diffusionModel: try modelURL("z-image-turbo-Q3_K_M.gguf"),
            vae: try modelURL("ae.safetensors"),
            textEncoder: try modelURL("Qwen3-4B-Instruct-2507-Q4_K_M.gguf")
        )
        // 2. Create the engine ONCE — loading weights is multi-GB I/O + GPU upload.
        self.engine = try Engine(models: models)
    }

    var body: some View {
        VStack {
            if let cg = output {
                Image(decorative: cg, scale: 1).resizable().scaledToFit()
            } else {
                Text(status).foregroundStyle(.secondary)
            }
            TextField("Prompt", text: $prompt).textFieldStyle(.roundedBorder)
            Button("Generate") {
                Task {
                    status = "Generating…"
                    do {
                        // Optional: live per-step progress for the UI. The
                        // callback fires on the sampler thread.
                        Mirage.setProgressCallback { step, total, _ in
                            Task { @MainActor in
                                status = "Step \(step) of \(total)"
                            }
                        }
                        output = try await engine.generate(.init(
                            prompt: prompt,
                            width: 1024, height: 1024,
                            steps: 9, cfgScale: 1.0
                        ))
                        Mirage.setProgressCallback(nil)
                    } catch {
                        status = "\(error)"
                    }
                }
            }
        }
        .padding()
    }
}
```

A complete reference app lives in [`Examples/MirageExampleApp`](Examples/MirageExampleApp).

---

## Running on iPhone

Mirage works on physical iPhones (tested on iPhone 17 Pro with Z-Image-Turbo Q3_K_M). Getting there required a few specific build- and link-time choices documented below — if you're embedding Mirage in an app that also links **llama.cpp** (or any other ggml fork), read this section before debugging weird Metal crashes.

### Required app-side setup

**1. Entitlements.** Multi-GB models hit jetsam without these on:

```xml
<key>com.apple.developer.kernel.increased-memory-limit</key>
<true/>
<key>com.apple.developer.kernel.extended-virtual-addressing</key>
<true/>
```

Without `increased-memory-limit` the per-app cap on a 12 GB iPhone is ~6 GB; with it, ~10 GB. Z-Image-Turbo at 6.5 GB will not load otherwise.

**2. Disable ggml-metal's experimental tensor API at app launch.** This codepath is unstable on A19-class iPhones; set the env var in your `@main` `init()` so it lands before any ggml-metal probe fires:

```swift
@main
struct MyApp: App {
    init() {
        setenv("GGML_METAL_TENSOR_DISABLE", "1", 1)
        setenv("GGML_METAL_FUSION_DISABLE",  "1", 1)
    }
    // …
}
```

`GGML_METAL_FUSION_DISABLE` works around a separate bug where the fused `rms_norm + mul` Metal kernel binds a NULL buffer at slot 2 for mmap-backed weights.

**3. Gate by available memory before loading.** `os_proc_available_memory()` returns the remaining bytes before jetsam; refuse the load if `diffusion_file_size + 1 GB > available`:

```swift
let diffusionBytes = try FileManager.default.attributesOfItem(
    atPath: diffusionURL.path
)[.size] as? Int64 ?? 0
let needed = diffusionBytes + 1_073_741_824 // 1 GB activations headroom
let avail  = Int64(os_proc_available_memory())
guard avail == 0 || needed <= avail else {
    throw MyError.notEnoughMemory(neededMB: needed/1_048_576, availMB: avail/1_048_576)
}
```

**4. Don't ship Chroma1-HD on any iPhone.** 14.5 GB of weights exceeds even the increased-memory-limit cap on any shipping iPhone. Filter it out of your picker. Z-Image-Turbo (6.5 GB) and ERNIE-Image-Turbo (5.9 GB) are the realistic iPhone targets.

### Symbol-collision gotcha (READ THIS if you also link llama.cpp)

`libmirage-sdcpp.a` partial-links sd.cpp + ggml + ggml-metal with `ld -r -exported_symbols_list` so only the public `mirage_*` C entry points are globally visible. Every other symbol — including `_ggml_metal_library_init`, `_ggml_backend_graph_compute`, all the kernel-encoder functions — is local to the archive.

This is intentional. The Mirage native engine is sd.cpp's vendored ggml-metal commit. Many apps that ship Mirage also vendor **llama.cpp**, which compiles its OWN copy of the same-named ggml-metal functions from a different commit. Without symbol hiding, Apple's ld picks the first definition it finds (usually llama's, depending on link order), and **every** sd.cpp call into ggml-metal at runtime routes into llama's older code — which has a different kernel set, a different metallib resource lookup, and a different binding ABI. The symptoms range from `"Function kernel_mul_mm_bf16_f32 was not found in the library"` (llama's metallib doesn't have bf16 mul_mm) to `EXC_BAD_ACCESS` deep in `ggml_metal_encoder_set_pipeline` (NULL pipeline lookup against a sibling's kernel cache).

If you rebuild the XCFramework yourself, **do not** replace the `ld -r` step with `libtool -static` — `libtool -static` leaves every symbol exported and reintroduces the collision. The build script's sanity-check fails the build if the collision is detectable.

### Memory tuning, in order of importance

The XCFramework ships with these `sd_ctx_params` overrides (see `Sources/CMirage/sd/MirageC.cpp`). Each is the result of a real on-device failure mode, so don't flip them away from these values without testing.

| Flag | Value | Why |
|---|---|---|
| `enable_mmap` | `true` | Cuts peak load memory from 2× weights to ~1× (lazily paged-in). Required or jetsam fires during load. |
| `offload_params_to_cpu` | `true` | Keeps diffusion weights in CPU-side mmap pages; copies per-op to GPU. Sets the routing to *not* use `buffer_from_host_ptr` directly. Faster paths exist on paper but tip into OOM at sampling step 7-9 on a 12 GB device. |
| `keep_clip_on_cpu` | `true` | The text encoder (Qwen3-4B / T5-XXL) is 2-3 GB on GPU; on CPU it's an N-second one-shot at the start of each generation, hidden under the longer sampling phase. |
| `keep_vae_on_cpu` | `true` | VAE decode on CPU is slow (~30-60 s) but adds zero GPU residency. Flipping to GPU saves time but adds ~300 MB to peak — flag preserved as a future opt-in. |
| `free_params_immediately` | `false` | sd.cpp's default is `true`, which releases the diffusion-model param tensors at the end of every `generate_image` call. The next generation against the same `sd_ctx` dereferences freed buffers and crashes. We keep params alive for the engine's lifetime — caller controls unload via `Engine` lifetime. |
| `diffusion_flash_attn` | `true` | Reduces attention working memory. |
| `diffusion_conv_direct` | `true` | Reduces conv working memory. |

### Embedded Metal library

`GGML_METAL_EMBED_LIBRARY=ON` is forced in the build script so the `.metal` source is embedded as `.incbin` bytes inside `libmirage-sdcpp.a` and JIT-compiled on first use. The alternative (default) ships a precompiled `default.metallib` as a bundle resource and scans `Bundle.main` for it at runtime — which would either grab the wrong consumer's metallib (see symbol-collision above) or fail outright when the host app doesn't ship `default.metallib` at all.

First engine load on iPhone adds ~8 s for the Metal source compile; subsequent loads in the same process are cached.

### Progress callback

For UIs that need step-level feedback (essential on iPhone where a 9-step generation is 5-10 minutes):

```swift
Mirage.setProgressCallback { step, total, elapsed in
    // step is 1-indexed, total = configured steps, elapsed = sec since last tick.
    // Fires on the sampler thread — hop to MainActor before touching UI.
    Task { @MainActor in
        progress = Double(step) / Double(total)
    }
}
```

The first callback also includes the warm-up + first-step shader-JIT time (~30-90 s on a fresh process). Subsequent ticks are honest per-step durations, so use steps ≥ 2 to compute a moving-average ETA.

---

## Memory & device sizing

Diffusion weights + text encoder + activations have to live in GPU memory at the same time. iPhone memory ceilings are real.

| Device | RAM | What fits? |
|---|---|---|
| iPhone 17 Pro / Air | 12 GB | Any model in this README, up to ~7 GB weights total (Z-Image-Turbo Q8, Flux Q5, SD3.5 Medium) |
| iPhone 16 Pro / iPad M-series | 8 GB | Z-Image-Turbo Q3_K (~6.5 GB total), SDXL-Turbo Q4 (~5 GB total) |
| iPhone 15 Pro | 8 GB | Same as 16 Pro, slightly tighter |
| iPhone 14 and older | 6 GB | SD1.5 / SDXL-Turbo at Q4 only. Larger models will OOM. |

Engine ships with `keep_clip_on_cpu = true` by default — keeps the text encoder off the GPU which saves ~2-3 GB on iPhone.

You should gate model availability by device:

```swift
let physicalRAM = ProcessInfo.processInfo.physicalMemory
guard physicalRAM >= 8 * 1024 * 1024 * 1024 else {
    // Show "Z-Image needs a newer iPhone" instead of trying to load.
    return
}
```

---

## Performance (rough)

Numbers from a 1024×1024 generation at the recommended step count for each family.

| Device | Z-Image-Turbo Q3 (9 steps) | SDXL-Turbo Q4 (4 steps) | SD3.5-Medium Q4 (28 steps) |
|---|---|---|---|
| iPhone 17 Pro | ~3 min | ~30 s | ~5 min |
| iPhone 16 Pro | ~5 min | ~45 s | ~8 min |
| M2 / M3 Mac | ~7 min | ~30 s | ~3 min |

These are **engine-side wall-clock** times, not including the first-time model load (multi-GB read + GPU upload, ~10-30 s once per app launch).

For "feels fast" generation on iPhone, ship the Turbo variants — they're distilled to 4-9 steps vs the 28-50 steps a non-turbo model needs.

---

## How it works

```
                                        ┌─────────────────────┐
your prompt + model paths               │   Mirage (Swift) │
       │                                │  ┌───────────────┐  │
       ▼                                │  │ public  API   │  │
   ┌───────┐    actor isolation         │  └──────┬────────┘  │
   │Engine │  ◄──────────────────────────┼────────┘           │
   │ actor │                             │                    │
   └───┬───┘                             │   CMirage (C)   │
       │ mirage_generate(...)              │  ┌───────────────┐ │
       ▼                                 │  │ MirageC.cpp│ │
   ┌──────────────────────────────────┐  │  └──────┬────────┘ │
   │  stable-diffusion.cpp / ggml     │  │         │          │
   │  + Metal backend (compute kernels)│ │         ▼          │
   └──────────────────────────────────┘  │   sdcpp.xcframework│
       │                                 │   (prebuilt binary)│
       ▼                                 └─────────────────────┘
   CGImage (you decide what to do with it)
```

- **Public Swift API** is one `actor Engine` + `Engine.generate(_:)` returning `CGImage`. Actor isolation serializes calls because the underlying C++ context isn't thread-safe against itself.
- **C bridge** is a 12-symbol header (`MirageC.h`) that's deliberately tiny so upstream sd.cpp churn doesn't reach Swift.
- **Native engine** is `stable-diffusion.cpp` (MIT) running on `ggml-metal`. We compile it into an XCFramework so SPM consumers don't need cmake/ninja installed.

---

## Getting the XCFramework

Two ways — fetch the prebuilt one (fast path, what app devs want):

```bash
./Scripts/fetch-xcframework.sh        # downloads the latest sdcpp-* release (~30 s)
```

…or rebuild it from the vendored sources (required whenever
`Sources/CMirage/vendor` changes; publish the result as a new `sdcpp-*`
release so fetchers stay current):

```bash
git submodule update --init --recursive
./Scripts/build-xcframework.sh        # all platforms (~5-10 min)
./Scripts/build-xcframework.sh macos  # macOS arm64 only (fastest, ~2 min)
./Scripts/build-xcframework.sh ios    # iOS device + simulator
```

Then `swift build` / `swift test` work normally.

---

## Tests

```bash
# Fast smoke (always run, < 10s)
swift test --filter MirageSmokeTests

# Heavy integration (requires a folder with model files)
MIRAGE_TEST_MODELS_DIR=$HOME/Downloads/kiln-models \
    swift test --filter MirageHeavyIntegrationTests
```

The heavy tests load real multi-GB weights and generate small images. They're gated on `MIRAGE_TEST_MODELS_DIR` so CI doesn't try to ship 6+ GB through every PR.

---

## Limitations

- **Generation is slow on iPhone.** Even the Turbo variants take 3-10 minutes for a 1024² image. Show a clear progress UI. SDXL-Turbo at 512² is the closest thing to interactive (~30 s on iPhone 17 Pro).
- **No upscaler integration yet.** sd.cpp supports ESRGAN/4x; we haven't surfaced it in the Swift API. Drop a feature request if you need it.
- **No LoRA / textual inversion API yet.** sd.cpp supports them; we just haven't surfaced them. Easy to add when needed.
- **ControlNet not exposed.** Same story.
- **iOS Simulator works** for smoke tests, but the Simulator's Metal stack is much slower than a real device — don't benchmark there.

---

## Built by

[Haplo](https://haplo.app) — on-device AI for iOS. The same engine powers Haplo's in-app image generation.

If Mirage shows up in your app, [tell us about it](https://twitter.com/jc_builds).

---

## Credits

- [`stable-diffusion.cpp`](https://github.com/leejet/stable-diffusion.cpp) — the engine doing all the actual work
- [`ggml`](https://github.com/ggml-org/ggml) — the tensor library underneath
- Apple's Metal team — for `ggml-metal` working at all on a phone

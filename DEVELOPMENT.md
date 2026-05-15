# Contributing to Mirage

## Source layout

```
Mirage/
├── Package.swift                 ← SPM manifest
├── Frameworks/sdcpp.xcframework  ← prebuilt binary (gitignored, built by Scripts/)
├── Scripts/
│   └── build-xcframework.sh      ← cmake + ninja → libmirage-sdcpp.a → xcframework
├── Sources/
│   ├── CMirage/
│   │   ├── include/
│   │   │   ├── MirageC.h      ← public C header Swift imports
│   │   │   └── module.modulemap
│   │   ├── sd/
│   │   │   └── MirageC.cpp    ← C wrapper that calls sd.cpp internals
│   │   └── vendor/
│   │       └── sd-cpp/           ← git submodule, stable-diffusion.cpp pinned
│   └── Mirage/
│       └── Mirage.swift       ← public Swift API (Engine actor + types)
├── Tests/MirageTests/         ← XCTest suites (smoke + heavy)
├── Examples/                     ← single-file SwiftUI reference app
└── Resources/                    ← README hero images + sample outputs
```

## Building locally

You need cmake + ninja + Xcode 15+ + an Apple Silicon Mac.

```bash
git submodule update --init --recursive
./Scripts/build-xcframework.sh macos   # ~2 min, builds the macOS arm64 slice
swift build
swift test --filter MirageSmokeTests
```

For iOS device + simulator slices (slower, ~5-10 min total):

```bash
./Scripts/build-xcframework.sh ios
```

For the full XCFramework that ships in releases:

```bash
./Scripts/build-xcframework.sh         # all platforms
```

## Tests

**Smoke tests** (`MirageSmokeTests`) — fast (<10 s), always run. They check:
- The native lib loads and reports a sane version string
- Bad model paths raise `MirageError.modelLoadFailed` with a useful message

**Heavy integration tests** (`MirageHeavyIntegrationTests`) — slow, env-gated. Set `MIRAGE_TEST_MODELS_DIR` to a folder containing:

```
diffusion.gguf           (or z-image-turbo-Q3_K_M.gguf)
vae.safetensors          (or ae.safetensors)
text-encoder.gguf        (or Qwen3-4B-Instruct-2507-Q4_K_M.gguf)
```

then run:

```bash
MIRAGE_TEST_MODELS_DIR=~/Downloads/kiln-models \
    swift test --filter MirageHeavyIntegrationTests
```

Heavy tests produce a small (256×256, 4 steps) image to `$TMPDIR/kiln-tiny.png` for human eyeballing.

## Releasing

1. Bump the version in `Sources/CMirage/sd/MirageC.cpp::mirage_version()`.
2. `./Scripts/build-xcframework.sh` — produces `Frameworks/sdcpp.xcframework`.
3. `cd Frameworks && zip -r sdcpp.xcframework.zip sdcpp.xcframework`
4. `swift package compute-checksum sdcpp.xcframework.zip` — copy the hash.
5. Update `Package.swift`'s `binaryTarget(url:checksum:)` (replace the local-path mode).
6. `git tag X.Y.Z && git push origin X.Y.Z`
7. `gh release create X.Y.Z sdcpp.xcframework.zip --title "Mirage X.Y.Z" --notes "…"`

## Updating stable-diffusion.cpp

```bash
cd Sources/CMirage/vendor/sd-cpp
git fetch origin
git checkout <new-commit-sha>
cd ../../../../..
./Scripts/build-xcframework.sh
swift test --filter MirageHeavyIntegrationTests
```

If `MirageC.cpp` doesn't compile, sd.cpp's public C ABI changed — check `vendor/sd-cpp/include/stable-diffusion.h` and update the wrapper. Keep the public `MirageC.h` ABI stable across these bumps.

## Memory profiling on iPhone

Use Xcode Instruments → Allocations + Metal Frame Capture. The hot spots:

- **Model load** — diffusion + text encoder + VAE all hit GPU memory simultaneously. Tuning `keep_clip_on_cpu`, `keep_vae_on_cpu`, and `offload_params_to_cpu` in `MirageC.cpp::mirage_ctx_create` is how we shrink the GPU residency.
- **Sampling** — activations grow with image resolution. 1024² uses ~3-4× the GPU memory of 512².
- **VAE decode** — final spike at the end. Tile the decode if you're already at the memory ceiling.

## Filing issues / PRs

PRs welcome. Especially:
- Adding LoRA / textual inversion to the Swift API (sd.cpp already supports them)
- ControlNet
- Upscaler integration
- More model bundles on HaploApps' HF org

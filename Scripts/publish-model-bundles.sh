#!/usr/bin/env bash
#
# publish-model-bundles.sh
#
# Mirror upstream model files into HaploApps/* HF repos so Mirage
# consumers can fetch a single, documented bundle per model family.
#
# Each bundle is a separate HF repo with:
#   - the diffusion weights (Q3_K_M GGUF for iPhone-friendly size)
#   - the matching text encoder (also GGUF when possible)
#   - the matching VAE (.safetensors)
#   - a custom model card cross-linking to upstream + to Mirage
#
# Run interactively from a checkout of Mirage. Requires `hf` CLI logged
# in to an account that can push to `HaploApps/*`.
#
# Usage:
#   ./Scripts/publish-model-bundles.sh z-image-turbo
#   ./Scripts/publish-model-bundles.sh qwen-image
#   ./Scripts/publish-model-bundles.sh ernie-image-turbo
#   ./Scripts/publish-model-bundles.sh chroma1-hd
#   ./Scripts/publish-model-bundles.sh all       # all of the above
#
set -euo pipefail

ORG="HaploApps"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
STAGE_ROOT="$ROOT/.build/model-stage"
mkdir -p "$STAGE_ROOT"

# Each function below:
#   1. Downloads the upstream files into a local stage dir
#   2. Writes a custom README.md (model card)
#   3. Creates the HaploApps/* repo if it doesn't exist
#   4. Uploads everything

publish_z_image_turbo() {
    local SLUG="Z-Image-Turbo-iOS"
    local STAGE="$STAGE_ROOT/$SLUG"
    mkdir -p "$STAGE"

    echo "==> Downloading Z-Image-Turbo bundle to $STAGE"
    hf download unsloth/Z-Image-Turbo-GGUF z-image-turbo-Q3_K_M.gguf  --local-dir "$STAGE"
    hf download unsloth/Z-Image-Turbo-GGUF z-image-turbo-Q4_K_M.gguf  --local-dir "$STAGE" || true
    hf download unsloth/Qwen3-4B-Instruct-2507-GGUF Qwen3-4B-Instruct-2507-Q4_K_M.gguf --local-dir "$STAGE"
    hf download ffxvs/vae-flux ae.safetensors --local-dir "$STAGE"

    write_z_image_card "$STAGE/README.md"

    echo "==> Creating $ORG/$SLUG (idempotent)"
    hf repos create "$ORG/$SLUG" --type model --exist-ok

    echo "==> Uploading"
    hf upload "$ORG/$SLUG" "$STAGE" . --commit-message "Mirage: Z-Image-Turbo iOS bundle"
}

publish_qwen_image() {
    local SLUG="Qwen-Image-2512-iOS"
    local STAGE="$STAGE_ROOT/$SLUG"
    mkdir -p "$STAGE"
    # TODO: fill in once the upstream model is published with confirmed file names.
    echo "[TODO] Qwen/Qwen-Image-2512 layout not yet published with stable filenames."
}

publish_ernie_image_turbo() {
    local SLUG="ERNIE-Image-Turbo-iOS"
    local STAGE="$STAGE_ROOT/$SLUG"
    mkdir -p "$STAGE"
    # Mirrors unsloth/ERNIE-Image-Turbo-GGUF — fill in the exact GGUF
    # filename once we've confirmed sd.cpp accepts it.
    echo "[TODO] ERNIE-Image-Turbo: confirm sd.cpp acceptance then mirror."
}

publish_chroma1_hd() {
    local SLUG="Chroma1-HD-iOS"
    local STAGE="$STAGE_ROOT/$SLUG"
    mkdir -p "$STAGE"
    # Mirrors lodestones/Chroma1-HD — same shape as Flux.
    echo "[TODO] Chroma1-HD: pick Q4_K_M weights + Flux VAE + T5/Llama text encoder."
}

write_z_image_card() {
    local OUT=$1
    cat > "$OUT" <<'EOF'
---
license: apache-2.0
language: ["en", "zh"]
pipeline_tag: text-to-image
tags:
  - text-to-image
  - diffusion
  - z-image
  - s3-dit
  - gguf
  - quantized
  - on-device
  - ios
  - mobile
base_model: Tongyi-MAI/Z-Image-Turbo
---

# Z-Image-Turbo — iOS bundle

A pre-flighted bundle of Z-Image-Turbo + Qwen3-4B text encoder + FLUX VAE,
sized and quantized to fit on iPhone 16 Pro / 17 Pro and run via
[Mirage](https://github.com/haplollc/Mirage).

## What's inside

| File | Role | Size |
|---|---|---|
| `z-image-turbo-Q3_K_M.gguf` | Diffusion transformer (6B params, Q3_K_M quant) | 4.2 GB |
| `z-image-turbo-Q4_K_M.gguf` | Diffusion transformer (6B params, Q4_K_M quant) | 5.0 GB |
| `Qwen3-4B-Instruct-2507-Q4_K_M.gguf` | Text encoder | 2.5 GB |
| `ae.safetensors` | VAE (from FLUX.1) | 320 MB |

**Pick one of the two diffusion files** — Q3 for iPhone 15/16 Pro, Q4 for iPhone 17 Pro / Mac.

## Usage (Mirage)

```swift
import Mirage

let docs = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
let engine = try Engine(models: ModelFiles(
    diffusionModel: docs.appendingPathComponent("z-image-turbo-Q3_K_M.gguf"),
    vae:            docs.appendingPathComponent("ae.safetensors"),
    textEncoder:    docs.appendingPathComponent("Qwen3-4B-Instruct-2507-Q4_K_M.gguf")
))

let image = try await engine.generate(.init(
    prompt: "a photorealistic golden retriever puppy in a sunlit field of wildflowers",
    width: 1024, height: 1024,
    steps: 9,         // Turbo distillation — don't go higher
    cfgScale: 1.0     // ditto, CFG is baked in
))
```

## Architecture

Z-Image-Turbo is a 6B-parameter **S3-DiT** (Scalable Single-Stream Diffusion
Transformer) — text, visual semantic tokens, and image VAE tokens are all
concatenated into one input stream, maximising parameter efficiency vs the
dual-stream MMDiT designs.

It's distilled to **8-9 sampling steps** via Decoupled-DMD + DMDR, which is
why generation is fast even with 6B params loaded.

## Performance (Mirage)

| Device | 1024² @ 9 steps |
|---|---|
| iPhone 17 Pro | ~3 min |
| iPhone 16 Pro | ~5 min |
| M2 / M3 Mac | ~7 min |

Memory ceiling — full pipeline residency is ~7 GB. iPhone 14 and older
cannot run this bundle; gate availability on `ProcessInfo.processInfo.physicalMemory`.

## Provenance

- Upstream diffusion model: [Tongyi-MAI/Z-Image-Turbo](https://huggingface.co/Tongyi-MAI/Z-Image-Turbo)
- GGUF conversion: [unsloth/Z-Image-Turbo-GGUF](https://huggingface.co/unsloth/Z-Image-Turbo-GGUF)
- Text encoder: [unsloth/Qwen3-4B-Instruct-2507-GGUF](https://huggingface.co/unsloth/Qwen3-4B-Instruct-2507-GGUF)
- VAE: [ffxvs/vae-flux](https://huggingface.co/ffxvs/vae-flux) (re-host of FLUX.1's `ae.safetensors`)

## License

Apache 2.0 — same as upstream Z-Image-Turbo. Text encoder weights are
distributed under the Tongyi-MAI license terms; VAE weights are under
Flux's non-commercial license.

## Built by

[Haplo](https://haplo.app) · [Mirage on GitHub](https://github.com/haplollc/Mirage)
EOF
}

case "${1:-help}" in
    z-image-turbo)        publish_z_image_turbo ;;
    qwen-image)           publish_qwen_image ;;
    ernie-image-turbo)    publish_ernie_image_turbo ;;
    chroma1-hd)           publish_chroma1_hd ;;
    all)
        publish_z_image_turbo
        publish_qwen_image
        publish_ernie_image_turbo
        publish_chroma1_hd
        ;;
    *)
        cat <<EOF
Usage: $0 <bundle>

Available bundles:
  z-image-turbo        — Z-Image-Turbo 6B + Qwen3-4B + Flux VAE
  qwen-image           — Qwen-Image-2512                          (TODO)
  ernie-image-turbo    — ERNIE-Image-Turbo                        (TODO)
  chroma1-hd           — Chroma1-HD (Flux-derived)                (TODO)
  all                  — publish every bundle in turn
EOF
        exit 1
        ;;
esac

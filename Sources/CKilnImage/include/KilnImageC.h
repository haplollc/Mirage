//
//  KilnImageC.h
//  C bridge between Swift and stable-diffusion.cpp.
//
//  This is the only header the Swift side imports. Hides every sd.cpp /
//  ggml type behind opaque pointers + C-friendly POD structs so the Swift
//  module map stays small and stable across upstream churn.
//

#ifndef KILN_IMAGE_C_H
#define KILN_IMAGE_C_H

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

// MARK: - Engine handle

typedef struct kiln_ctx kiln_ctx;

// MARK: - Model paths

/// Paths to the three model files Kiln needs to load. All UTF-8 nul-terminated
/// C strings. Pass NULL for fields that don't apply (e.g. some models bundle
/// the text encoder; in that case leave `llm_path` NULL).
typedef struct {
    const char* diffusion_model_path;   ///< The .gguf or .safetensors diffusion transformer weights.
    const char* vae_path;               ///< VAE encoder/decoder weights (often Flux's ae.safetensors).
    const char* llm_path;               ///< Text encoder GGUF (Qwen3-4B for Z-Image, T5 for SD3/Flux, …).
} kiln_model_paths;

// MARK: - Generation parameters

typedef struct {
    const char* prompt;                 ///< User prompt, UTF-8.
    const char* negative_prompt;        ///< Optional negative prompt. NULL = none.
    int32_t width;                      ///< Output width in pixels (must be a multiple of 8). Default 1024.
    int32_t height;                     ///< Output height in pixels (must be a multiple of 8). Default 1024.
    int32_t steps;                      ///< Number of sampling steps. Z-Image-Turbo: 8-9.
    float   cfg_scale;                  ///< Classifier-free guidance scale. Turbo models use 1.0.
    int64_t seed;                       ///< RNG seed. -1 picks a random one.
    int32_t batch_size;                 ///< Number of images per call. Default 1.
} kiln_gen_params;

/// One generated image as a tightly-packed RGBA8 buffer the Swift side can
/// hand to CGImage / UIImage without further allocation.
typedef struct {
    int32_t  width;
    int32_t  height;
    int32_t  channels;                  ///< Always 4 (RGBA).
    uint8_t* pixels;                    ///< Owned by Kiln; freed by `kiln_free_image`.
} kiln_image;

// MARK: - Lifecycle

/// Load the given model files into a new engine context. Returns NULL on
/// failure (and writes a human-readable reason to `kiln_last_error`).
kiln_ctx* kiln_ctx_create(const kiln_model_paths* paths);

/// Tear down an engine context and release its weights.
void kiln_ctx_free(kiln_ctx* ctx);

// MARK: - Generation

/// Run the diffusion sampler against `params` and return a heap-allocated
/// `kiln_image*`. NULL on failure. Caller frees with `kiln_free_image`.
kiln_image* kiln_generate(kiln_ctx* ctx, const kiln_gen_params* params);

/// Release an image returned by `kiln_generate`.
void kiln_free_image(kiln_image* img);

// MARK: - Diagnostics

/// Human-readable description of the last failure on this thread. Empty
/// string if no error has been recorded. Lifetime: until the next Kiln call
/// on this thread.
const char* kiln_last_error(void);

/// Engine version, in the format "MAJOR.MINOR.PATCH" — bumped on breaking
/// changes to the C ABI above.
const char* kiln_version(void);

#ifdef __cplusplus
}
#endif

#endif // KILN_IMAGE_C_H

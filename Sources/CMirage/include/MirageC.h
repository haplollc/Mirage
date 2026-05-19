//
//  MirageC.h
//  C bridge between Swift and stable-diffusion.cpp.
//
//  This is the only header the Swift side imports. Hides every sd.cpp /
//  ggml type behind opaque pointers + C-friendly POD structs so the Swift
//  module map stays small and stable across upstream churn.
//

#ifndef MIRAGE_C_H
#define MIRAGE_C_H

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

// MARK: - Engine handle

typedef struct mirage_ctx mirage_ctx;

// MARK: - Model paths

/// Paths to the three model files Kiln needs to load. All UTF-8 nul-terminated
/// C strings. Pass NULL for fields that don't apply (e.g. some models bundle
/// the text encoder; in that case leave `llm_path` NULL).
typedef struct {
    const char* diffusion_model_path;   ///< The .gguf or .safetensors diffusion transformer weights.
    const char* vae_path;               ///< VAE encoder/decoder weights (often Flux's ae.safetensors).
    const char* llm_path;               ///< Text encoder GGUF (Qwen3-4B for Z-Image, T5 for SD3/Flux, …).
} mirage_model_paths;

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
} mirage_gen_params;

/// One generated image as a tightly-packed RGBA8 buffer the Swift side can
/// hand to CGImage / UIImage without further allocation.
typedef struct {
    int32_t  width;
    int32_t  height;
    int32_t  channels;                  ///< Always 4 (RGBA).
    uint8_t* pixels;                    ///< Owned by Kiln; freed by `mirage_free_image`.
} mirage_image;

// MARK: - Lifecycle

/// Load the given model files into a new engine context. Returns NULL on
/// failure (and writes a human-readable reason to `mirage_last_error`).
mirage_ctx* mirage_ctx_create(const mirage_model_paths* paths);

/// Tear down an engine context and release its weights.
void mirage_ctx_free(mirage_ctx* ctx);

// MARK: - Generation

/// Run the diffusion sampler against `params` and return a heap-allocated
/// `mirage_image*`. NULL on failure. Caller frees with `mirage_free_image`.
mirage_image* mirage_generate(mirage_ctx* ctx, const mirage_gen_params* params);

/// Release an image returned by `mirage_generate`.
void mirage_free_image(mirage_image* img);

// MARK: - Diagnostics

/// Human-readable description of the last failure on this thread. Empty
/// string if no error has been recorded. Lifetime: until the next Kiln call
/// on this thread.
const char* mirage_last_error(void);

/// Engine version, in the format "MAJOR.MINOR.PATCH" — bumped on breaking
/// changes to the C ABI above.
const char* mirage_version(void);

// MARK: - Progress callback

/// Called by the sampler once per denoising step. `step` is 1-indexed (1..steps),
/// `total` is the configured `steps`, `time_s` is the elapsed seconds since the
/// previous step (the first call reports cumulative warm-up + step 1 time).
/// Fires on the engine's worker thread — bounce to your UI actor before
/// touching any view state.
typedef void (*mirage_progress_cb)(int32_t step, int32_t total, float time_s, void* user_data);

/// Install a global progress callback. Pass NULL to clear. `user_data` is
/// forwarded verbatim to each call. Safe to set before `mirage_ctx_create`.
void mirage_set_progress_callback(mirage_progress_cb cb, void* user_data);

#ifdef __cplusplus
}
#endif

#endif // MIRAGE_C_H

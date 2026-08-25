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
    const char* llm_path;               ///< LLM-style text encoder GGUF (Qwen3-4B for Z-Image, …).
    const char* t5xxl_path;             ///< T5-family text encoder GGUF (umt5-xxl for Wan video, T5 for SD3/Flux).
                                        ///< Distinct from `llm_path`: sd.cpp binds them to different tensor prefixes.
    const char* taesd_path;             ///< Optional tiny autoencoder (TAESD/TAEHV, e.g. taew2_2 for Wan 2.2).
                                        ///< When set it replaces the full VAE for decode — ~60× smaller file,
                                        ///< seconds instead of minutes, softer texture.
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

// MARK: - Video generation

/// Inputs to one video-generation call (Wan-family models). `init_image_*`
/// enables image-to-video: pass a tightly-packed RGB8 buffer to animate a
/// still; leave `init_image_pixels` NULL for pure text-to-video.
typedef struct {
    const char* prompt;                 ///< User prompt, UTF-8.
    const char* negative_prompt;        ///< Optional negative prompt. NULL = none.
    int32_t width;                      ///< Frame width in pixels (multiple of 16 for Wan's VAE).
    int32_t height;                     ///< Frame height in pixels (multiple of 16).
    int32_t frames;                     ///< Frame count. Wan requires 4n+1 (13, 17, …, 33).
    int32_t steps;                      ///< Sampling steps. Wan 2.2 5B: 15-25.
    float   cfg_scale;                  ///< CFG scale. Wan 2.2 TI2V 5B: 6.0.
    float   flow_shift;                 ///< Flow-matching shift. Wan: 3.0. <= 0 keeps sd.cpp's default.
    int64_t seed;                       ///< RNG seed. -1 picks a random one.
    const uint8_t* init_image_pixels;   ///< Optional RGB8 init image for image-to-video. NULL = text-to-video.
    int32_t init_image_width;
    int32_t init_image_height;
    bool    vae_tiling;                 ///< Tile the VAE decode to cap peak memory (recommended on iPhone).
    int32_t vae_tile_size;              ///< Latent-space tile edge when tiling (0 = engine default, 32).
                                        ///< Smaller tiles trade decode speed for a lower peak allocation:
                                        ///< 480x832x13 measured ~12.7 GB at default tiles on CPU.
} mirage_video_params;

/// A generated clip as `frame_count` tightly-packed RGB8 frames laid out
/// back-to-back in one buffer (`frame_count * width * height * channels`
/// bytes). Encode to a movie container on the Swift side (AVAssetWriter).
typedef struct {
    int32_t  width;
    int32_t  height;
    int32_t  channels;                  ///< 3 (RGB) for video decode.
    int32_t  frame_count;
    uint8_t* pixels;                    ///< Owned by Mirage; freed by `mirage_free_video`.
} mirage_video;

/// True if the loaded model can run `mirage_generate_video` (Wan / SVD
/// family). Image-only models return false.
bool mirage_supports_video(mirage_ctx* ctx);

/// Run the video sampler and return a heap-allocated `mirage_video*`.
/// NULL on failure (see `mirage_last_error`). Caller frees with
/// `mirage_free_video`. Progress callbacks fire once per denoising step.
mirage_video* mirage_generate_video(mirage_ctx* ctx, const mirage_video_params* params);

/// Release a clip returned by `mirage_generate_video`.
void mirage_free_video(mirage_video* video);

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

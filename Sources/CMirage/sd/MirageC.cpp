//
//  MirageC.cpp
//  Thin wrapper around stable-diffusion.cpp's C++ API, exposed via the
//  C ABI declared in MirageC.h.
//
//  Memory ownership rules:
//    - `mirage_ctx` is heap-allocated; freed by `mirage_ctx_free`.
//    - `mirage_image` and its `pixels` buffer are heap-allocated; freed by
//      `mirage_free_image`.
//    - Strings inside `mirage_model_paths` / `mirage_gen_params` are borrowed
//      from the caller and must outlive the call (they're not retained).
//

#include "MirageC.h"
#include <TargetConditionals.h>

// stable-diffusion.cpp pulls in <stable-diffusion.h> which declares the
// `sd_ctx_t` opaque type and the `new_sd_ctx`, `txt2img`, etc. entry points.
#include "stable-diffusion.h"
#include "ggml.h"

#include <cstring>
#include <cstdlib>
#include <string>
#include <vector>
#include <mutex>

// MARK: - Thread-local error buffer

namespace {

thread_local std::string g_last_error;

void set_last_error(const char* msg) {
    g_last_error = msg ? msg : "";
}

} // namespace

// MARK: - Engine context

struct mirage_ctx {
    sd_ctx_t* sd = nullptr;
};

namespace {

// Funnel every sd.cpp / ggml log line into the iOS device console with a
// recognisable prefix. Without this, GGML's error logs (compile failures,
// shape mismatches, etc.) go to plain stderr and get lost in the noise
// from sd.cpp's progress bars on iPhone.
void mirage_sd_log_cb(enum sd_log_level_t level, const char* text, void* /*data*/) {
    if (!text) return;
    const char* tag = "info";
    switch (level) {
        case SD_LOG_ERROR: tag = "ERR "; break;
        case SD_LOG_WARN:  tag = "WARN"; break;
        case SD_LOG_INFO:  tag = "info"; break;
        case SD_LOG_DEBUG: tag = "dbg "; break;
        case SD_LOG_VERBOSE: tag = "verb"; break;
    }
    fprintf(stderr, "[mirage sd %s] %s%s", tag, text,
            (text[0] && text[strlen(text)-1] == '\n') ? "" : "\n");
    fflush(stderr);
}

void mirage_ggml_log_cb(enum ggml_log_level level, const char* text, void* /*data*/) {
    if (!text) return;
    const char* tag = "info";
    switch (level) {
        case GGML_LOG_LEVEL_ERROR: tag = "ERR "; break;
        case GGML_LOG_LEVEL_WARN:  tag = "WARN"; break;
        case GGML_LOG_LEVEL_INFO:  tag = "info"; break;
        case GGML_LOG_LEVEL_DEBUG: tag = "dbg "; break;
        default: break;
    }
    fprintf(stderr, "[mirage ggml %s] %s%s", tag, text,
            (text[0] && text[strlen(text)-1] == '\n') ? "" : "\n");
    fflush(stderr);
}

bool g_log_cb_installed = false;

} // namespace

extern "C" mirage_ctx* mirage_ctx_create(const mirage_model_paths* paths) {
    if (!paths || !paths->diffusion_model_path) {
        set_last_error("mirage_ctx_create: diffusion_model_path is required");
        return nullptr;
    }

    if (!g_log_cb_installed) {
        sd_set_log_callback(mirage_sd_log_cb, nullptr);
        ggml_log_set(mirage_ggml_log_cb, nullptr);
        g_log_cb_installed = true;
    }

#if TARGET_OS_IPHONE
    // Split Metal graphs across command buffers on iOS. With the default
    // single command buffer, one video-DiT sampling step runs the GPU for
    // tens of seconds uninterrupted — the iOS GPU watchdog kills it (seen
    // as a whole-device crash mid-generation on iPhone 17 Pro Max). Eight
    // shorter buffers keep each submission comfortably under the watchdog.
    // setenv-without-overwrite so a debug override still wins.
    setenv("GGML_METAL_N_CB", "32", 0);
#endif

    // Build a default sd_ctx_params and override only what we expose.
    sd_ctx_params_t p;
    sd_ctx_params_init(&p);
    p.diffusion_model_path = paths->diffusion_model_path;
    if (paths->vae_path) { p.vae_path = paths->vae_path; }
    if (paths->llm_path) { p.llm_path = paths->llm_path; }
    if (paths->t5xxl_path) { p.t5xxl_path = paths->t5xxl_path; }
    if (paths->taesd_path) { p.taesd_path = paths->taesd_path; }

    // Memory model (upstream 2026-09 residency rework): module placement is
    // expressed as assignment strings instead of the old boolean knobs.
    //
    // `backend` = where each module's COMPUTE runs. Unlisted modules use the
    // default backend (Metal on Apple silicon). The text encoder and VAE run
    // on CPU — same proven split as the old keep_clip_on_cpu/keep_vae_on_cpu
    // config: TE runs once per generation (latency hidden by sampling) and
    // the tiny-TAE video decode is CPU-fast, while the GPU budget stays
    // dedicated to the diffusion model.
    //
    // `params_backend` = where module WEIGHTS live between ops. "cpu" is the
    // old offload_params_to_cpu=true: weights held in (mmap-backed) host
    // memory, streamed to Metal per-op — the stable config on iPhone's 6 GB
    // process cap. On iOS the text encoder goes one step further with
    // "te=disk": its weights stream straight from disk for the one
    // conditioning pass and are never resident during sampling. This
    // replaces the old fork-only free_cond_stage_immediately patch — and
    // unlike that patch, the engine stays reusable after conditioning.
    //
    // `enable_mmap = true`: cuts peak load memory from ~2x weights to ~1x.
    // Required on iPhone or jetsam kills the app during weight load.
    //
    // `diffusion_flash_attn` + `diffusion_conv_direct`: reduce attention +
    // conv working memory.
    p.backend        = "te=cpu,vae=cpu,controlnet=cpu";
#if TARGET_OS_IPHONE
    p.params_backend = "all=cpu,te=disk";
#else
    p.params_backend = "cpu";
#endif
    p.enable_mmap           = true;
    p.diffusion_flash_attn  = true;
    p.diffusion_conv_direct = true;

    // Log the resolved params before handing them to sd.cpp so we can verify
    // from the device console which knobs actually took effect.
    if (char* dump = sd_ctx_params_to_str(&p)) {
        fprintf(stderr, "[mirage] sd_ctx_params resolved:\n%s\n", dump);
        free(dump);
    }

    sd_ctx_t* sd = new_sd_ctx(&p);
    if (!sd) {
        set_last_error("new_sd_ctx returned NULL — model failed to load (check paths + quantization compatibility)");
        return nullptr;
    }

    auto* ctx = new mirage_ctx();
    ctx->sd = sd;
    return ctx;
}

extern "C" void mirage_ctx_free(mirage_ctx* ctx) {
    if (!ctx) return;
    if (ctx->sd) free_sd_ctx(ctx->sd);
    delete ctx;
}

// MARK: - Generation

extern "C" mirage_image* mirage_generate(mirage_ctx* ctx, const mirage_gen_params* params) {
    if (!ctx || !ctx->sd) {
        set_last_error("mirage_generate: invalid context");
        return nullptr;
    }
    if (!params || !params->prompt) {
        set_last_error("mirage_generate: prompt is required");
        return nullptr;
    }

    sd_img_gen_params_t g;
    sd_img_gen_params_init(&g);
    g.prompt = params->prompt;
    g.negative_prompt = params->negative_prompt ? params->negative_prompt : "";
    g.width = params->width  > 0 ? params->width  : 1024;
    g.height = params->height > 0 ? params->height : 1024;
    g.sample_params.sample_steps = params->steps > 0 ? params->steps : 9;
    g.sample_params.guidance.txt_cfg = params->cfg_scale > 0 ? params->cfg_scale : 1.0f;
    g.sample_params.sample_method = EULER_SAMPLE_METHOD;
    g.seed = params->seed;
    g.batch_count = params->batch_size > 0 ? params->batch_size : 1;

    sd_image_t* result = nullptr;
    int num_images = 0;
    if (!generate_image(ctx->sd, &g, &result, &num_images) || !result || num_images <= 0) {
        if (result) std::free(result);
        set_last_error("generate_image failed");
        return nullptr;
    }

    auto* img = static_cast<mirage_image*>(std::malloc(sizeof(mirage_image)));
    if (!img) {
        set_last_error("mirage_generate: out of memory allocating image struct");
        for (int i = 0; i < num_images; ++i) {
            if (result[i].data) std::free(result[i].data);
        }
        std::free(result);
        return nullptr;
    }

    img->width = result[0].width;
    img->height = result[0].height;
    img->channels = result[0].channel;
    const size_t bytes = static_cast<size_t>(img->width) *
                         static_cast<size_t>(img->height) *
                         static_cast<size_t>(img->channels);
    img->pixels = static_cast<uint8_t*>(std::malloc(bytes));
    if (!img->pixels) {
        set_last_error("mirage_generate: out of memory allocating pixel buffer");
        for (int i = 0; i < num_images; ++i) {
            if (result[i].data) std::free(result[i].data);
        }
        std::free(result);
        std::free(img);
        return nullptr;
    }
    std::memcpy(img->pixels, result[0].data, bytes);

    for (int i = 0; i < num_images; ++i) {
        if (result[i].data) std::free(result[i].data);
    }
    std::free(result);

    return img;
}

extern "C" void mirage_free_image(mirage_image* img) {
    if (!img) return;
    if (img->pixels) std::free(img->pixels);
    std::free(img);
}

// MARK: - Video generation

extern "C" bool mirage_supports_video(mirage_ctx* ctx) {
    if (!ctx || !ctx->sd) return false;
    return sd_ctx_supports_video_generation(ctx->sd);
}

extern "C" mirage_video* mirage_generate_video(mirage_ctx* ctx, const mirage_video_params* params) {
    if (!ctx || !ctx->sd) {
        set_last_error("mirage_generate_video: invalid context");
        return nullptr;
    }
    if (!params || !params->prompt) {
        set_last_error("mirage_generate_video: prompt is required");
        return nullptr;
    }
    if (!sd_ctx_supports_video_generation(ctx->sd)) {
        set_last_error("mirage_generate_video: loaded model does not support video generation");
        return nullptr;
    }

    sd_vid_gen_params_t v;
    sd_vid_gen_params_init(&v);
    v.prompt          = params->prompt;
    v.negative_prompt = params->negative_prompt ? params->negative_prompt : "";
    v.width           = params->width  > 0 ? params->width  : 480;
    v.height          = params->height > 0 ? params->height : 320;
    // Wan's VAE compresses time 4x — the frame count must be 4n+1. Round
    // down to the nearest legal count rather than failing the whole run.
    int32_t frames = params->frames > 0 ? params->frames : 13;
    if (frames > 1 && (frames - 1) % 4 != 0) {
        frames = ((frames - 1) / 4) * 4 + 1;
    }
    v.video_frames = frames;
    v.sample_params.sample_steps     = params->steps > 0 ? params->steps : 20;
    v.sample_params.guidance.txt_cfg = params->cfg_scale > 0 ? params->cfg_scale : 6.0f;
    v.sample_params.sample_method    = EULER_SAMPLE_METHOD;
    if (params->flow_shift > 0) {
        v.sample_params.flow_shift = params->flow_shift;
    }
    v.seed = params->seed;

    // Optional init image → image-to-video. sd.cpp borrows the pixel
    // buffer for the duration of the call, matching our borrow contract.
    if (params->init_image_pixels &&
        params->init_image_width > 0 && params->init_image_height > 0) {
        v.init_image.width   = static_cast<uint32_t>(params->init_image_width);
        v.init_image.height  = static_cast<uint32_t>(params->init_image_height);
        v.init_image.channel = 3;
        v.init_image.data    = const_cast<uint8_t*>(params->init_image_pixels);
    }

    // Tiled VAE decode caps the largest single allocation of the whole
    // pipeline (decoding the full frame stack at once). Essential on iPhone.
    if (params->vae_tiling) {
        v.vae_tiling_params.enabled = true;
        if (params->vae_tile_size > 0) {
            v.vae_tiling_params.tile_size_x = params->vae_tile_size;
            v.vae_tiling_params.tile_size_y = params->vae_tile_size;
        }
    }

    int num_frames = 0;
    sd_image_t* result = nullptr;
    // audio_out=nullptr is null-guarded upstream: models with an audio
    // branch (MiniMax H3) simply skip audio decode.
    if (!generate_video(ctx->sd, &v, &result, &num_frames, nullptr) ||
        !result || num_frames <= 0) {
        if (result) std::free(result);
        set_last_error("generate_video failed");
        return nullptr;
    }

    const int32_t w = static_cast<int32_t>(result[0].width);
    const int32_t h = static_cast<int32_t>(result[0].height);
    const int32_t c = static_cast<int32_t>(result[0].channel);
    const size_t frame_bytes = static_cast<size_t>(w) * h * c;

    auto free_result = [&]() {
        for (int i = 0; i < num_frames; ++i) {
            if (result[i].data) std::free(result[i].data);
        }
        std::free(result);
    };

    auto* video = static_cast<mirage_video*>(std::malloc(sizeof(mirage_video)));
    if (!video) {
        set_last_error("mirage_generate_video: out of memory allocating video struct");
        free_result();
        return nullptr;
    }
    video->width       = w;
    video->height      = h;
    video->channels    = c;
    video->frame_count = num_frames;
    video->pixels      = static_cast<uint8_t*>(std::malloc(frame_bytes * num_frames));
    if (!video->pixels) {
        set_last_error("mirage_generate_video: out of memory allocating frame buffer");
        free_result();
        std::free(video);
        return nullptr;
    }
    for (int i = 0; i < num_frames; ++i) {
        // Frames should be homogeneous; guard anyway so a short frame can't
        // read out of bounds.
        if (result[i].data &&
            result[i].width == result[0].width &&
            result[i].height == result[0].height &&
            result[i].channel == result[0].channel) {
            std::memcpy(video->pixels + frame_bytes * i, result[i].data, frame_bytes);
        } else {
            std::memset(video->pixels + frame_bytes * i, 0, frame_bytes);
        }
    }
    free_result();
    return video;
}

extern "C" void mirage_free_video(mirage_video* video) {
    if (!video) return;
    if (video->pixels) std::free(video->pixels);
    std::free(video);
}

// MARK: - Diagnostics

extern "C" const char* mirage_last_error(void) {
    return g_last_error.c_str();
}

extern "C" const char* mirage_version(void) {
    return "0.3.0";
}

// MARK: - Progress callback

namespace {

mirage_progress_cb g_progress_cb = nullptr;
void*              g_progress_user_data = nullptr;

void mirage_sd_progress_trampoline(int step, int steps, float time_s, void* /*data*/) {
    if (g_progress_cb) {
        g_progress_cb(step, steps, time_s, g_progress_user_data);
    }
}

} // namespace

extern "C" void mirage_set_progress_callback(mirage_progress_cb cb, void* user_data) {
    g_progress_cb = cb;
    g_progress_user_data = user_data;
    // sd.cpp accepts NULL to clear too.
    sd_set_progress_callback(cb ? mirage_sd_progress_trampoline : nullptr, nullptr);
}


// MARK: - Upscaling (ESRGAN)
//
// Thin bridge over sd.cpp's upscaler_ctx_t. The Swift side traffics in
// RGBA8 (matching mirage_image); sd.cpp's ESRGAN wants RGB8, so both
// directions convert here, keeping the ABI a single pixel format.

struct mirage_upscaler {
    upscaler_ctx_t* ctx;
};

extern "C" mirage_upscaler* mirage_upscaler_create(const char* esrgan_path, int32_t tile_size) {
    if (!esrgan_path) return nullptr;
    // Default backend placement (Metal): ESRGAN weights are ~65MB — keep
    // them resident for speed. direct=false: standard conv path.
    upscaler_ctx_t* ctx = new_upscaler_ctx(esrgan_path,
                                           /*direct=*/false,
                                           /*n_threads=*/-1,
                                           tile_size,
                                           /*backend=*/nullptr,
                                           /*params_backend=*/nullptr);
    if (!ctx) {
        set_last_error("new_upscaler_ctx returned NULL — bad path or unsupported arch");
        return nullptr;
    }
    auto* up = new mirage_upscaler{ctx};
    return up;
}

extern "C" void mirage_upscaler_free(mirage_upscaler* up) {
    if (!up) return;
    if (up->ctx) free_upscaler_ctx(up->ctx);
    delete up;
}

extern "C" mirage_image* mirage_upscale(mirage_upscaler* up,
                                        const uint8_t* rgba,
                                        int32_t width,
                                        int32_t height,
                                        int32_t factor) {
    if (!up || !up->ctx || !rgba || width <= 0 || height <= 0) return nullptr;

    // RGBA8 → RGB8 for sd.cpp.
    const size_t n = (size_t)width * height;
    uint8_t* rgb = (uint8_t*)malloc(n * 3);
    if (!rgb) return nullptr;
    for (size_t i = 0; i < n; i++) {
        rgb[i * 3 + 0] = rgba[i * 4 + 0];
        rgb[i * 3 + 1] = rgba[i * 4 + 1];
        rgb[i * 3 + 2] = rgba[i * 4 + 2];
    }
    sd_image_t in{ (uint32_t)width, (uint32_t)height, 3, rgb };

    sd_image_t* outs = nullptr;
    int num_out = 0;
    bool ok = upscale(up->ctx, in, (uint32_t)factor, &outs, &num_out);
    free(rgb);
    if (!ok || !outs || num_out <= 0 || !outs[0].data) {
        if (outs) free(outs);
        set_last_error("upscale failed");
        return nullptr;
    }
    sd_image_t out = outs[0];

    // RGB8 → RGBA8 mirage_image (Swift hands this straight to CGImage).
    auto* img = (mirage_image*)malloc(sizeof(mirage_image));
    const size_t on = (size_t)out.width * out.height;
    img->width = (int32_t)out.width;
    img->height = (int32_t)out.height;
    img->channels = 4;
    img->pixels = (uint8_t*)malloc(on * 4);
    for (size_t i = 0; i < on; i++) {
        img->pixels[i * 4 + 0] = out.data[i * 3 + 0];
        img->pixels[i * 4 + 1] = out.data[i * 3 + 1];
        img->pixels[i * 4 + 2] = out.data[i * 3 + 2];
        img->pixels[i * 4 + 3] = 255;
    }
    free(out.data);
    for (int i = 1; i < num_out; i++) {
        if (outs[i].data) free(outs[i].data);
    }
    free(outs);
    return img;
}

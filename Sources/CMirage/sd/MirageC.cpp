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

// stable-diffusion.cpp pulls in <stable-diffusion.h> which declares the
// `sd_ctx_t` opaque type and the `new_sd_ctx`, `txt2img`, etc. entry points.
#include "stable-diffusion.h"

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

extern "C" mirage_ctx* mirage_ctx_create(const mirage_model_paths* paths) {
    if (!paths || !paths->diffusion_model_path) {
        set_last_error("mirage_ctx_create: diffusion_model_path is required");
        return nullptr;
    }

    // Build a default sd_ctx_params and override only what we expose.
    sd_ctx_params_t p;
    sd_ctx_params_init(&p);
    p.diffusion_model_path = paths->diffusion_model_path;
    if (paths->vae_path) { p.vae_path = paths->vae_path; }
    if (paths->llm_path) { p.llm_path = paths->llm_path; }

    // Memory-saver knobs that matter on iPhone.
    p.keep_clip_on_cpu = true;
    p.keep_control_net_on_cpu = true;
    p.keep_vae_on_cpu = false;
    p.diffusion_flash_attn = true;

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

    sd_image_t* result = generate_image(ctx->sd, &g);
    if (!result) {
        set_last_error("generate_image returned NULL");
        return nullptr;
    }

    auto* img = static_cast<mirage_image*>(std::malloc(sizeof(mirage_image)));
    if (!img) {
        set_last_error("mirage_generate: out of memory allocating image struct");
        for (int i = 0; i < g.batch_count; ++i) {
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
        for (int i = 0; i < g.batch_count; ++i) {
            if (result[i].data) std::free(result[i].data);
        }
        std::free(result);
        std::free(img);
        return nullptr;
    }
    std::memcpy(img->pixels, result[0].data, bytes);

    for (int i = 0; i < g.batch_count; ++i) {
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

// MARK: - Diagnostics

extern "C" const char* mirage_last_error(void) {
    return g_last_error.c_str();
}

extern "C" const char* mirage_version(void) {
    return "0.1.0";
}

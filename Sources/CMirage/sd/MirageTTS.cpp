//
//  MirageTTS.cpp
//  Speech synthesis half of the C bridge: Qwen3-TTS through qwentts.cpp.
//
//  qwentts.cpp is a GGML port of Qwen3-TTS 12 Hz. It is compiled into the
//  same partially-linked object as sd.cpp, so it runs on Mirage's copy of
//  ggml (hidden from Kuzco's) and needs no second Metal runtime.
//

#include "MirageC.h"

#include "qwen.h"

#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

struct mirage_tts {
    qt_context* q = nullptr;
};

namespace {

thread_local std::string g_tts_error;

void tts_log_cb(enum qt_log_level level, const char* msg, void* /*data*/) {
    if (!msg || level < QT_LOG_INFO) return;
    const char* tag = level == QT_LOG_ERROR ? "ERR " : level == QT_LOG_WARN ? "WARN" : "info";
    fprintf(stderr, "[mirage tts %s] %s\n", tag, msg);
}

struct chunk_relay {
    mirage_tts_chunk_cb cb;
    void* user_data;
};

bool relay_chunk(const float* samples, int n_samples, void* user_data) {
    auto* relay = static_cast<chunk_relay*>(user_data);
    return relay->cb(samples, n_samples, relay->user_data);
}

} // namespace

extern "C" mirage_tts* mirage_tts_create(const char* talker_path, const char* codec_path) {
    if (!talker_path || !codec_path) {
        g_tts_error = "mirage_tts_create: talker_path and codec_path are required";
        return nullptr;
    }
    qt_log_set(tts_log_cb, nullptr);

    qt_init_params p;
    qt_init_default_params(&p);
    p.talker_path = talker_path;
    p.codec_path = codec_path;
    // Short codec chunks bound the decode's peak memory on iPhone; the
    // streaming path frames its own chunks and ignores this.
    p.codec_chunk_sec = 6.0f;

    qt_context* q = qt_init(&p);
    if (!q) {
        const char* e = qt_last_error();
        g_tts_error = std::string("qt_init failed: ") + (e ? e : "unknown error");
        return nullptr;
    }
    auto* tts = new mirage_tts();
    tts->q = q;
    return tts;
}

extern "C" void mirage_tts_free(mirage_tts* tts) {
    if (!tts) return;
    if (tts->q) qt_free(tts->q);
    delete tts;
}

extern "C" int32_t mirage_tts_speaker_count(mirage_tts* tts) {
    return tts ? qt_n_speakers(tts->q) : 0;
}

extern "C" const char* mirage_tts_speaker_name(mirage_tts* tts, int32_t index) {
    return tts ? qt_speaker_name(tts->q, index) : nullptr;
}

extern "C" int32_t mirage_tts_language_count(mirage_tts* tts) {
    return tts ? qt_n_languages(tts->q) : 0;
}

extern "C" const char* mirage_tts_language_name(mirage_tts* tts, int32_t index) {
    return tts ? qt_language_name(tts->q, index) : nullptr;
}

extern "C" int32_t mirage_tts_sample_rate(void) {
    return 24000;
}

extern "C" bool mirage_tts_synthesize(mirage_tts* tts,
                                      const mirage_tts_params* params,
                                      mirage_tts_chunk_cb on_chunk,
                                      void* user_data) {
    if (!tts || !tts->q || !params || !params->text || !params->text[0] || !on_chunk) {
        g_tts_error = "mirage_tts_synthesize: engine, text and on_chunk are required";
        return false;
    }

    qt_tts_params p;
    qt_tts_default_params(&p);
    p.text = params->text;
    p.lang = params->language;
    p.speaker = params->speaker;
    p.instruct = params->instruct;
    p.seed = params->seed;

    chunk_relay relay{on_chunk, user_data};
    p.on_chunk = relay_chunk;
    p.on_chunk_user_data = &relay;

    qt_audio out = {};
    const qt_status status = qt_synthesize(tts->q, &p, &out);
    qt_audio_free(&out);
    if (status != QT_STATUS_OK) {
        const char* e = qt_last_error();
        g_tts_error = std::string("qt_synthesize failed: ") + (e ? e : "unknown error");
        return false;
    }
    return true;
}

extern "C" const char* mirage_tts_last_error(void) {
    return g_tts_error.c_str();
}

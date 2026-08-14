#include "text_encoder.h"
#include <functional>
#include <android/log.h>

#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, "TextEncoder", __VA_ARGS__)

TextEncoder::TextEncoder() : loaded_(false), embedding_dim_(768), max_tokens_(77) {}
TextEncoder::~TextEncoder() { release(); }

void TextEncoder::load(const std::string& path) {
    LOGI("Loading TextEncoder from: %s", path.c_str());
    // TODO: 真实加载 CLIP/T5 权重文件
    loaded_ = true;
}

std::vector<float> TextEncoder::encode(const std::string& text) {
    // Simplified pseudo-embedding using std::hash
    // Real impl would use BPE tokenizer + transformer inference
    std::vector<float> embedding(embedding_dim_);

    size_t hash = std::hash<std::string>{}(text);
    for (int i = 0; i < embedding_dim_; i++) {
        hash = hash * 1103515245ULL + 12345;
        embedding[i] = ((hash & 0x7FFF) / 16384.0f) - 1.0f;
    }

    return embedding;
}

void TextEncoder::release() {
    if (loaded_) {
        LOGI("Releasing TextEncoder");
        loaded_ = false;
    }
}

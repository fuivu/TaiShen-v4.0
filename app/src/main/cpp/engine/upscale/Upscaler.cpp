#include "Upscaler.h"
#include <android/log.h>
#include <chrono>
#include <cmath>
#include <algorithm>
#include <vector>

#define TAG "Upscaler"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO,  TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, TAG, __VA_ARGS__)

namespace localai {
namespace upscale {

// ============ PImpl ============

struct Upscaler::Impl {
    std::vector<float> rrdb_weights;  // RRDB 块权重
    std::vector<float> upsampler_weights;
    int num_rrdb = 23;  // RRDB 块数量
    float alpha = 0.2f;  // residual scaling
};

// ============ 构造/析构 ============

Upscaler::Upscaler() : pImpl_(std::make_unique<Impl>()) {}
Upscaler::~Upscaler() = default;

// ============ 模型加载 ============

bool Upscaler::load(const std::string& model_path, const Config& config) {
    config_ = config;
    model_name_ = model_path.substr(model_path.find_last_of("/") + 1);
    LOGI("Loading Upscaler: model=%s scale=%d tile=%d", model_name_.c_str(), config.scale, config.tile_size);

    // stub: 分配 RRDB 权重
    int params_per_block = 64 * 64 * 3; // 简化
    pImpl_->rrdb_weights.assign(pImpl_->num_rrdb * params_per_block, 0.0f);
    pImpl_->upsampler_weights.assign(64 * 3 * 3 * 3, 0.0f); // 上采样卷积

    srand(42);
    for (auto& w : pImpl_->rrdb_weights) {
        w = ((float)rand()/RAND_MAX - 0.5f) * 0.02f;
    }
    loaded_ = true;
    LOGI("Upscaler loaded: %d RRDB blocks", pImpl_->num_rrdb);
    return true;
}

void Upscaler::unload() {
    if (pImpl_) {
        pImpl_->rrdb_weights.clear();
        pImpl_->upsampler_weights.clear();
    }
    loaded_ = false;
    LOGI("Upscaler unloaded");
}

// ============ Bicubic 上采样 (fallback) ============

std::vector<uint8_t> Upscaler::bicubicUpsample(const std::vector<uint8_t>& img, int w, int h, int scale) {
    int ow = w * scale, oh = h * scale;
    std::vector<uint8_t> out(ow * oh * 3);
    for (int c = 0; c < 3; ++c) {
        for (int oy = 0; oy < oh; ++oy) {
            for (int ox = 0; ox < ow; ++ox) {
                float fx = (float)ox / scale;
                float fy = (float)oy / scale;
                int x0 = (int)floorf(fx - 1);
                int y0 = (int)floorf(fy - 1);
                float dx = fx - x0 - 1;
                float dy = fy - y0 - 1;

                float acc = 0;
                float wsum = 0;
                for (int j = 0; j < 4; ++j) {
                    for (int i = 0; i < 4; ++i) {
                        int xi = std::max(0, std::min(w-1, x0+i));
                        int yi = std::max(0, std::min(h-1, y0+j));
                        // bicubic 权重 (简化)
                        float wx = 1.0f - fabsf(dx - i);
                        float wy = 1.0f - fabsf(dy - j);
                        float weight = wx * wy;
                        acc += img[(yi*w+xi)*3+c] * weight;
                        wsum += weight;
                    }
                }
                int v = (int)(acc / std::max(wsum, 1e-5f));
                out[(oy*ow+ox)*3+c] = (uint8_t)std::min(255, std::max(0, v));
            }
        }
    }
    return out;
}

// ============ 单块推理 (stub) ============

std::vector<uint8_t> Upscaler::inferTile(const std::vector<uint8_t>& tile, int tw, int th) {
    // 实际应运行 RRDB 网络
    // stub: 用 bicubic 代替
    return bicubicUpsample(tile, tw, th, config_.scale);
}

// ============ 羽化混合 ============

void Upscaler::applyFeather(std::vector<uint8_t>& output, int w, int h, int feather) {
    // 对边缘做简单平滑 (stub)
    for (int c = 0; c < 3; ++c) {
        for (int y = 0; y < std::min(feather, h); ++y) {
            float alpha = (float)y / feather;
            for (int x = 0; x < w; ++x) {
                int idx = (y*w+x)*3+c;
                output[idx] = (uint8_t)(output[idx] * alpha);
            }
        }
    }
}

// ============ 分块融合 ============

std::vector<uint8_t> Upscaler::blendTiles(const std::vector<std::vector<uint8_t>>& tiles,
                                            const std::vector<int>& offsets,
                                            int out_w, int out_h, int overlap) {
    std::vector<uint8_t> out(out_w * out_h * 3, 0);
    std::vector<float> weight_map(out_w * out_h, 0);

    int tile_idx = 0;
    for (size_t t = 0; t < tiles.size(); ++t) {
        int ox = offsets[t*2];
        int oy = offsets[t*2+1];
        int tw = offsets[t*2+2];
        int th = offsets[t*2+3];
        const auto& tile = tiles[t];

        for (int y = 0; y < th; ++y) {
            for (int x = 0; x < tw; ++x) {
                int dx = ox + x;
                int dy = oy + y;
                if (dx >= out_w || dy >= out_h) continue;

                // 羽化权重：离边缘越远权重越大
                float wx = 1.0f, wy = 1.0f;
                if (overlap > 0) {
                    if (x < overlap) wx = (float)x / overlap;
                    if (y < overlap) wy = (float)y / overlap;
                    int xr = tw - 1 - x;
                    int yr = th - 1 - y;
                    if (xr < overlap && xr >= 0) wx = std::max(wx, (float)xr / overlap);
                    if (yr < overlap && yr >= 0) wy = std::max(wy, (float)yr / overlap);
                }
                float w = wx * wy;

                int di = (dy*out_w+dx)*3;
                int si = (y*tw+x)*3;
                for (int c = 0; c < 3; ++c) {
                    out[di+c] += (uint8_t)(tile[si+c] * w);
                }
                weight_map[dy*out_w+dx] += w;
            }
        }
    }

    // 归一化
    for (int i = 0; i < out_w*out_h; ++i) {
        if (weight_map[i] > 0) {
            float inv = 1.0f / weight_map[i];
            out[i*3]   = (uint8_t)(out[i*3]   * inv);
            out[i*3+1] = (uint8_t)(out[i*3+1] * inv);
            out[i*3+2] = (uint8_t)(out[i*3+2] * inv);
        }
    }
    return out;
}

// ============ 分块超分 ============

std::vector<uint8_t> Upscaler::upscaleTiled(const std::vector<uint8_t>& image, int w, int h) {
    int tile = config_.tile_size;
    int overlap = config_.tile_overlap;
    int step = tile - overlap;
    int ow = w * config_.scale;
    int oh = h * config_.scale;

    std::vector<std::vector<uint8_t>> tiles;
    std::vector<int> offsets;

    for (int y = 0; y < h; y += step) {
        for (int x = 0; x < w; x += step) {
            int tw = std::min(tile, w - x);
            int th = std::min(tile, h - y);
            // 裁剪
            std::vector<uint8_t> crop(tw*th*3);
            for (int cy = 0; cy < th; ++cy) {
                for (int cx = 0; cx < tw; ++cx) {
                    int sx = x+cx, sy = y+cy;
                    crop[(cy*tw+cx)*3]   = image[(sy*w+sx)*3];
                    crop[(cy*tw+cx)*3+1] = image[(sy*w+sx)*3+1];
                    crop[(cy*tw+cx)*3+2] = image[(sy*w+sx)*3+2];
                }
            }
            auto up = inferTile(crop, tw, th);
            tiles.push_back(std::move(up));
            offsets.push_back(x * config_.scale);   // ox
            offsets.push_back(y * config_.scale);   // oy
            offsets.push_back(tw * config_.scale);  // tw_out
            offsets.push_back(th * config_.scale);  // th_out
        }
    }

    LOGI("Tiled upscale: %d tiles, output %dx%d", (int)tiles.size(), ow, oh);
    return blendTiles(tiles, offsets, ow, oh, overlap * config_.scale);
}

// ============ 公开接口 ============

std::vector<uint8_t> Upscaler::upscale(const std::vector<uint8_t>& image, int w, int h) {
    if (!loaded_) {
        LOGE("upscale() called but model not loaded!");
        return image;
    }

    auto t0 = std::chrono::high_resolution_clock::now();

    int total_pixels = w * h;
    int threshold = config_.tile_size * config_.tile_size;

    std::vector<uint8_t> result;
    if (total_pixels > threshold && config_.tile_mode != TileMode::NONE) {
        result = upscaleTiled(image, w, h);
    } else {
        result = inferTile(image, w, h);
    }

    auto t1 = std::chrono::high_resolution_clock::now();
    last_infer_time_ms_ = std::chrono::duration<float, std::milli>(t1-t0).count();

    int ow = w * config_.scale, oh = h * config_.scale;
    LOGI("Upscale done: %dx%d → %dx%d, %.1f ms", w, h, ow, oh, last_infer_time_ms_);
    return result;
}

} // namespace upscale
} // namespace localai

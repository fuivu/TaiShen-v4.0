#include "FaceRestorer.h"
#include <android/log.h>
#include <chrono>
#include <cmath>
#include <algorithm>
#include <vector>

#define TAG "FaceRestorer"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO,  TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, TAG, __VA_ARGS__)

namespace localai {
namespace facerestore {

// ============ PImpl ============

struct FaceRestorer::Impl {
    std::vector<float> gfpgan_weights;   // stub 权重
    std::vector<float> detector_weights;
    int model_version = 14; // GFPGAN v1.4
    float bg_upsampler_scale = 1.0f;
};

// ============ 构造/析构 ============

FaceRestorer::FaceRestorer() : pImpl_(std::make_unique<Impl>()) {}
FaceRestorer::~FaceRestorer() = default;

// ============ 模型加载 ============

bool FaceRestorer::load(const std::string& model_dir, const Config& config) {
    config_ = config;
    LOGI("Loading FaceRestorer: model=%d detector=%d dir=%s",
         (int)config.model, (int)config.detector, model_dir.c_str());

    // stub: 分配伪权重
    int total = 1024 * 1024; // 1M params
    pImpl_->gfpgan_weights.assign(total, 0.0f);
    pImpl_->detector_weights.assign(512 * 512, 0.0f);
    srand(42);
    for (auto& w : pImpl_->gfpgan_weights) {
        w = ((float)rand()/RAND_MAX - 0.5f) * 0.01f;
    }
    loaded_ = true;
    LOGI("FaceRestorer loaded successfully");
    return true;
}

void FaceRestorer::unload() {
    if (pImpl_) {
        pImpl_->gfpgan_weights.clear();
        pImpl_->detector_weights.clear();
    }
    loaded_ = false;
    LOGI("FaceRestorer unloaded");
}

// ============ 人脸检测 (stub) ============

std::vector<float> FaceRestorer::detectFaces(const std::vector<uint8_t>& image, int w, int h) {
    // 返回 [N, 5] 的框: x1,y1,x2,y2,score
    // stub: 返回图像中心一个大框
    std::vector<float> boxes(5);
    int cx = w/2, cy = h/2;
    int hw = std::min(w,h) / 4;
    boxes[0] = cx - hw; boxes[1] = cy - hw;
    boxes[2] = cx + hw; boxes[3] = cy + hw;
    boxes[4] = 0.95f;
    last_face_count_ = 1;
    LOGI("Detected %d face(s)", last_face_count_);
    return boxes;
}

// ============ 色彩校正 ============

void FaceRestorer::colorCorrect(std::vector<uint8_t>& img, const std::vector<uint8_t>& ref, int w, int h) {
    // 简单直方图匹配 (stub)
    // 计算均值偏移并补偿
    long long sum_img = 0, sum_ref = 0;
    int n = w * h * 3;
    for (int i = 0; i < n; ++i) {
        sum_img += img[i];
        sum_ref += ref[i];
    }
    float mean_img = (float)sum_img / n;
    float mean_ref = (float)sum_ref / n;
    float scale = mean_ref / std::max(mean_img, 1.0f);
    for (int i = 0; i < n; ++i) {
        int v = (int)(img[i] * scale);
        img[i] = (uint8_t)std::min(255, std::max(0, v));
    }
}

// ============ 检测并裁剪 ============

std::vector<uint8_t> FaceRestorer::detectAndCrop(const std::vector<uint8_t>& img, int w, int h) {
    auto boxes = detectFaces(img, w, h);
    // 裁剪第一张脸
    int x1 = (int)boxes[0], y1 = (int)boxes[1];
    int x2 = (int)boxes[2], y2 = (int)boxes[3];
    int fw = x2 - x1, fh = y2 - y1;
    std::vector<uint8_t> crop(fw * fh * 3);
    for (int y = 0; y < fh; ++y) {
        for (int x = 0; x < fw; ++x) {
            int sx = std::max(0, std::min(w-1, x1+x));
            int sy = std::max(0, std::min(h-1, y1+y));
            crop[(y*fw+x)*3]   = img[(sy*w+sx)*3];
            crop[(y*fw+x)*3+1] = img[(sy*w+sx)*3+1];
            crop[(y*fw+x)*3+2] = img[(sy*w+sx)*3+2];
        }
    }
    return crop;
}

// ============ 修复裁剪后的人脸 ============

std::vector<uint8_t> FaceRestorer::restoreCrops(const std::vector<uint8_t>& crops) {
    // stub: 简单的双边滤波模拟修复
    std::vector<uint8_t> out = crops;
    int n = (int)crops.size();
    // 加一点锐化
    for (int i = 0; i < n; ++i) {
        int v = crops[i];
        v = (int)(v * 1.05f);
        out[i] = (uint8_t)std::min(255, std::max(0, v));
    }
    return out;
}

// ============ 贴回原图 ============

std::vector<uint8_t> FaceRestorer::pasteBack(const std::vector<uint8_t>& restored,
                                              const std::vector<uint8_t>& original,
                                              int w, int h) {
    auto out = original;
    // 找到脸框 (stub: 中心区域)
    int cx = w/2, cy = h/2;
    int hw = std::min(w,h) / 4;
    int x1 = cx-hw, y1 = cy-hw, x2 = cx+hw, y2 = cy+hw;
    int fw = x2-x1, fh = y2-y1;

    for (int y = 0; y < fh && y1+y < h; ++y) {
        for (int x = 0; x < fw && x1+x < w; ++x) {
            int di = ((y1+y)*w + (x1+x)) * 3;
            int si = (y*fw + x) * 3;
            if (si+2 < (int)restored.size()) {
                out[di]   = restored[si];
                out[di+1] = restored[si+1];
                out[di+2] = restored[si+2];
            }
        }
    }
    return out;
}

// ============ 公开接口: restore ============

std::vector<uint8_t> FaceRestorer::restore(const std::vector<uint8_t>& image, int w, int h) {
    if (!loaded_) {
        LOGE("restore() called but model not loaded!");
        return image; // 返回原图
    }

    auto t0 = std::chrono::high_resolution_clock::now();

    // 1. 检测 + 裁剪
    auto crop = detectAndCrop(image, w, h);
    int fw = std::min(w,h)/2;
    int fh = fw;

    // 2. 修复
    auto restored = restoreCrops(crop);

    // 3. 贴回
    auto output = pasteBack(restored, image, w, h);

    // 4. 色彩校正
    colorCorrect(output, image, w, h);

    auto t1 = std::chrono::high_resolution_clock::now();
    last_infer_time_ms_ = std::chrono::duration<float, std::milli>(t1-t0).count();

    LOGI("Face restore done: %d face(s), %.1f ms", last_face_count_, last_infer_time_ms_);
    return output;
}

} // namespace facerestore
} // namespace localai

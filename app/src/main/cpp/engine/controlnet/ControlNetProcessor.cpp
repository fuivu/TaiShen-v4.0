#include "ControlNetProcessor.h"
#include <android/log.h>
#include <cmath>
#include <algorithm>
#include <vector>

#define TAG "ControlNet"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO,  TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, TAG, __VA_ARGS__)

namespace localai {
namespace controlnet {

// ============ PImpl ============

struct ControlNetProcessor::Impl {
    // 模型权重（简化 stub）
    std::vector<float> conv_weights;
    int model_loaded = 0;
};

// ============ 构造/析构 ============

ControlNetProcessor::ControlNetProcessor() : pImpl_(std::make_unique<Impl>()) {}
ControlNetProcessor::~ControlNetProcessor() = default;

// ============ 模型加载 ============

bool ControlNetProcessor::loadModel(ControlType type, const std::string& model_path) {
    type_ = type;
    model_path_ = model_path;
    LOGI("Loading ControlNet model: type=%d path=%s", (int)type, model_path.c_str());

    // stub: 分配伪权重
    int kernel_size = 64 * 64;
    pImpl_->conv_weights.assign(kernel_size, 0.0f);
    srand(42);
    for (auto& w : pImpl_->conv_weights) {
        w = ((float)rand() / RAND_MAX - 0.5f) * 0.02f;
    }
    pImpl_->model_loaded = 1;
    loaded_ = true;
    return true;
}

void ControlNetProcessor::unload() {
    if (pImpl_) {
        pImpl_->conv_weights.clear();
        pImpl_->model_loaded = 0;
    }
    loaded_ = false;
    LOGI("ControlNet unloaded");
}

// ============ RGB → Gray ============

std::vector<float> ControlNetProcessor::rgbToGray(const std::vector<float>& rgb, int w, int h) {
    std::vector<float> gray(w * h);
    for (int i = 0; i < w * h; ++i) {
        gray[i] = 0.299f * rgb[i*3] + 0.587f * rgb[i*3+1] + 0.114f * rgb[i*3+2];
    }
    return gray;
}

// ============ Gaussian Blur ============

std::vector<float> ControlNetProcessor::gaussianBlur(const std::vector<float>& img, int w, int h, float sigma) {
    std::vector<float> out(w * h);
    int r = (int)ceilf(3 * sigma);
    std::vector<float> kernel(2*r+1);
    float sum = 0;
    for (int i = -r; i <= r; ++i) {
        kernel[i+r] = expf(-i*i / (2*sigma*sigma));
        sum += kernel[i+r];
    }
    for (auto& k : kernel) k /= sum;

    // 水平
    std::vector<float> tmp(w * h);
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            float acc = 0;
            for (int k = -r; k <= r; ++k) {
                int xi = std::max(0, std::min(w-1, x+k));
                acc += img[y*w + xi] * kernel[k+r];
            }
            tmp[y*w + x] = acc;
        }
    }
    // 垂直
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            float acc = 0;
            for (int k = -r; k <= r; ++k) {
                int yi = std::max(0, std::min(h-1, y+k));
                acc += tmp[yi*w + x] * kernel[k+r];
            }
            out[y*w + x] = acc;
        }
    }
    return out;
}

// ============ Sobel 边缘 ============

std::vector<float> ControlNetProcessor::sobel(const std::vector<float>& gray, int w, int h) {
    std::vector<float> dx(w*h, 0), dy(w*h, 0);
    for (int y = 1; y < h-1; ++y) {
        for (int x = 1; x < w-1; ++x) {
            dx[y*w+x] = gray[y*w+(x+1)] - gray[y*w+(x-1)];
            dy[y*w+x] = gray[(y+1)*w+x] - gray[(y-1)*w+x];
        }
    }
    std::vector<float> mag(w*h);
    for (int i = 0; i < w*h; ++i) {
        mag[i] = sqrtf(dx[i]*dx[i] + dy[i]*dy[i]);
    }
    return mag;
}

// ============ Canny ============

std::vector<float> ControlNetProcessor::cannyDetect(const std::vector<float>& image, int w, int h) {
    auto gray = rgbToGray(image, w, h);
    gray = gaussianBlur(gray, w, h, 1.4f);
    auto mag = sobel(gray, w, h);

    // 双阈值
    std::vector<float> edges(w*h, 0);
    float low  = canny_low_  / 255.0f;
    float high = canny_high_ / 255.0f;
    for (int i = 0; i < w*h; ++i) {
        if (mag[i] >= high)       edges[i] = 1.0f;
        else if (mag[i] >= low)  edges[i] = 0.5f;
    }
    // 简单非极大值抑制 + 边缘连接 (stub)
    LOGI("Canny edges: low=%.1f high=%.1f", low, high);
    return edges; // 单通道
}

// ============ OpenPose (stub) ============

std::vector<float> ControlNetProcessor::openPoseDetect(const std::vector<float>& image, int w, int h) {
    // 实际应运行 OpenPose 模型
    // stub: 返回零图
    std::vector<float> pose(w * h * 3, 0.0f);
    LOGI("OpenPose: returning empty pose map");
    return pose;
}

// ============ Depth (stub) ============

std::vector<float> ControlNetProcessor::depthEstimate(const std::vector<float>& image, int w, int h) {
    auto gray = rgbToGray(image, w, h);
    // 反转作为伪深度
    std::vector<float> depth(w*h);
    for (int i = 0; i < w*h; ++i) depth[i] = 1.0f - gray[i];
    LOGI("Depth: simple inversion pseudo-depth");
    return depth;
}

// ============ Scribble (stub) ============

std::vector<float> ControlNetProcessor::scribbleExtract(const std::vector<float>& image, int w, int h) {
    // 提取高饱和度区域
    std::vector<float> scrib(w*h*3);
    for (int i = 0; i < w*h; ++i) {
        float r = image[i*3], g = image[i*3+1], b = image[i*3+2];
        float maxv = std::max({r,g,b});
        float minv = std::min({r,g,b});
        float sat = (maxv > 0) ? (maxv - minv) / maxv : 0;
        if (sat > 0.5f) {
            scrib[i*3] = r; scrib[i*3+1] = g; scrib[i*3+2] = b;
        }
    }
    return scrib;
}

// ============ MLSD (stub) ============

std::vector<float> ControlNetProcessor::mlsdDetect(const std::vector<float>& image, int w, int h) {
    auto gray = rgbToGray(image, w, h);
    auto edges = sobel(gray, w, h);
    LOGI("MLSD: line detection stub (using Sobel)");
    return edges;
}

// ============ Seg (stub) ============

std::vector<float> ControlNetProcessor::segDetect(const std::vector<float>& image, int w, int h) {
    // 简化：按亮度分桶
    std::vector<float> seg(w*h*3, 0);
    for (int i = 0; i < w*h; ++i) {
        float lum = 0.299f*image[i*3] + 0.587f*image[i*3+1] + 0.114f*image[i*3+2];
        int cls = (int)(lum * 10) % 10;
        seg[i*3]   = (cls & 1) ? 1.0f : 0.0f;
        seg[i*3+1] = (cls & 2) ? 1.0f : 0.0f;
        seg[i*3+2] = (cls & 4) ? 1.0f : 0.0f;
    }
    return seg;
}

// ============ 公开接口 ============

std::vector<float> ControlNetProcessor::preprocess(const std::vector<float>& image, int w, int h) {
    if (!loaded_) {
        LOGE("preprocess() called but model not loaded!");
        return {};
    }
    LOGI("ControlNet preprocess: type=%d size=%dx%d strength=%.2f",
         (int)type_, w, h, strength_);
    switch (type_) {
        case ControlType::CANNY:    return cannyDetect(image, w, h);
        case ControlType::OPENPOSE: return openPoseDetect(image, w, h);
        case ControlType::DEPTH:    return depthEstimate(image, w, h);
        case ControlType::SCRIBBLE: return scribbleExtract(image, w, h);
        case ControlType::MLSD:     return mlsdDetect(image, w, h);
        case ControlType::SEG:      return segDetect(image, w, h);
        default: return cannyDetect(image, w, h);
    }
}

} // namespace controlnet
} // namespace localai

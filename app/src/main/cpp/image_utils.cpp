#include "image_utils.h"
#include <cmath>

namespace image {

void float_to_rgb(const float* data, unsigned char* rgb, int width, int height, int channels) {
    int total = width * height * channels;
    for (int i = 0; i < total; i++) {
        float v = data[i];
        v = (v + 1.0f) * 127.5f; // [-1,1] → [0,255]
        v = fmaxf(0.0f, fminf(255.0f, v));
        rgb[i] = (unsigned char)(v + 0.5f);
    }
}

void rgb_to_float(const unsigned char* rgb, float* data, int width, int height, int channels) {
    int total = width * height * channels;
    for (int i = 0; i < total; i++) {
        data[i] = (float)rgb[i] / 127.5f - 1.0f;
    }
}

void resize_bilinear(const float* src, float* dst, int sw, int sh, int dw, int dh, int channels) {
    float x_ratio = (float)sw / dw;
    float y_ratio = (float)sh / dh;
    
    for (int y = 0; y < dh; y++) {
        for (int x = 0; x < dw; x++) {
            float sx = x * x_ratio;
            float sy = y * y_ratio;
            int x0 = (int)sx, y0 = (int)sy;
            int x1 = (x0 + 1 < sw) ? x0 + 1 : x0;
            int y1 = (y0 + 1 < sh) ? y0 + 1 : y0;
            float fx = sx - x0, fy = sy - y0;
            
            for (int c = 0; c < channels; c++) {
                float v00 = src[(y0 * sw + x0) * channels + c];
                float v01 = src[(y0 * sw + x1) * channels + c];
                float v10 = src[(y1 * sw + x0) * channels + c];
                float v11 = src[(y1 * sw + x1) * channels + c];
                
                float v0 = v00 * (1 - fx) + v01 * fx;
                float v1 = v10 * (1 - fx) + v11 * fx;
                dst[(y * dw + x) * channels + c] = v0 * (1 - fy) + v1 * fy;
            }
        }
    }
}

void apply_vae_scaling(float* data, size_t count) {
    const float scale = 0.18215f;
    for (size_t i = 0; i < count; i++) {
        data[i] *= scale;
    }
}

void remove_vae_scaling(float* data, size_t count) {
    const float inv_scale = 1.0f / 0.18215f;
    for (size_t i = 0; i < count; i++) {
        data[i] *= inv_scale;
    }
}

} // namespace image

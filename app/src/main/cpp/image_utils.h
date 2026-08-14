#pragma once
#include <vector>

namespace image {

void float_to_rgb(const float* data, unsigned char* rgb, int width, int height, int channels = 3);
void rgb_to_float(const unsigned char* rgb, float* data, int width, int height, int channels = 3);
void resize_bilinear(const float* src, float* dst, int sw, int sh, int dw, int dh, int channels);
void apply_vae_scaling(float* data, size_t count);  // x * 0.18215
void remove_vae_scaling(float* data, size_t count); // x / 0.18215

} // namespace image

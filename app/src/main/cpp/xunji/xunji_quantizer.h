/*
 * ══════════════════════════════════════════════════════════════
 *  迅集 (XunJi) — 量化核心  v4.0 TaiShen
 *  7 种精度量化 · 统一 API · 安全宏
 * ══════════════════════════════════════════════════════════════
 */
#pragma once
#include <cstdint>
#include <vector>
#include <string>
#include <cmath>
#include <limits>
#include <algorithm>

namespace xunji {

// ─── 精度枚举 ────────────────────────────────────────────────
enum class QuantLevel : int {
    FP32 = 0, FP16, BF16, INT8, INT4, INT2, FP8
};

const char* quant_level_name(QuantLevel lvl);
int         quant_bits(QuantLevel lvl);
float       quant_mem_ratio(QuantLevel lvl);

// ─── FP32 ↔ FP16 ─────────────────────────────────────────────
uint16_t fp32_to_fp16(float v);
float    fp16_to_fp32(uint16_t h);
void     quantize_fp16(const float* src, uint16_t* dst, size_t n);
void     dequantize_fp16(const uint16_t* src, float* dst, size_t n);

// ─── FP32 ↔ BF16 ─────────────────────────────────────────────
uint16_t fp32_to_bf16(float v);
float    bf16_to_fp32(uint16_t b);
void     quantize_bf16(const float* src, uint16_t* dst, size_t n);
void     dequantize_bf16(const uint16_t* src, float* dst, size_t n);

// ─── FP32 ↔ INT8 (对称/非对称) ──────────────────────────────
struct Int8Params { float scale; int zero_point; float min_val; float max_val; };
void     compute_int8_params(const float* src, size_t n, Int8Params* p);
void     quantize_int8(const float* src, int8_t* dst, size_t n, const Int8Params& p);
void     dequantize_int8(const int8_t* src, float* dst, size_t n, const Int8Params& p);

// ─── FP32 ↔ INT4 (2 值/字节, 符号扩展) ──────────────────────
struct Int4Params { float scale; int zero_point; };
void     compute_int4_params(const float* src, size_t n, Int4Params* p);
void     quantize_int4(const float* src, uint8_t* dst_packed, size_t n, const Int4Params& p);
void     dequantize_int4(const uint8_t* src_packed, float* dst, size_t n, const Int4Params& p);

// ─── FP32 ↔ INT2 (4 值/字节) ────────────────────────────────
struct Int2Params { float scale; int zero_point; };
void     compute_int2_params(const float* src, size_t n, Int2Params* p);
void     quantize_int2(const float* src, uint8_t* dst_packed, size_t n, const Int2Params& p);
void     dequantize_int2(const uint8_t* src_packed, float* dst, size_t n, const Int2Params& p);

// ─── FP32 ↔ FP8 (E4M3) ──────────────────────────────────────
uint8_t  fp32_to_fp8_e4m3(float v);
float    fp8_e4m3_to_fp32(uint8_t e);
void     quantize_fp8(const float* src, uint8_t* dst, size_t n);
void     dequantize_fp8(const uint8_t* src, float* dst, size_t n);

// ─── 统一分发 ─────────────────────────────────────────────────
struct QuantResult {
    std::vector<uint8_t> data;
    std::string           type;       // "fp16","int8",...
    size_t                elem_count;
    Int8Params            int8_p;     // 有效当 type=="int8"
    Int4Params            int4_p;
    Int2Params            int2_p;
};

bool quantize_uniform(const float* src, size_t n, QuantLevel lvl, QuantResult* out);
bool dequantize_uniform(const QuantResult& in, float* dst);

} // namespace xunji

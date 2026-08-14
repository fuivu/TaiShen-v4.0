#pragma once
/**
 * Local AI Painter v4.0 TaiShen — C API 头文件
 * 供 JNI / Android / iOS / Desktop 统一调用
 */
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// ═══════════════════════════════════════════════
//  版本
// ═══════════════════════════════════════════════

const char* locai_v35_version(); // "4.0.0 TaiShen"

// ═══════════════════════════════════════════════
//  引擎生命周期
// ═══════════════════════════════════════════════

void* locai_v35_create();   // 创建并初始化引擎
void  locai_v35_destroy(void* engine);
int   locai_v35_initialize(void* engine);
void  locai_v35_shutdown(void* engine);

// ═══════════════════════════════════════════════
//  推理
// ═══════════════════════════════════════════════

int locai_v35_run_inference(
    void* engine,
    const char* model_path,
    const char* prompt,
    const char* neg_prompt,
    int    width,
    int    height,
    int    steps,
    float  cfg_scale,
    int    seed,
    const char* sampler,       // "Euler"/"Euler_a"/"DPM++2M"/"LCM"/etc
    const char* quantization,   // "auto"/"INT2"/"INT4"/"INT8"/"FP8"/"FP16"
    const char* backend,       // "auto"/"vulkan"/"opengl"/"npu"/"cpu"/"hybrid"
    int    use_cim,           // 0/1 (天玑 CIM)
    int    use_spd_plus,      // 0/1 (投机解码+)
    float* output_image,       // [width*height*4] RGBA float in [-1,1]
    int*   out_width,
    int*   out_height,
    double* out_time_ms
);

// ═══════════════════════════════════════════════
//  取消
// ═══════════════════════════════════════════════

void locai_v35_cancel(void* engine);

// ═══════════════════════════════════════════════
//  进度回调
// ═══════════════════════════════════════════════

typedef void (*LocaiProgressCb)(int step, int total, float progress, double eta_ms, void* userdata);
void locai_v35_set_progress_callback(void* engine, LocaiProgressCb cb, void* userdata);

// ═══════════════════════════════════════════════
//  能力查询
// ═══════════════════════════════════════════════

void locai_v35_get_caps(void* engine, char* out_buffer, int buffer_size);

#ifdef __cplusplus
} // extern "C"
#endif

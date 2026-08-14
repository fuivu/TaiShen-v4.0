/**
 * ============================================================================
 *  JniDimensity8400.cpp
 *  ────────────────────────────────────────────────────────────────────────────
 *  JNI 桥接：Kotlin ↔ C++ Dimensity8400Adapter
 *
 *  包名：com.localaipainter.engine.mediatek
 *  类  ：Dimensity8400Bridge
 *
 *  设计：所有 JNI 调用都通过全局单例 adapter 转发。
 *        dlopen 失败时 adapter 进入 STUB 模式，Java 层拿到 null / 0 / false，
 *        绝不崩溃。
 *  ============================================================================
 */

#include <jni.h>
#include <android/log.h>
#include <string>
#include <memory>

#include "Dimensity8400Adapter.h"

#define TAG "D8400-JNI"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO,  TAG, __VA_ARGS__)
#define LOGW(...) __android_log_print(ANDROID_LOG_WARN,  TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, TAG, __VA_ARGS__)

using namespace mediatek::dimensity8400;

// ══════════════════════════════════════════
//  全局单例
// ══════════════════════════════════════════

static std::unique_ptr<Dimensity8400Adapter> g_adapter;
static JavaVM* g_vm = nullptr;

// JNI_OnLoad
extern "C" JNIEXPORT jint JNICALL JNI_OnLoad(JavaVM* vm, void* reserved) {
    LOGI("JNI_OnLoad — Dimensity8400Adapter");
    g_vm = vm;
    return JNI_VERSION_1_6;
}

extern "C" JNIEXPORT void JNICALL JNI_OnUnload(JavaVM* vm, void* reserved) {
    LOGI("JNI_OnUnload");
    if (g_adapter) { g_adapter->Shutdown(); g_adapter.reset(); }
}

// ══════════════════════════════════════════
//  生命周期
// ══════════════════════════════════════════

extern "C" JNIEXPORT jboolean JNICALL
Java_com_localaipainter_engine_mediatek_Dimensity8400Bridge_nativeInit(
    JNIEnv* env, jclass /*clazz*/)
{
    if (!g_adapter) g_adapter = Dimensity8400Adapter::Create();
    bool ok = g_adapter->Initialize();
    LOGI("nativeInit → %s", ok ? "OK" : "STUB MODE");
    return ok ? JNI_TRUE : JNI_FALSE;
}

extern "C" JNIEXPORT void JNICALL
Java_com_localaipainter_engine_mediatek_Dimensity8400Bridge_nativeShutdown(
    JNIEnv* env, jclass /*clazz*/)
{
    if (g_adapter) { g_adapter->Shutdown(); g_adapter.reset(); }
    LOGI("nativeShutdown done");
}

extern "C" JNIEXPORT jboolean JNICALL
Java_com_localaipainter_engine_mediatek_Dimensity8400Bridge_nativeIsAvailable(
    JNIEnv* env, jclass /*clazz*/)
{
    return (g_adapter && g_adapter->IsAvailable()) ? JNI_TRUE : JNI_FALSE;
}

// ══════════════════════════════════════════
//  能力探测 → 返回 String（JSON 格式，Kotlin 侧解析）
// ══════════════════════════════════════════

extern "C" JNIEXPORT jstring JNICALL
Java_com_localaipainter_engine_mediatek_Dimensity8400Bridge_nativeGetCapabilities(
    JNIEnv* env, jclass /*clazz*/)
{
    if (!g_adapter) g_adapter = Dimensity8400Adapter::Create();
    auto caps = g_adapter->DetectCapabilities();

    // 构建简易 JSON（不引入 nlohmann/json，减少依赖）
    std::ostringstream j;
    j << "{";
    j << "\"soc\":\""       << caps.soc_model << "\",";
    j << "\"name\":\""      << caps.marketing_name << "\",";
    j << "\"npu_gen\":"     << caps.npu_generation << ",";
    j << "\"npu_avail\":"   << (caps.npu_available ? "true" : "false") << ",";
    j << "\"gpu\":\""       << caps.gpu_name << "\",";
    j << "\"gpu_shaders\":" << caps.gpu_shader_cores << ",";
    j << "\"gpu_freq\":"    << caps.gpu_freq_mhz << ",";
    j << "\"gpu_vk\":"     << (caps.gpu_supports_vulkan ? "true" : "false") << ",";
    j << "\"gpu_cl\":"     << (caps.gpu_supports_opencl ? "true" : "false") << ",";
    j << "\"ram_mb\":"     << caps.total_ram_mb << ",";
    j << "\"l3_kb\":"      << caps.l3_cache_kb << ",";
    j << "\"slc_kb\":"     << caps.slc_cache_kb << ",";
    j << "\"mem_mhz\":"    << caps.mem_freq_mhz << ",";
    j << "\"cpu_cores\":"   << caps.cpu_cores << ",";
    j << "\"cpu_max\":"     << caps.cpu_max_freq_khz << ",";
    j << "\"all_big\":"    << (caps.cpu_all_big_core ? "true" : "false") << ",";
    j << "\"int4\":"       << (caps.supports_int4_quant ? "true" : "false") << ",";
    j << "\"dit\":"        << (caps.supports_dit ? "true" : "false") << ",";
    j << "\"dae\":"        << (caps.supports_agentic_ai ? "true" : "false") << ",";
    j << "\"ufs4\":"       << (caps.supports_ufs4 ? "true" : "false") << ",";
    j << "\"rec_prec\":"   << (int)caps.recommended_precision << ",";
    j << "\"rec_gpu\":"    << (int)caps.recommended_gpu << ",";
    j << "\"rec_threads\":"<< caps.recommended_threads << ",";
    j << "\"rec_tile\":"   << caps.recommended_tile_size << ",";
    j << "\"est_sd15\":"   << caps.estimated_sd15_speed;
    j << "}";

    return env->NewStringUTF(j.str().c_str());
}

// ══════════════════════════════════════════
//  版本信息
// ══════════════════════════════════════════

extern "C" JNIEXPORT jstring JNICALL
Java_com_localaipainter_engine_mediatek_Dimensity8400Bridge_nativeGetDriverVersion(
    JNIEnv* env, jclass /*clazz*/)
{
    if (!g_adapter) return env->NewStringUTF("not initialized");
    return env->NewStringUTF(g_adapter->GetDriverVersion().c_str());
}

extern "C" JNIEXPORT jstring JNICALL
Java_com_localaipainter_engine_mediatek_Dimensity8400Bridge_nativeGetSdkVersion(
    JNIEnv* env, jclass /*clazz*/)
{
    if (!g_adapter) return env->NewStringUTF("not initialized");
    return env->NewStringUTF(g_adapter->GetSdkVersion().c_str());
}

// ══════════════════════════════════════════
//  模型编译
// ══════════════════════════════════════════

extern "C" JNIEXPORT jobject JNICALL
Java_com_localaipainter_engine_mediatek_Dimensity8400Bridge_nativeCompileModel(
    JNIEnv* env, jclass /*clazz*/,
    jint    role,
    jstring src_path,
    jstring out_path,
    jint    precision,
    jint    width,
    jint    height,
    jint    batch_size)
{
    if (!g_adapter) g_adapter = Dimensity8400Adapter::Create();

    ModelCompileRequest req;
    req.role              = (ModelRole)(int)role;
    req.source_path       = env->GetStringUTFChars(src_path, nullptr);
    req.output_dla_path   = env->GetStringUTFChars(out_path, nullptr);
    req.target_precision  = (NpuPrecision)(int)precision;
    req.input_width       = (uint32_t)width;
    req.input_height      = (uint32_t)height;
    req.batch_size        = (uint32_t)batch_size;
    req.prefer_low_latency = true;

    auto res = g_adapter->CompileModel(req);

    // 返回 Kotlin data class ModelCompileResult 的字段映射
    // 用 jclass / jmethodID 构造对象
    jclass    cls  = env->FindClass("com/localaipainter/engine/mediatek/ModelCompileResult");
    jmethodID ctor = env->GetMethodID(cls, "<init>", "(ZLjava/lang/String;Ljava/lang/String;JFFI)V");
    jstring   jpath  = env->NewStringUTF(res.dla_path.c_str());
    jstring   jerr   = env->NewStringUTF(res.error_message.c_str());
    jobject   obj = env->NewObject(cls, ctor,
                                   res.success ? JNI_TRUE : JNI_FALSE,
                                   jpath, jerr,
                                   (jlong)res.compiled_size_bytes,
                                   (jfloat)res.compile_time_sec,
                                   (jfloat)0.0f, // runtime_memory placeholder
                                   (jint)(int)res.actual_precision);
    return obj;
}

// ══════════════════════════════════════════
//  缓存管理
// ══════════════════════════════════════════

extern "C" JNIEXPORT void JNICALL
Java_com_localaipainter_engine_mediatek_Dimensity8400Bridge_nativeConfigureCache(
    JNIEnv* env, jclass /*clazz*/,
    jboolean pin_l3, jboolean pin_slc,
    jint l3_mb, jint slc_mb,
    jboolean prefetch, jint prefetch_dist,
    jboolean kv_cache, jint kv_max_tokens)
{
    if (!g_adapter) return;
    WeightCacheStrategy s;
    s.pin_hot_weights_to_l3  = pin_l3   == JNI_TRUE;
    s.pin_hot_weights_to_slc = pin_slc  == JNI_TRUE;
    s.l3_cache_capacity_mb   = (uint32_t)l3_mb;
    s.slc_cache_capacity_mb  = (uint32_t)slc_mb;
    s.prefetch_next_block    = prefetch == JNI_TRUE;
    s.prefetch_distance      = (uint32_t)prefetch_dist;
    s.enable_kv_cache        = kv_cache == JNI_TRUE;
    s.kv_cache_max_tokens    = (uint32_t)kv_max_tokens;
    g_adapter->ConfigureWeightCache(s);
}

extern "C" JNIEXPORT void JNICALL
Java_com_localaipainter_engine_mediatek_Dimensity8400Bridge_nativeWarmupCache(
    JNIEnv* env, jclass /*clazz*/, jstring dla_path)
{
    if (!g_adapter) return;
    const char* p = env->GetStringUTFChars(dla_path, nullptr);
    g_adapter->WarmupCache(std::string(p));
    env->ReleaseStringUTFChars(dla_path, p);
}

extern "C" JNIEXPORT void JNICALL
Java_com_localaipainter_engine_mediatek_Dimensity8400Bridge_nativeClearCache(
    JNIEnv* env, jclass /*clazz*/)
{
    if (g_adapter) g_adapter->ClearCache();
}

// ══════════════════════════════════════════
//  性能监控
// ══════════════════════════════════════════

extern "C" JNIEXPORT jfloatArray JNICALL
Java_com_localaipainter_engine_mediatek_Dimensity8400Bridge_nativeGetPerfCounters(
    JNIEnv* env, jclass /*clazz*/)
{
    jfloatArray arr = env->NewFloatArray(6);
    if (!g_adapter) {
        float z[6] = {0};
        env->SetFloatArrayRegion(arr, 0, 6, z);
        return arr;
    }
    auto pc = g_adapter->GetPerfCounters();
    float v[6] = {
        (float)pc.npu_active_cycles,
        (float)pc.npu_idle_cycles,
        (float)pc.gpu_active_cycles,
        (float)pc.memory_bandwidth_gb_s,
        pc.avg_power_mw,
        pc.temperature_c
    };
    env->SetFloatArrayRegion(arr, 0, 6, v);
    return arr;
}

// ══════════════════════════════════════════
//  功耗 / 降频
// ══════════════════════════════════════════

extern "C" JNIEXPORT void JNICALL
Java_com_localaipainter_engine_mediatek_Dimensity8400Bridge_nativeOnThermalThrottling(
    JNIEnv* env, jclass /*clazz*/, jint level)
{
    if (g_adapter) g_adapter->OnThermalThrottling((uint32_t)level);
}

extern "C" JNIEXPORT void JNICALL
Java_com_localaipainter_engine_mediatek_Dimensity8400Bridge_nativeSetPowerMode(
    JNIEnv* env, jclass /*clazz*/, jint mode)
{
    if (!g_adapter) return;
    SessionConfig::PowerMode pm = (SessionConfig::PowerMode)(int)mode;
    g_adapter->SetPowerMode(pm);
}

// ══════════════════════════════════════════
//  工具：量化
// ══════════════════════════════════════════

extern "C" JNIEXPORT jfloatArray JNICALL
Java_com_localaipainter_engine_mediatek_Dimensity8400Bridge_nativeQuantizeToInt8(
    JNIEnv* env, jclass /*clazz*/, jfloatArray src)
{
    jsize count = env->GetArrayLength(src);
    auto floats = std::unique_ptr<float[]>(new float[count]);
    auto int8s  = std::unique_ptr<int8_t[]>(new int8_t[count]);
    env->GetFloatArrayRegion(src, 0, count, floats.get());

    float scale = 1.0f;
    quantize::QuantizeToInt8(floats.get(), int8s.get(), (size_t)count, &scale);

    // 返回 [count] int8 打包成 floatArray（Kotlin 侧解包）+ 最后一个元素 = scale
    jfloatArray out = env->NewFloatArray(count + 1);
    auto fout = std::unique_ptr<float[]>(new float[count + 1]);
    for (jsize i = 0; i < count; ++i) fout[i] = (float)int8s[i];
    fout[count] = scale;
    env->SetFloatArrayRegion(out, 0, count + 1, fout.get());
    return out;
}

extern "C" JNIEXPORT jlong JNICALL
Java_com_localaipainter_engine_mediatek_Dimensity8400Bridge_nativeEstimateQuantizedSize(
    JNIEnv* env, jclass /*clazz*/, jlong param_count, jint precision)
{
    return (jlong)quantize::EstimateQuantizedSize(
        (size_t)param_count, (NpuPrecision)(int)precision);
}

// ══════════════════════════════════════════
//  工具：速度估算
// ══════════════════════════════════════════

extern "C" JNIEXPORT jfloat JNICALL
Java_com_localaipainter_engine_mediatek_Dimensity8400Bridge_nativeEstimateSpeed(
    JNIEnv* env, jclass /*clazz*/,
    jint precision, jint steps, jint width, jint height)
{
    if (!g_adapter) g_adapter = Dimensity8400Adapter::Create();
    auto caps = g_adapter->DetectCapabilities();
    return EstimateSD15Speed(caps, (NpuPrecision)(int)precision,
                               (uint32_t)steps, (uint32_t)width, (uint32_t)height);
}

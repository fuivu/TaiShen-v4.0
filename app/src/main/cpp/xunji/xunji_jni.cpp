/*
 * ═════════════════════════════════════════════════════════════
 *  迅集 (XunJi) — JNI 实现  v4.0 TaiShen
 *  15 个 native 方法 · 量化 · 内存 · 监控
 * ═════════════════════════════════════════════════════════════
 */
#include "xunji_jni.h"
#include <string>
#include <unordered_map>
#include <mutex>

namespace {

xunji::QuantLevel jstring_to_level(JNIEnv* env, jstring jstr) {
    const char* c = env->GetStringUTFChars(jstr, nullptr);
    std::string s(c);
    env->ReleaseStringUTFChars(jstr, c);
    if (s=="FP32") return xunji::QuantLevel::FP32;
    if (s=="FP16") return xunji::QuantLevel::FP16;
    if (s=="BF16") return xunji::QuantLevel::BF16;
    if (s=="INT8") return xunji::QuantLevel::INT8;
    if (s=="INT4") return xunji::QuantLevel::INT4;
    if (s=="INT2") return xunji::QuantLevel::INT2;
    if (s=="FP8")  return xunji::QuantLevel::FP8;
    return xunji::QuantLevel::FP16;
}

// 权重句柄注册表
struct WeightHandle {
    void* data; size_t n; xunji::QuantLevel lvl;
};
std::unordered_map<jlong, WeightHandle> g_handles;
std::mutex g_hmu;
jlong g_next_handle = 1;

jlong register_handle(void* d, size_t n, xunji::QuantLevel l) {
    std::lock_guard<std::mutex> lk(g_hmu);
    jlong h = g_next_handle++;
    g_handles[h] = {d, n, l};
    return h;
}

} // anon

// ─── 量化 ─────────────────────────────────────────────────
extern "C" JNIEXPORT jint JNICALL
Java_com_localaipainter_xunji_XunJiQuantExecutor_nativeSetPrecision(
    JNIEnv* env, jobject, jstring level)
{
    auto lvl = jstring_to_level(env, level);
    // 全局精度标记（简化实现）
    return (jint)xunji::quant_bits(lvl);
}

extern "C" JNIEXPORT jlong JNICALL
Java_com_localaipainter_xunji_XunJiQuantExecutor_nativeLoadModel(
    JNIEnv* env, jobject, jstring path, jstring level)
{
    const char* cp = env->GetStringUTFChars(path, nullptr);
    std::string p(cp); env->ReleaseStringUTFChars(path, cp);
    auto lvl = jstring_to_level(env, level);

    // 模拟加载：分配一段内存代表权重
    size_t n = 1024 * 1024; // 1M 元素
    float* data = (float*)xunji::xunji_malloc(n * sizeof(float));
    if (!data) return 0;
    // 填充伪数据
    for (size_t i = 0; i < n; ++i) data[i] = (float)i / (float)n;

    // 量化
    xunji::QuantResult qr;
    xunji::quantize_uniform(data, n, lvl, &qr);
    xunji::xunji_free(data);

    void* qdata = xunji::xunji_malloc(qr.data.size());
    if (!qdata) return 0;
    memcpy(qdata, qr.data.data(), qr.data.size());

    return register_handle(qdata, qr.data.size(), lvl);
}

extern "C" JNIEXPORT void JNICALL
Java_com_localaipainter_xunji_XunJiQuantExecutor_nativeFreeHandle(
    JNIEnv*, jobject, jlong handle)
{
    std::lock_guard<std::mutex> lk(g_hmu);
    auto it = g_handles.find(handle);
    if (it != g_handles.end()) {
        xunji::xunji_free(it->second.data);
        g_handles.erase(it);
    }
}

extern "C" JNIEXPORT jfloatArray JNICALL
Java_com_localaipainter_xunji_XunJiQuantExecutor_nativeRun(
    JNIEnv* env, jobject, jlong handle, jfloatArray input)
{
    jsize n = env->GetArrayLength(input);
    jfloat* buf = env->GetFloatArrayElements(input, nullptr);

    // 反量化 → 处理 → 再量化（简化：直接返回输入）
    float* out = new float[n];
    for (jsize i = 0; i < n; ++i) out[i] = buf[i] * 0.95f; // 模拟推理

    env->ReleaseFloatArrayElements(input, buf, JNI_ABORT);

    jfloatArray result = env->NewFloatArray(n);
    env->SetFloatArrayRegion(result, 0, n, out);
    delete[] out;
    return result;
}

// ─── 自动选择 ────────────────────────────────────────────
extern "C" JNIEXPORT jstring JNICALL
Java_com_localaipainter_xunji_XunJiScheduler_nativeAutoSelect(
    JNIEnv* env, jobject, jlong totalKb, jlong availKb, jlong, jlong)
{
    float ratio = totalKb > 0 ? (float)availKb / (float)totalKb : 0.5f;
    xunji::QuantLevel pick;
    if (ratio > 0.7f)      pick = xunji::QuantLevel::FP16;
    else if (ratio > 0.5f) pick = xunji::QuantLevel::INT8;
    else if (ratio > 0.3f) pick = xunji::QuantLevel::INT4;
    else                    pick = xunji::QuantLevel::INT2;
    return env->NewStringUTF(xunji::quant_level_name(pick));
}

// ─── 内存 ────────────────────────────────────────────────
extern "C" JNIEXPORT jlong JNICALL
Java_com_localaipainter_xunji_XunJiMemoryMonitor_nativeSample(JNIEnv*, jobject) {
    return xunji::xunji_mem_available_kb();
}

extern "C" JNIEXPORT jint JNICALL
Java_com_localaipainter_xunji_XunJiMemoryMonitor_nativeGetPressure(JNIEnv*, jobject) {
    return (jint)(int)xunji::xunji_mem_pressure();
}

extern "C" JNIEXPORT jlong JNICALL
Java_com_localaipainter_xunji_XunJiMemoryMonitor_nativeAvailKb(JNIEnv*, jobject) {
    return xunji::xunji_mem_available_kb();
}

extern "C" JNIEXPORT void JNICALL
Java_com_localaipainter_xunji_XunJiMemoryMonitor_nativeStartMonitor(JNIEnv*, jobject) {
    xunji::xunji_start_monitor();
}

extern "C" JNIEXPORT void JNICALL
Java_com_localaipainter_xunji_XunJiMemoryMonitor_nativeStopMonitor(JNIEnv*, jobject) {
    xunji::xunji_stop_monitor();
}

// ─── 工具 ────────────────────────────────────────────────
extern "C" JNIEXPORT jint JNICALL
Java_com_localaipainter_xunji_XunJiQuantExecutor_nativeGetBits(
    JNIEnv* env, jobject, jstring level)
{
    return xunji::quant_bits(jstring_to_level(env, level));
}

extern "C" JNIEXPORT jfloat JNICALL
Java_com_localaipainter_xunji_XunJiQuantExecutor_nativeGetMemRatio(
    JNIEnv* env, jobject, jstring level)
{
    return xunji::quant_mem_ratio(jstring_to_level(env, level));
}

extern "C" JNIEXPORT void JNICALL
Java_com_localaipainter_xunji_XunJiMemoryMonitor_nativeReportLeaks(JNIEnv*, jobject) {
    xunji::xunji_report_leaks();
}

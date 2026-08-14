/*
 * ════════════════════════════════════════════════════════════
 *  混元 (HunYuan) — JNI 桥接  v4.0 TaiShen
 *  15 个 native 方法 · 算法调度 · 策略切换
 * ════════════════════════════════════════════════════════════
 */
#include <jni.h>
#include "../hunyuan/hunyuan_algorithms.h"
#include <chrono>
#include <cstring>

using namespace hunyuan;

extern "C" {

// ─── 引擎生命周期 ──────────────────────────────────────
JNIEXPORT jlong JNICALL Java_com_localaipainter_hunyuan_HunYuanEngine_nativeInit
  (JNIEnv*, jobject) {
    auto* e = &HybridEngine::instance();
    return (jlong)e;
}

JNIEXPORT void JNICALL Java_com_localaipainter_hunyuan_HunYuanEngine_nativeShutdown
  (JNIEnv*, jobject, jlong handle) {
    // 单例不销毁
}

// ─── 策略设置 ──────────────────────────────────────────
JNIEXPORT void JNICALL Java_com_localaipainter_hunyuan_HunYuanEngine_nativeSetStrategy
  (JNIEnv* env, jobject, jlong handle, jstring jstr) {
    const char* c = env->GetStringUTFChars(jstr, nullptr);
    std::string s(c); env->ReleaseStringUTFChars(jstr, c);
    Strategy st = STRAT_ADAPTIVE;
    if(s=="max_speed")st=STRAT_MAX_SPEED;
    else if(s=="max_quality")st=STRAT_MAX_QUALITY;
    else if(s=="power_save")st=STRAT_POWER_SAVE;
    else if(s=="balanced")st=STRAT_BALANCED;
    else if(s=="custom")st=STRAT_CUSTOM;
    ((HybridEngine*)handle)->set_strategy(st);
}

JNIEXPORT jstring JNICALL Java_com_localaipainter_hunyuan_HunYuanEngine_nativeGetStrategy
  (JNIEnv* env, jobject, jlong handle) {
    auto st = ((HybridEngine*)handle)->get_strategy();
    return env->NewStringUTF(strategy_name(st));
}

// ─── 自定义顺序 ────────────────────────────────────────
JNIEXPORT void JNICALL Java_com_localaipainter_hunyuan_HunYuanEngine_nativeSetCustomOrder
  (JNIEnv* env, jobject, jlong handle, jintArray jorder) {
    jsize n = env->GetArrayLength(jorder);
    jint* buf = env->GetIntArrayElements(jorder, nullptr);
    AlgoID order[ALGO_COUNT];
    int m = n < ALGO_COUNT ? n : ALGO_COUNT;
    for (int i = 0; i < m; ++i) order[i] = (AlgoID)(int)buf[i];
    env->ReleaseIntArrayElements(jorder, buf, JNI_ABORT);
    ((HybridEngine*)handle)->set_custom_order(order, m);
}

// ─── 设备能力 ──────────────────────────────────────────
JNIEXPORT void JNICALL Java_com_localaipainter_hunyuan_HunYuanEngine_nativeSetDeviceCap
  (JNIEnv* env, jobject, jlong handle, jint cores, jfloat freq, jint ramGb,
   jboolean npu, jboolean gpu, jboolean vulkan, jint socTier) {
    DeviceCap c; c.cpu_cores=cores; c.cpu_freq_ghz=freq; c.ram_gb=ramGb;
    c.has_npu=npu; c.has_gpu=gpu; c.has_vulkan=vulkan; c.soc_tier=socTier;
    ((HybridEngine*)handle)->set_device_cap(c);
}

// ─── 性能喂入 ──────────────────────────────────────────
JNIEXPORT void JNICALL Java_com_localaipainter_hunyuan_HunYuanEngine_nativeFeedPerf
  (JNIEnv* env, jobject, jlong handle, jfloat cpuU, jfloat cpuT, jfloat memR,
   jfloat batP, jfloat batT, jboolean charging, jfloat gpuU, jlong availKb) {
    PerfSnapshot s; s.cpu_usage=cpuU; s.cpu_temp_c=cpuT; s.mem_used_ratio=memR;
    s.battery_pct=batP; s.battery_temp=batT; s.charging=charging; s.gpu_usage=gpuU; s.avail_mem_kb=availKb;
    ((HybridEngine*)handle)->feed_perf(s);
}

// ─── 核心：选择 + 执行 ────────────────────────────────
JNIEXPORT jint JNICALL Java_com_localaipainter_hunyuan_HunYuanEngine_nativePick
  (JNIEnv* env, jobject, jlong handle) {
    return (jint)((HybridEngine*)handle)->pick_algorithm();
}

JNIEXPORT jfloatArray JNICALL Java_com_localaipainter_hunyuan_HunYuanEngine_nativeExecute
  (JNIEnv* env, jobject, jlong handle, jfloatArray input) {
    jsize n = env->GetArrayLength(input);
    jfloat* buf = env->GetFloatArrayElements(input, nullptr);
    float* out = new float[n];
    bool ok = ((HybridEngine*)handle)->execute(buf, n, out);
    env->ReleaseFloatArrayElements(input, buf, JNI_ABORT);
    if (!ok) { delete[] out; return nullptr; }
    jfloatArray result = env->NewFloatArray(n);
    env->SetFloatArrayRegion(result, 0, n, out);
    delete[] out;
    return result;
}

// ─── 状态 / 统计 ───────────────────────────────────────
JNIEXPORT jstring JNICALL Java_com_localaipainter_hunyuan_HunYuanEngine_nativeStatus
  (JNIEnv* env, jobject, jlong handle) {
    auto s = ((HybridEngine*)handle)->status_json();
    return env->NewStringUTF(s.c_str());
}

JNIEXPORT void JNICALL Java_com_localaipainter_hunyuan_HunYuanEngine_nativeResetStats
  (JNIEnv*, jobject, jlong handle) {
    ((HybridEngine*)handle)->reset_stats();
}

// ─── 算法元数据查询 ────────────────────────────────────
JNIEXPORT jint JNICALL Java_com_localaipainter_hunyuan_HunYuanEngine_nativeAlgoCount
  (JNIEnv*, jobject) { return ALGO_COUNT; }

JNIEXPORT jstring JNICALL Java_com_localaipainter_hunyuan_HunYuanEngine_nativeAlgoName
  (JNIEnv* env, jobject, jint id) {
    return env->NewStringUTF(algo_name((AlgoID)(int)id));
}

JNIEXPORT jfloat JNICALL Java_com_localaipainter_hunyuan_HunYuanEngine_nativeAlgoQuality
  (JNIEnv*, jobject, jint id) { return algo_quality_score((AlgoID)(int)id); }

JNIEXPORT jfloat JNICALL Java_com_localaipainter_hunyuan_HunYuanEngine_nativeAlgoSpeed
  (JNIEnv*, jobject, jint id) { return (jfloat)algo_speed_tier((AlgoID)(int)id); }

JNIEXPORT jfloat JNICALL Java_com_localaipainter_hunyuan_HunYuanEngine_nativeAlgoPower
  (JNIEnv*, jobject, jint id) { return algo_power_eff((AlgoID)(int)id); }

// ─── 排序查询 ───────────────────────────────────────────
JNIEXPORT jintArray JNICALL Java_com_localaipainter_hunyuan_HunYuanEngine_nativeSort
  (JNIEnv* env, jobject, jlong handle, jstring jstr) {
    const char* c = env->GetStringUTFChars(jstr, nullptr);
    std::string s(c); env->ReleaseStringUTFChars(jstr, c);
    Strategy st = STRAT_ADAPTIVE;
    if(s=="max_speed")st=STRAT_MAX_SPEED;
    else if(s=="max_quality")st=STRAT_MAX_QUALITY;
    else if(s=="power_save")st=STRAT_POWER_SAVE;
    else if(s=="balanced")st=STRAT_BALANCED;
    else if(s=="custom")st=STRAT_CUSTOM;
    AlgoID ranked[ALGO_COUNT];
    // 用默认设备/快照做排序
    DeviceCap dc; dc.cpu_cores=8; dc.cpu_freq_ghz=2.0f; dc.ram_gb=8;
    dc.has_npu=false; dc.has_gpu=true; dc.has_vulkan=true; dc.soc_tier=2;
    PerfSnapshot ps; ps.cpu_usage=0.5f; ps.cpu_temp_c=50; ps.mem_used_ratio=0.5f;
    ps.battery_pct=80; ps.battery_temp=35; ps.charging=false; ps.gpu_usage=0.4f; ps.avail_mem_kb=4000000;
    sort_algos_by_strategy(ranked, st, dc, ps);
    jintArray arr = env->NewIntArray(ALGO_COUNT);
    jint buf[ALGO_COUNT];
    for(int i=0;i<ALGO_COUNT;i++)buf[i]=(jint)ranked[i];
    env->SetIntArrayRegion(arr,0,ALGO_COUNT,buf);
    return arr;
}

} // extern "C"

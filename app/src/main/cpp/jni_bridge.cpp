#include <jni.h>
#include <android/log.h>
#include <string>
#include "sd_pipeline.h"
#include "scheduler.h"
#include "quantize.h"
#include "memory_pool.h"
#include "lora_loader.h"

#define TAG "LocalAIPainter"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, TAG, __VA_ARGS__)

static SDPipeline* g_pipeline = nullptr;
static MemoryPool* g_mem_pool = nullptr;

extern "C" {

JNIEXPORT jint JNICALL JNI_OnLoad(JavaVM* vm, void* reserved) {
    LOGI("JNI_OnLoad - LocalAIPainter native library loaded");
    g_mem_pool = new MemoryPool(512); // 512MB default pool
    return JNI_VERSION_1_6;
}

JNIEXPORT void JNICALL JNI_OnUnload(JavaVM* vm, void* reserved) {
    LOGI("JNI_OnUnload");
    delete g_pipeline;
    delete g_mem_pool;
}

// ========== Pipeline ==========

JNIEXPORT void JNICALL
Java_com_localaipainter_pipeline_SDPipeline_nativeLoad(
    JNIEnv* env, jobject thiz, jstring model_path
) {
    const char* path = env->GetStringUTFChars(model_path, nullptr);
    LOGI("Loading SD pipeline from: %s", path);
    
    if (!g_pipeline) g_pipeline = new SDPipeline(g_mem_pool);
    g_pipeline->load(path);
    
    env->ReleaseStringUTFChars(model_path, path);
}

JNIEXPORT jfloatArray JNICALL
Java_com_localaipainter_pipeline_SDPipeline_nativeGenerate(
    JNIEnv* env, jobject thiz,
    jstring prompt, jstring neg_prompt,
    jint steps, jfloat cfg_scale, jlong seed,
    jint width, jint height
) {
    const char* p = env->GetStringUTFChars(prompt, nullptr);
    const char* np = env->GetStringUTFChars(neg_prompt, nullptr);
    
    auto result = g_pipeline->generate(
        std::string(p), std::string(np),
        (int)steps, (float)cfg_scale, (long)seed,
        (int)width, (int)height
    );
    
    env->ReleaseStringUTFChars(prompt, p);
    env->ReleaseStringUTFChars(neg_prompt, np);
    
    // Convert to jfloatArray
    jfloatArray arr = env->NewFloatArray(result.size());
    env->SetFloatArrayRegion(arr, 0, result.size(), result.data());
    return arr;
}

JNIEXPORT void JNICALL
Java_com_localaipainter_pipeline_SDPipeline_nativeRelease(
    JNIEnv* env, jobject thiz
) {
    if (g_pipeline) {
        g_pipeline->release();
    }
}

// ========== Text Encoder ==========

JNIEXPORT void JNICALL
Java_com_localaipainter_pipeline_TextEncoder_nativeLoad(
    JNIEnv* env, jobject thiz, jstring path
) {
    // Load CLIP text encoder model
    LOGI("Loading TextEncoder...");
}

JNIEXPORT jfloatArray JNICALL
Java_com_localaipainter_pipeline_TextEncoder_nativeEncode(
    JNIEnv* env, jobject thiz, jstring text
) {
    // Tokenize + encode text → embedding vector
    const char* t = env->GetStringUTFChars(text, nullptr);
    // Simplified: return dummy embedding
    static float dummy[768] = {0};
    env->ReleaseStringUTFChars(text, t);
    jfloatArray arr = env->NewFloatArray(768);
    env->SetFloatArrayRegion(arr, 0, 768, dummy);
    return arr;
}

// ========== Memory Info ==========

JNIEXPORT jlong JNICALL
Java_com_localaipainter_core_InferenceEngine_nativeGetMemoryUsage(
    JNIEnv* env, jobject thiz
) {
    if (g_mem_pool) {
        return (jlong)g_mem_pool->getUsedMB();
    }
    return 0;
}

} // extern "C"

#include <jni.h>
#include <android/log.h>
#include <cstring>
#include <vector>

#define TAG "TensorUtils"
#define LOGD(...) __android_log_print(ANDROID_LOG_DEBUG, TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, TAG, __VA_ARGS__)

// ---------- 张量内存池 ----------
struct TensorBlock {
    void* ptr;
    size_t size;
    bool inUse;
};

static std::vector<TensorBlock> g_pool;
static size_t g_poolUsed = 0;
static const size_t MAX_POOL = 512 * 1024 * 1024; // 512MB 张量池

extern "C" {

JNIEXPORT void JNICALL
Java_com_localaipainter_engine_TensorPool_nativeInit(JNIEnv*, jobject, jint maxMB) {
    LOGD("TensorPool init: %dMB", maxMB);
}

JNIEXPORT jlong JNICALL
Java_com_localaipainter_engine_TensorPool_nativeAlloc(JNIEnv* env, jobject, jint sizeBytes) {
    // 对齐到 64 字节
    size_t aligned = (sizeBytes + 63) & ~63;
    if (g_poolUsed + aligned > MAX_POOL) {
        LOGE("TensorPool full: used=%zu requested=%zu", g_poolUsed, aligned);
        return 0;
    }
    void* p = malloc(aligned);
    if (!p) return 0;
    g_pool.push_back({p, aligned, true});
    g_poolUsed += aligned;
    LOGD("Allocated %zu bytes, pool used=%zuMB", aligned, g_poolUsed/1024/1024);
    return (jlong)(uintptr_t)p;
}

JNIEXPORT void JNICALL
Java_com_localaipainter_engine_TensorPool_nativeFree(JNIEnv*, jobject, jlong handle) {
    void* p = (void*)(uintptr_t)handle;
    for (auto& b : g_pool) {
        if (b.ptr == p && b.inUse) {
            b.inUse = false;
            g_poolUsed -= b.size;
            LOGD("Freed %zu bytes", b.size);
            return;
        }
    }
}

JNIEXPORT void JNICALL
Java_com_localaipainter_engine_TensorPool_nativeClear(JNIEnv*, jobject) {
    for (auto& b : g_pool) {
        if (b.ptr) free(b.ptr);
    }
    g_pool.clear();
    g_poolUsed = 0;
    LOGD("TensorPool cleared");
}

// ---------- FloatArray <-> native buffer ----------
JNIEXPORT jlong JNICALL
Java_com_localaipainter_engine_TensorPool_nativeCopyFromFloatArray(JNIEnv* env, jobject, jfloatArray arr) {
    jsize len = env->GetArrayLength(arr);
    void* p = malloc(len * sizeof(jfloat));
    env->GetFloatArrayRegion(arr, 0, len, (jfloat*)p);
    g_poolUsed += len * sizeof(jfloat);
    return (jlong)(uintptr_t)p;
}

JNIEXPORT void JNICALL
Java_com_localaipainter_engine_TensorPool_nativeCopyToFloatArray(JNIEnv* env, jobject, jlong handle, jfloatArray arr) {
    jsize len = env->GetArrayLength(arr);
    void* p = (void*)(uintptr_t)handle;
    env->SetFloatArrayRegion(arr, 0, len, (const jfloat*)p);
}

}

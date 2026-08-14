#include <jni.h>
#include <android/log.h>

#define TAG "EngineFactory"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, TAG, __VA_ARGS__)

extern "C" {

JNIEXPORT jstring JNICALL
Java_com_localaipainter_core_EngineFactory_nativeGetSupportedBackends(
    JNIEnv* env,
    jclass /* clazz */
) {
    LOGI("nativeGetSupportedBackends called");
    const char* backends = "MNN,NCNN,VULKAN,CPU";
    return env->NewStringUTF(backends);
}

} // extern "C"

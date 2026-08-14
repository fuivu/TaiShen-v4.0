#include <jni.h>
#include <android/log.h>
#include "VulkanCompute.h"

#define TAG "JniVulkanBridge"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, TAG, __VA_ARGS__)

using namespace localaipainter::vulkan;

// 全局存储 VulkanComputeContext 指针
static VulkanComputeContext* g_vkContext = nullptr;

extern "C" {

// ===== 生命周期 =====

JNIEXPORT jlong JNICALL
Java_com_localaipainter_engine_VulkanEngine_nativeInit(JNIEnv* env, jobject thiz) {
    LOGI("nativeInit: creating VulkanComputeContext");
    auto* ctx = new VulkanComputeContext();
    if (!ctx->init()) {
        LOGE("Vulkan init failed!");
        delete ctx;
        return 0L;
    }
    g_vkContext = ctx;
    LOGI("nativeInit: success, handle=%p", ctx);
    return (jlong)ctx;
}

JNIEXPORT void JNICALL
Java_com_localaipainter_engine_VulkanEngine_nativeDestroy(JNIEnv* env, jobject thiz, jlong handle) {
    auto* ctx = (VulkanComputeContext*)handle;
    if (ctx) {
        ctx->destroy();
        delete ctx;
    }
    if (g_vkContext == ctx) g_vkContext = nullptr;
    LOGI("nativeDestroy: done");
}

// ===== 设备信息 =====

JNIEXPORT jstring JNICALL
Java_com_localaipainter_engine_VulkanEngine_nativeGetDeviceInfo(JNIEnv* env, jobject thiz, jlong handle) {
    auto* ctx = (VulkanComputeContext*)handle;
    if (!ctx) return env->NewStringUTF("Unknown|0|0|0|0");
    std::string info = ctx->getDeviceInfo();
    return env->NewStringUTF(info.c_str());
}

// ===== 模型加载 =====

JNIEXPORT jboolean JNICALL
Java_com_localaipainter_engine_VulkanEngine_nativeLoadModel(JNIEnv* env, jobject thiz, jlong handle, jstring modelPath) {
    auto* ctx = (VulkanComputeContext*)handle;
    if (!ctx) return JNI_FALSE;
    const char* path = env->GetStringUTFChars(modelPath, nullptr);
    bool ok = ctx->loadModel(std::string(path));
    env->ReleaseStringUTFChars(modelPath, path);
    return ok ? JNI_TRUE : JNI_FALSE;
}

JNIEXPORT void JNICALL
Java_com_localaipainter_engine_VulkanEngine_nativeUnloadModel(JNIEnv* env, jobject thiz, jlong handle) {
    auto* ctx = (VulkanComputeContext*)handle;
    if (ctx) ctx->unloadModel();
}

// ===== Dispatch =====

JNIEXPORT jboolean JNICALL
Java_com_localaipainter_engine_VulkanEngine_nativeDispatch(
    JNIEnv* env, jobject thiz, jlong handle,
    jint opType, jlong inputPtr, jlong outputPtr,
    jint param0, jint param1, jint param2,
    jfloat paramF0, jfloat paramF1
) {
    auto* ctx = (VulkanComputeContext*)handle;
    if (!ctx) return JNI_FALSE;

    const char* opNames[] = {
        "matmul", "conv2d", "gelu", "silu", "relu",
        "layernorm", "softmax", "add", "upsample", "attention", "transpose"
    };

    // 将 opType 映射到管线名
    const char* pipelineName = "conv2d"; // default
    if (opType >= 0 && opType <= 10) {
        pipelineName = opNames[opType];
    }

    // 计算工作组数量
    uint32_t gx = 1, gy = 1, gz = 1;
    switch (opType) {
        case 0: // matmul: param0=M, param1=N
            gx = (param0 + 7) / 8;
            gy = (param1 + 7) / 8;
            break;
        case 1: // conv2d: param0=width, param1=height
            gx = (param0 + 7) / 8;
            gy = (param1 + 7) / 8;
            break;
        case 2: case 3: case 4: // elementwise
            gx = (param0 * param1 + 63) / 64;
            break;
        case 5: // layernorm
            gx = param1; // per row
            break;
        case 6: // softmax
            gx = param1;
            break;
        case 7: // add
            gx = (param0 * param1 + 63) / 64;
            break;
        case 8: // upsample: param0=inW*scale, param1=inH*scale
            gx = (param0 + 7) / 8;
            gy = (param1 + 7) / 8;
            break;
        case 9: // attention: param0=heads
            gx = param0;
            break;
        case 10: // transpose
            gx = (param0 + 7) / 8;
            gy = (param1 + 7) / 8;
            break;
    }

    bool ok = ctx->dispatch(pipelineName, inputPtr, outputPtr, gx, gy, gz);
    return ok ? JNI_TRUE : JNI_FALSE;
}

// ===== 缓冲区管理 =====

JNIEXPORT jlong JNICALL
Java_com_localaipainter_engine_VulkanEngine_nativeAllocateBuffer(JNIEnv* env, jobject thiz, jlong handle, jlong sizeBytes) {
    auto* ctx = (VulkanComputeContext*)handle;
    if (!ctx) return 0L;
    uint64_t buf = ctx->allocateBuffer((VkDeviceSize)sizeBytes,
                                        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                                        VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
                                        VK_BUFFER_USAGE_TRANSFER_DST_BIT);
    return (jlong)buf;
}

JNIEXPORT void JNICALL
Java_com_localaipainter_engine_VulkanEngine_nativeFreeBuffer(JNIEnv* env, jobject thiz, jlong handle, jlong bufferPtr) {
    auto* ctx = (VulkanComputeContext*)handle;
    if (ctx) ctx->freeBuffer((uint64_t)bufferPtr);
}

// ===== 数据传输 =====

JNIEXPORT void JNICALL
Java_com_localaipainter_engine_VulkanEngine_nativeUploadData(JNIEnv* env, jobject thiz, jlong handle, jlong bufferPtr, jfloatArray data, jint offset, jint count) {
    auto* ctx = (VulkanComputeContext*)handle;
    if (!ctx) return;
    jfloat* elems = env->GetFloatArrayElements(data, nullptr);
    ctx->uploadData((uint64_t)bufferPtr, elems + offset, (size_t)offset, (size_t)count);
    env->ReleaseFloatArrayElements(data, elems, JNI_ABORT); // 不回写
}

JNIEXPORT jfloatArray JNICALL
Java_com_localaipainter_engine_VulkanEngine_nativeDownloadData(JNIEnv* env, jobject thiz, jlong handle, jlong bufferPtr, jint size) {
    auto* ctx = (VulkanComputeContext*)handle;
    if (!ctx) return env->NewFloatArray(0);

    std::vector<float> data(size);
    if (!ctx->downloadData((uint64_t)bufferPtr, data.data(), (size_t)size)) {
        return env->NewFloatArray(0);
    }

    jfloatArray result = env->NewFloatArray(size);
    env->SetFloatArrayRegion(result, 0, size, data.data());
    return result;
}

// ===== 同步 =====

JNIEXPORT jboolean JNICALL
Java_com_localaipainter_engine_VulkanEngine_nativeSubmitAndWait(JNIEnv* env, jobject thiz, jlong handle) {
    auto* ctx = (VulkanComputeContext*)handle;
    if (!ctx) return JNI_FALSE;
    return ctx->submitAndWait() ? JNI_TRUE : JNI_FALSE;
}

// ===== 内存查询 =====

JNIEXPORT jint JNICALL
Java_com_localaipainter_engine_VulkanEngine_nativeGetMemoryUsage(JNIEnv* env, jobject thiz, jlong handle) {
    auto* ctx = (VulkanComputeContext*)handle;
    if (!ctx) return 0;
    return ctx->getMemoryUsageMB();
}

} // extern "C"

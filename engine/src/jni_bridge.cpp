// JNI Bridge - 完整实现
#include <jni.h>
#include <string>
#include <android/log.h>
#include <android/asset_manager.h>
#include <android/asset_manager_jni.h>
#include <mutex>
#include <unordered_map>
#include <vector>
#include <memory>

#include "engine_factory.h"
#include "scheduler.h"
#include "tensor.h"
#include "unified_memory.h"
#include "image_io.h"

#define TAG "LocalAIPainter-JNI"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, TAG, __VA_ARGS__)
#define LOGD(...) __android_log_print(ANDROID_LOG_DEBUG, TAG, __VA_ARGS__)

namespace sd_engine {

// 全局引擎实例管理
static std::mutex g_engine_mutex;
static std::unordered_map<int, std::shared_ptr<EngineFactory>> g_engines;
static int g_next_engine_id = 1;

// 进度回调全局引用
static JavaVM* g_jvm = nullptr;
static jobject g_progress_callback = nullptr;
static jmethodID g_progress_method = nullptr;

// ============ 生命周期 ============

extern "C" JNIEXPORT jint JNICALL
Java_com_localaipainter_engine_JNIBridge_nativeInit(
    JNIEnv* env, jclass clazz, jobject context, jstring model_dir) {

    const char* dir = env->GetStringUTFChars(model_dir, nullptr);
    LOGI("nativeInit: model_dir=%s", dir);

    auto factory = std::make_shared<EngineFactory>();
    factory->set_model_dir(dir);

    std::lock_guard<std::mutex> lock(g_engine_mutex);
    int id = g_next_engine_id++;
    g_engines[id] = factory;

    env->ReleaseStringUTFChars(model_dir, dir);
    LOGI("Engine created with ID=%d", id);
    return id;
}

extern "C" JNIEXPORT void JNICALL
Java_com_localaipainter_engine_JNIBridge_nativeDestroy(
    JNIEnv* env, jclass clazz, jint engine_id) {

    std::lock_guard<std::mutex> lock(g_engine_mutex);
    auto it = g_engines.find(engine_id);
    if (it != g_engines.end()) {
        it->second->shutdown();
        g_engines.erase(it);
        LOGI("Engine %d destroyed", engine_id);
    }
}

extern "C" JNIEXPORT jboolean JNICALL
Java_com_localaipainter_engine_JNIBridge_nativeLoadModel(
    JNIEnv* env, jclass clazz, jint engine_id, jstring model_path, jint model_type) {

    const char* path = env->GetStringUTFChars(model_path, nullptr);

    std::lock_guard<std::mutex> lock(g_engine_mutex);
    auto it = g_engines.find(engine_id);
    if (it == g_engines.end()) {
        env->ReleaseStringUTFChars(model_path, path);
        LOGE("Engine %d not found", engine_id);
        return JNI_FALSE;
    }

    bool result = it->second->load_model(path, static_cast<ModelType>(model_type));
    env->ReleaseStringUTFChars(model_path, path);
    LOGI("Load model result: %s", result ? "SUCCESS" : "FAILED");
    return result ? JNI_TRUE : JNI_FALSE;
}

// ============ 配置 ============

extern "C" JNIEXPORT void JNICALL
Java_com_localaipainter_engine_JNIBridge_nativeSetScheduler(
    JNIEnv* env, jclass clazz, jint engine_id, jint scheduler_type) {

    std::lock_guard<std::mutex> lock(g_engine_mutex);
    auto it = g_engines.find(engine_id);
    if (it != g_engines.end()) {
        it->second->set_scheduler(static_cast<SchedulerType>(scheduler_type));
    }
}

extern "C" JNIEXPORT void JNICALL
Java_com_localaipainter_engine_JNIBridge_nativeSetSteps(
    JNIEnv* env, jclass clazz, jint engine_id, jint steps) {

    std::lock_guard<std::mutex> lock(g_engine_mutex);
    auto it = g_engines.find(engine_id);
    if (it != g_engines.end()) {
        it->second->set_steps(steps);
    }
}

extern "C" JNIEXPORT void JNICALL
Java_com_localaipainter_engine_JNIBridge_nativeSetCfgScale(
    JNIEnv* env, jclass clazz, jint engine_id, jfloat cfg) {

    std::lock_guard<std::mutex> lock(g_engine_mutex);
    auto it = g_engines.find(engine_id);
    if (it != g_engines.end()) {
        it->second->set_cfg_scale(cfg);
    }
}

extern "C" JNIEXPORT void JNICALL
Java_com_localaipainter_engine_JNIBridge_nativeSetSeed(
    JNIEnv* env, jclass clazz, jint engine_id, jlong seed) {

    std::lock_guard<std::mutex> lock(g_engine_mutex);
    auto it = g_engines.find(engine_id);
    if (it != g_engines.end()) {
        it->second->set_seed(static_cast<uint64_t>(seed));
    }
}

extern "C" JNIEXPORT void JNICALL
Java_com_localaipainter_engine_JNIBridge_nativeSetResolution(
    JNIEnv* env, jclass clazz, jint engine_id, jint width, jint height) {

    std::lock_guard<std::mutex> lock(g_engine_mutex);
    auto it = g_engines.find(engine_id);
    if (it != g_engines.end()) {
        it->second->set_resolution(width, height);
    }
}

extern "C" JNIEXPORT void JNICALL
Java_com_localaipainter_engine_JNIBridge_nativeSetClipSkip(
    JNIEnv* env, jclass clazz, jint engine_id, jint skip) {

    std::lock_guard<std::mutex> lock(g_engine_mutex);
    auto it = g_engines.find(engine_id);
    if (it != g_engines.end()) {
        it->second->set_clip_skip(skip);
    }
}

extern "C" JNIEXPORT void JNICALL
Java_com_localaipainter_engine_JNIBridge_nativeSetBatchSize(
    JNIEnv* env, jclass clazz, jint engine_id, jint batch) {

    std::lock_guard<std::mutex> lock(g_engine_mutex);
    auto it = g_engines.find(engine_id);
    if (it != g_engines.end()) {
        it->second->set_batch_size(batch);
    }
}

extern "C" JNIEXPORT void JNICALL
Java_com_localaipainter_engine_JNIBridge_nativeSetDenoisingStrength(
    JNIEnv* env, jclass clazz, jint engine_id, jfloat strength) {

    std::lock_guard<std::mutex> lock(g_engine_mutex);
    auto it = g_engines.find(engine_id);
    if (it != g_engines.end()) {
        it->second->set_denoising_strength(strength);
    }
}

// ============ 推理 ============

extern "C" JNIEXPORT jboolean JNICALL
Java_com_localaipainter_engine_JNIBridge_nativeGenerate(
    JNIEnv* env, jclass clazz, jint engine_id,
    jstring prompt, jstring negative_prompt, jstring output_path) {

    const char* p = env->GetStringUTFChars(prompt, nullptr);
    const char* np = env->GetStringUTFChars(negative_prompt, nullptr);
    const char* op = env->GetStringUTFChars(output_path, nullptr);

    std::lock_guard<std::mutex> lock(g_engine_mutex);
    auto it = g_engines.find(engine_id);
    if (it == g_engines.end()) {
        env->ReleaseStringUTFChars(prompt, p);
        env->ReleaseStringUTFChars(negative_prompt, np);
        env->ReleaseStringUTFChars(output_path, op);
        return JNI_FALSE;
    }

    bool result = it->second->generate(p, np, op);

    env->ReleaseStringUTFChars(prompt, p);
    env->ReleaseStringUTFChars(negative_prompt, np);
    env->ReleaseStringUTFChars(output_path, op);
    return result ? JNI_TRUE : JNI_FALSE;
}

extern "C" JNIEXPORT jboolean JNICALL
Java_com_localaipainter_engine_JNIBridge_nativeGenerateImageToImage(
    JNIEnv* env, jclass clazz, jint engine_id,
    jstring prompt, jstring negative_prompt,
    jstring input_image_path, jfloat strength, jstring output_path) {

    const char* p = env->GetStringUTFChars(prompt, nullptr);
    const char* np = env->GetStringUTFChars(negative_prompt, nullptr);
    const char* ip = env->GetStringUTFChars(input_image_path, nullptr);
    const char* op = env->GetStringUTFChars(output_path, nullptr);

    std::lock_guard<std::mutex> lock(g_engine_mutex);
    auto it = g_engines.find(engine_id);
    bool result = false;
    if (it != g_engines.end()) {
        result = it->second->generate_img2img(p, np, ip, strength, op);
    }

    env->ReleaseStringUTFChars(prompt, p);
    env->ReleaseStringUTFChars(negative_prompt, np);
    env->ReleaseStringUTFChars(input_image_path, ip);
    env->ReleaseStringUTFChars(output_path, op);
    return result ? JNI_TRUE : JNI_FALSE;
}

extern "C" JNIEXPORT void JNICALL
Java_com_localaipainter_engine_JNIBridge_nativeStop(
    JNIEnv* env, jclass clazz, jint engine_id) {

    std::lock_guard<std::mutex> lock(g_engine_mutex);
    auto it = g_engines.find(engine_id);
    if (it != g_engines.end()) {
        it->second->stop();
        LOGI("Generation stopped for engine %d", engine_id);
    }
}

// ============ 进度回调 ============

extern "C" JNIEXPORT void JNICALL
Java_com_localaipainter_engine_JNIBridge_nativeSetProgressCallback(
    JNIEnv* env, jclass clazz, jobject callback) {

    if (g_progress_callback) {
        env->DeleteGlobalRef(g_progress_callback);
    }

    if (callback) {
        g_progress_callback = env->NewGlobalRef(callback);

        jclass cb_class = env->GetObjectClass(callback);
        g_progress_method = env->GetMethodID(cb_class, "onProgress",
            "(IIDD)V");  // step, totalSteps, progress, etaSeconds

        env->DeleteLocalRef(cb_class);
    } else {
        g_progress_callback = nullptr;
        g_progress_method = nullptr;
    }

    // 保存 JVM 引用
    env->GetJavaVM(&g_jvm);
}

// C++ 端调用此函数通知进度
void report_progress(int step, int total_steps, double progress, double eta) {
    if (!g_progress_callback || !g_progress_method || !g_jvm) return;

    JNIEnv* env = nullptr;
    bool attached = false;
    if (g_jvm->GetEnv(reinterpret_cast<void**>(&env), JNI_VERSION_1_6) != JNI_OK) {
        if (g_jvm->AttachCurrentThread(&env, nullptr) == JNI_OK) {
            attached = true;
        }
    }

    if (env) {
        env->CallVoidMethod(g_progress_callback, g_progress_method,
            step, total_steps, progress, eta);
    }

    if (attached) {
        g_jvm->DetachCurrentThread();
    }
}

// ============ LoRA ============

extern "C" JNIEXPORT jboolean JNICALL
Java_com_localaipainter_engine_JNIBridge_nativeLoadLora(
    JNIEnv* env, jclass clazz, jint engine_id,
    jstring lora_path, jfloat weight) {

    const char* path = env->GetStringUTFChars(lora_path, nullptr);

    std::lock_guard<std::mutex> lock(g_engine_mutex);
    auto it = g_engines.find(engine_id);
    bool result = false;
    if (it != g_engines.end()) {
        result = it->second->load_lora(path, weight);
    }

    env->ReleaseStringUTFChars(lora_path, path);
    return result ? JNI_TRUE : JNI_FALSE;
}

extern "C" JNIEXPORT void JNICALL
Java_com_localaipainter_engine_JNIBridge_nativeUnloadLora(
    JNIEnv* env, jclass clazz, jint engine_id, jstring lora_name) {

    const char* name = env->GetStringUTFChars(lora_name, nullptr);

    std::lock_guard<std::mutex> lock(g_engine_mutex);
    auto it = g_engines.find(engine_id);
    if (it != g_engines.end()) {
        it->second->unload_lora(name);
    }

    env->ReleaseStringUTFChars(lora_name, name);
}

extern "C" JNIEXPORT void JNICALL
Java_com_localaipainter_engine_JNIBridge_nativeSetLoraWeight(
    JNIEnv* env, jclass clazz, jint engine_id, jstring lora_name, jfloat weight) {

    const char* name = env->GetStringUTFChars(lora_name, nullptr);

    std::lock_guard<std::mutex> lock(g_engine_mutex);
    auto it = g_engines.find(engine_id);
    if (it != g_engines.end()) {
        it->second->set_lora_weight(name, weight);
    }

    env->ReleaseStringUTFChars(lora_name, name);
}

// ============ ControlNet ============

extern "C" JNIEXPORT jboolean JNICALL
Java_com_localaipainter_engine_JNIBridge_nativeLoadControlNet(
    JNIEnv* env, jclass clazz, jint engine_id,
    jstring controlnet_path, jint control_type) {

    const char* path = env->GetStringUTFChars(controlnet_path, nullptr);

    std::lock_guard<std::mutex> lock(g_engine_mutex);
    auto it = g_engines.find(engine_id);
    bool result = false;
    if (it != g_engines.end()) {
        result = it->second->load_controlnet(path,
            static_cast<ControlNetType>(control_type));
    }

    env->ReleaseStringUTFChars(controlnet_path, path);
    return result ? JNI_TRUE : JNI_FALSE;
}

extern "C" JNIEXPORT void JNICALL
Java_com_localaipainter_engine_JNIBridge_nativeSetControlImage(
    JNIEnv* env, jclass clazz, jint engine_id,
    jstring image_path, jfloat strength) {

    const char* path = env->GetStringUTFChars(image_path, nullptr);

    std::lock_guard<std::mutex> lock(g_engine_mutex);
    auto it = g_engines.find(engine_id);
    if (it != g_engines.end()) {
        it->second->set_control_image(path, strength);
    }

    env->ReleaseStringUTFChars(image_path, path);
}

// ============ 超分 ============

extern "C" JNIEXPORT jboolean JNICALL
Java_com_localaipainter_engine_JNIBridge_nativeUpscale(
    JNIEnv* env, jclass clazz, jint engine_id,
    jstring input_path, jstring output_path, jint scale) {

    const char* ip = env->GetStringUTFChars(input_path, nullptr);
    const char* op = env->GetStringUTFChars(output_path, nullptr);

    std::lock_guard<std::mutex> lock(g_engine_mutex);
    auto it = g_engines.find(engine_id);
    bool result = false;
    if (it != g_engines.end()) {
        result = it->second->upscale(ip, op, scale);
    }

    env->ReleaseStringUTFChars(input_path, ip);
    env->ReleaseStringUTFChars(output_path, op);
    return result ? JNI_TRUE : JNI_FALSE;
}

// ============ 人脸修复 ============

extern "C" JNIEXPORT jboolean JNICALL
Java_com_localaipainter_engine_JNIBridge_nativeRestoreFace(
    JNIEnv* env, jclass clazz, jint engine_id,
    jstring input_path, jstring output_path, jint model_type) {

    const char* ip = env->GetStringUTFChars(input_path, nullptr);
    const char* op = env->GetStringUTFChars(output_path, nullptr);

    std::lock_guard<std::mutex> lock(g_engine_mutex);
    auto it = g_engines.find(engine_id);
    bool result = false;
    if (it != g_engines.end()) {
        result = it->second->restore_face(ip, op, model_type);
    }

    env->ReleaseStringUTFChars(input_path, ip);
    env->ReleaseStringUTFChars(output_path, op);
    return result ? JNI_TRUE : JNI_FALSE;
}

// ============ 设备检测 ============

extern "C" JNIEXPORT jstring JNICALL
Java_com_localaipainter_engine_JNIBridge_nativeGetBestBackend(
    JNIEnv* env, jclass clazz, jint engine_id) {

    std::lock_guard<std::mutex> lock(g_engine_mutex);
    auto it = g_engines.find(engine_id);
    if (it == g_engines.end()) {
        return env->NewStringUTF("cpu");
    }

    std::string backend = it->second->get_best_backend();
    return env->NewStringUTF(backend.c_str());
}

extern "C" JNIEXPORT jboolean JNICALL
Java_com_localaipainter_engine_JNIBridge_nativeHasGpu(
    JNIEnv* env, jclass clazz) {
    // 检测 GPU 可用性
    // 实际应通过 NDK 检测 OpenGL ES / Vulkan 支持
    return JNI_TRUE;  // placeholder
}

extern "C" JNIEXPORT jint JNICALL
Java_com_localaipainter_engine_JNIBridge_nativeGetMemoryInfo(
    JNIEnv* env, jclass clazz, jintArray out_info) {

    jint* info = env->GetIntArrayElements(out_info, nullptr);
    if (info) {
        // info[0] = total RAM (MB)
        // info[1] = available RAM (MB)
        // info[2] = GPU memory (MB, 0 if unknown)
        info[0] = 8192;   // placeholder: 8GB
        info[1] = 4096;   // placeholder: 4GB available
        info[2] = 2048;   // placeholder: 2GB GPU
        env->ReleaseIntArrayElements(out_info, info, 0);
    }
    return 0;
}

// ============ 模型管理 ============

extern "C" JNIEXPORT jboolean JNICALL
Java_com_localaipainter_engine_JNIBridge_nativeValidateModel(
    JNIEnv* env, jclass clazz, jstring model_path) {

    const char* path = env->GetStringUTFChars(model_path, nullptr);

    // 检查文件存在性和基本格式
    FILE* f = fopen(path, "rb");
    bool valid = false;
    if (f) {
        // 检查 magic bytes
        char header[8] = {0};
        fread(header, 1, 8, f);

        // safetensors: "safetensor"
        // gguf: "GGUF"
        // onnx: "ONNX"
        if (memcmp(header, "safetensor", 8) == 0 ||
            memcmp(header, "GGUF", 4) == 0 ||
            memcmp(header, "ONNX", 4) == 0) {
            valid = true;
        }
        fclose(f);
    }

    env->ReleaseStringUTFChars(model_path, path);
    return valid ? JNI_TRUE : JNI_FALSE;
}

extern "C" JNIEXPORT jstring JNICALL
Java_com_localaipainter_engine_JNIBridge_nativeGetModelInfo(
    JNIEnv* env, jclass clazz, jstring model_path) {

    const char* path = env->GetStringUTFChars(model_path, nullptr);

    // 返回 JSON 格式的模型信息
    // {"type": "safetensors", "size_mb": 2048, "sd_version": "SD1.5"}
    std::string info = "{\"type\":\"unknown\",\"size_mb\":0}";

    FILE* f = fopen(path, "rb");
    if (f) {
        char header[8] = {0};
        fread(header, 1, 8, f);
        fclose(f);

        // 获取文件大小
        long size = 0;
        FILE* fs = fopen(path, "rb");
        if (fs) {
            fseek(fs, 0, SEEK_END);
            size = ftell(fs);
            fclose(fs);
        }
        long size_mb = size / (1024 * 1024);

        if (memcmp(header, "safetensor", 8) == 0) {
            info = "{\"type\":\"safetensors\",\"size_mb\":" + std::to_string(size_mb) + "}";
        } else if (memcmp(header, "GGUF", 4) == 0) {
            info = "{\"type\":\"gguf\",\"size_mb\":" + std::to_string(size_mb) + "}";
        } else if (memcmp(header, "ONNX", 4) == 0) {
            info = "{\"type\":\"onnx\",\"size_mb\":" + std::to_string(size_mb) + "}";
        }
    }

    env->ReleaseStringUTFChars(model_path, path);
    return env->NewStringUTF(info.c_str());
}

// ============ 性能统计 ============

extern "C" JNIEXPORT jstring JNICALL
Java_com_localaipainter_engine_JNIBridge_nativeGetPerfStats(
    JNIEnv* env, jclass clazz, jint engine_id) {

    std::lock_guard<std::mutex> lock(g_engine_mutex);
    auto it = g_engines.find(engine_id);
    if (it == g_engines.end()) {
        return env->NewStringUTF("{}");
    }

    // 返回 JSON 格式的性能统计
    std::string stats = "{\"avg_step_ms\":0,\"total_time_s\":0,\"memory_mb\":0}";
    // 实际应从 EngineFactory 获取
    return env->NewStringUTF(stats.c_str());
}

// ============ JNI 初始化 ============

JNIEXPORT jint JNICALL JNI_OnLoad(JavaVM* vm, void* reserved) {
    g_jvm = vm;
    JNIEnv* env = nullptr;
    if (vm->GetEnv(reinterpret_cast<void**>(&env), JNI_VERSION_1_6) != JNI_OK) {
        return JNI_ERR;
    }
    LOGI("Local AI Painter JNI loaded (v3.0)");
    return JNI_VERSION_1_6;
}

JNIEXPORT void JNICALL JNI_OnUnload(JavaVM* vm, void* reserved) {
    LOGI("Local AI Painter JNI unloaded");

    // 清理所有引擎
    std::lock_guard<std::mutex> lock(g_engine_mutex);
    for (auto& pair : g_engines) {
        pair.second->shutdown();
    }
    g_engines.clear();

    if (g_progress_callback) {
        JNIEnv* env = nullptr;
        if (vm->GetEnv(reinterpret_cast<void**>(&env), JNI_VERSION_1_6) == JNI_OK) {
            env->DeleteGlobalRef(g_progress_callback);
        }
        g_progress_callback = nullptr;
    }
    g_progress_method = nullptr;
    g_jvm = nullptr;
}

}  // namespace sd_engine

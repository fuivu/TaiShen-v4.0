package com.localaipainter.core

import android.content.Context
import android.content.SharedPreferences
import com.localaipainter.util.Logger

/**
 * 🎚️ 功能开关系统 —— 远程配置 + 本地覆盖 + A/B 实验
 *
 * 设计目标：
 *   - 每个功能有唯一 key，可远程开关（未来接 Firebase Remote Config）
 *   - 本地可强制覆盖（开发者选项）
 *   - 默认按设备能力自动决定（天玑8400 自动开 GPU 缓存等）
 *   - 新增功能只需声明一个 FeatureFlag，零侵入
 *
 * 用法：
 *   if (FeatureToggle.isEnabled(FeatureFlag.GPU_CACHE)) { ... }
 */
object FeatureToggle {

    enum class FeatureFlag(
        val key: String,
        val defaultValue: Boolean,
        val minSdk: Int = 26,
        val needsNpu: Boolean = false,
        val needsVulkan: Boolean = false,
        val description: String = ""
    ) {
        // ---- 核心功能 ----
        HETEROGENEOUS_PIPELINE("hetero_pipeline", true, description = "异构计算管线"),
        GPU_CACHE("gpu_cache", true, needsVulkan = true, description = "Vulkan 显存缓存"),
        LRU_TENSOR_CACHE("lru_tensor_cache", true, description = "张量 LRU 缓存"),
        BATCH_GENERATION("batch_generation", true, description = "批量生成"),
        LORA_LOADING("lora_loading", true, description = "LoRA 模型加载"),
        IMG2IMG("img2img", true, description = "图生图"),
        INPAINTING("inpaint", true, description = "局部重绘"),
        CLIP_SKIP("clip_skip", true, description = "CLIP 跳跃数"),

        // ---- 调度器组 ----
        SCHEDULER_LCM("sched_lcm", true, description = "LCM 极速"),
        SCHEDULER_TURBO("sched_turbo", true, description = "SD-Turbo"),
        SCHEDULER_DPM_SOLVER("sched_dpm_solver", true, description = "DPM-Solver++"),
        SCHEDULER_RESTART("sched_restart", false, description = "Restart 调度器（实验）"),

        // ---- 量化 ----
        QUANT_INT8("quant_int8", true, description = "INT8 量化"),
        QUANT_INT4("quant_int4", false, description = "INT4 量化（实验）"),
        QUANT_KL("quant_kl", true, description = "KL 散度量化"),

        // ---- UI ----
        DYNAMIC_THEME("dynamic_theme", true, minSdk = 31, description = "动态配色（Android 12+）"),
        BG_IMAGE_THEME("bg_image_theme", true, description = "背景图取色主题"),
        SMOOTH_TRANSITION("smooth_transition", true, minSdk = 31, description = "共享元素过渡"),
        ADAPTIVE_ICON("adaptive_icon", true, minSdk = 26, description = "自适应图标"),

        // ---- 性能 ----
        FOREGROUND_SERVICE("foreground_service", true, description = "前台保活服务"),
        AUTO_START("auto_start", true, minSdk = 26, description = "开机自启"),
        MEMORY_MONITOR("memory_monitor", true, description = "内存监控"),
        CRASH_REPORT("crash_report", true, description = "崩溃报告"),

        // ---- 导出 ----
        EXPORT_PNG("export_png", true, description = "PNG 导出"),
        EXPORT_WEBP("export_webp", true, description = "WebP 导出"),
        EXPORT_JPEG("export_jpeg", true, description = "JPEG 导出"),
        EXPORT_AVIF("export_avif", false, minSdk = 30, description = "AVIF 导出（实验）"),

        // ---- 后处理 ----
        POST_GAUSSIAN("post_gaussian", true, description = "高斯模糊"),
        POST_UPSCALE("post_upscale", false, description = "AI 超分（实验）"),
        POST_FACE_FIX("post_face_fix", false, description = "人脸修复（实验）"),
        POST_COLOR_CORRECT("post_color_correct", true, description = "色彩校正"),

        // ---- 调试 ----
        DEBUG_OVERLAY("debug_overlay", false, description = "调试覆盖层"),
        PERF_METRICS("perf_metrics", true, description = "性能指标采集"),
        VERBOSE_LOG("verbose_log", false, description = "详细日志"),
    }

    private lateinit var prefs: SharedPreferences
    private val overrides = mutableMapOf<String, Boolean>()
    private val remoteValues = mutableMapOf<String, Boolean>()

    fun init(context: Context) {
        prefs = context.getSharedPreferences("feature_toggles", Context.MODE_PRIVATE)
        // 写入默认值
        FeatureFlag.values().forEach { flag ->
            if (!prefs.contains(flag.key)) {
                prefs.edit().putBoolean(flag.key, flag.defaultValue).apply()
            }
        }
        Logger.i("FeatureToggle", "${FeatureFlag.values().size} 个功能开关已初始化")
    }

    fun isEnabled(flag: FeatureFlag, context: Context? = null): Boolean {
        // 1. 本地强制覆盖（最高优先级）
        overrides[flag.key]?.let { return it }

        // 2. 远程配置（未来接 Firebase）
        remoteValues[flag.key]?.let { return it }

        // 3. 本地持久化值
        val stored = prefs.getBoolean(flag.key, flag.defaultValue)

        // 4. 能力检查（不满足自动关）
        if (context != null) {
            if (flag.needsNpu) {
                val chip = com.localaipainter.util.DeviceDetector(context).detectChip()
                val hasNpu = chip.npuName.isNotBlank() && chip.npuName != "None"
                if (!hasNpu) return false
            }
            if (flag.needsVulkan) {
                val hasVk = com.localaipainter.util.DeviceDetector(context).detectChip().hasVulkan
                if (!hasVk) return false
            }
            if (android.os.Build.VERSION.SDK_INT < flag.minSdk) return false
        }

        return stored
    }

    fun setOverride(flag: FeatureFlag, enabled: Boolean) {
        overrides[flag.key] = enabled
        Logger.i("FeatureToggle", "本地覆盖: ${flag.key} = $enabled")
    }

    fun setRemoteValue(flag: FeatureFlag, enabled: Boolean) {
        remoteValues[flag.key] = enabled
        Logger.i("FeatureToggle", "远程配置: ${flag.key} = $enabled")
    }

    fun resetOverride(flag: FeatureFlag) {
        overrides.remove(flag.key)
    }

    fun resetAll() {
        overrides.clear()
        remoteValues.clear()
        prefs.edit().clear().apply()
        Logger.i("FeatureToggle", "所有开关已重置为默认")
    }

    fun getAllStates(context: Context): Map<String, Boolean> =
        FeatureFlag.values().associate { it.key to isEnabled(it, context) }

    fun getEnabledList(context: Context): List<String> =
        FeatureFlag.values().filter { isEnabled(it, context) }.map { it.key }
}

package com.localaipainter.util

import android.content.Context
import com.localaipainter.engine.DeviceDetector as EngineDetector
import com.localaipainter.core.DevicePowerScore

/**
 * DeviceProfile —— util 层轻量封装
 *
 * 与 engine.DeviceDetector 的关系：
 *   - engine.DeviceDetector 是完整版（含 GPU/OpenGL/Vulkan 深度检测 + NPU 识别）
 *   - 本类是轻量封装，供 UI/工具代码使用，避免直接依赖 engine 包
 *
 * 太神架构 v4.0：
 *   - 集成 DevicePowerScore
 *   - 统一返回"太神 AI 引擎 4.0"标识
 *   - 委托给 engine.DeviceDetector 获取完整信息
 */
class DeviceProfile(private val context: Context) {

    companion object {
        private const val TAG = "DeviceProfile"
        const val ENGINE_IDENTITY = "太神 AI 引擎 4.0"
    }

    data class ChipInfo(
        val vendor: String,
        val model: String,
        val socName: String,
        val totalRamMB: Long,
        val supportedBackends: List<String>,
        val preferredBackend: String,
        val npuName: String = "",
        val hasVulkan: Boolean = false,
        val hasOpenGLES: Boolean = false,
        val supportsHybrid: Boolean = false,
        val engineIdentity: String = ENGINE_IDENTITY,
        val powerScore: Float = 0f,
        val recommendedPath: String = "LOCAL_INT4_VULKAN",
    )

    fun detectChip(): ChipInfo {
        val fullInfo = EngineDetector(context).detect()
        val score = DevicePowerScore(context).compute()
        val path = DevicePowerScore(context).decidePath(score)

        return ChipInfo(
            vendor = fullInfo.gpuVendor.ifBlank { "Unknown" },
            model = fullInfo.model,
            socName = fullInfo.chipset,
            totalRamMB = (fullInfo.totalRAMGB * 1024L),
            supportedBackends = fullInfo.supportedBackends,
            preferredBackend = fullInfo.preferredBackend,
            npuName = fullInfo.npuType,
            hasVulkan = fullInfo.hasVulkan,
            hasOpenGLES = fullInfo.hasOpenGLES,
            supportsHybrid = fullInfo.supportsHybrid,
            engineIdentity = ENGINE_IDENTITY,
            powerScore = score,
            recommendedPath = path,
        )
    }

    /**
     * 获取统一引擎标识（红线2：所有设备显示同一身份）
     */
    fun getEngineIdentity(): String = ENGINE_IDENTITY

    /**
     * 获取路径描述（红线1：不暴露性能差异）
     */
    fun getPathLabel(): String = "⚡ 太神引擎极速响应中"

    // ─── 旧 API 兼容 ────────────────────────

    fun isVulkanAvailable(): Boolean =
        EngineDetector(context).detect().hasVulkan

    fun isOpenGLAvailable(): Boolean =
        EngineDetector(context).detect().hasOpenGLES

    fun supportsHybridMode(): Boolean =
        EngineDetector(context).detect().supportsHybrid
}

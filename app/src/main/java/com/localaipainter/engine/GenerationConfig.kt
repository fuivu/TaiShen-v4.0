package com.localaipainter.engine

import com.localaipainter.data.entity.LoRAInfo

/**
 * 生成配置 — 对应 C++ 端 sd_pipeline 的全部参数
 */
data class GenerationConfig(
    // === 基础 ===
    val prompt: String = "",
    val negativePrompt: String = "",

    // === 采样 ===
    val scheduler: SchedulerType = SchedulerType.EULER_A,
    val steps: Int = 30,
    val cfgScale: Float = 7.5f,
    val seed: Long = -1, // -1 = 随机

    // === 分辨率 ===
    val width: Int = 512,
    val height: Int = 512,

    // === 高级 ===
    val batchSize: Int = 1,
    val clipSkip: Int = 1,
    val denoisingStrength: Float = 1.0f,

    // === LoRA ===
    val loras: List<LoRAInfo> = emptyList(),

    // === ControlNet ===
    val controlNetType: ControlNetType? = null,
    val controlNetImagePath: String? = null,
    val controlNetStrength: Float = 1.0f,

    // === 模型 ===
    val modelPath: String = "",
    val vaePath: String = "",

    // === 后处理 ===
    val upscaleFactor: Int = 1,    // 1=不超分, 2=2x, 4=4x
    val faceRestore: FaceRestoreType = FaceRestoreType.NONE,

    // === 性能 ===
    val powerMode: PowerMode = PowerMode.BALANCED,
    val useGpu: Boolean = true,
    val threads: Int = 4,
) {
    fun effectiveSeed(): Long = if (seed < 0) System.currentTimeMillis() else seed

    fun totalImages(): Int = batchSize

    companion object {
        val DEFAULT = GenerationConfig()
    }
}

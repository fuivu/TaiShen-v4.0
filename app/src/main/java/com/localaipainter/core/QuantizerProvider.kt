package com.localaipainter.core

/**
 * 量化方案提供者 —— 插件式注册
 *
 * 新增量化方案的步骤：
 *   1. 实现量化逻辑（继承 BaseQuantizer 或独立函数）
 *   2. 在 PluginRegistry.registerBuiltinQuantizers() 中调用 registerQuantizer()
 *
 * 支持的目标：INT8 / INT4 / FP16 / 混合精度 / KL 散度最优
 */
data class QuantizerProvider(
    val name: String,
    val description: String,
    val targetPrecision: String = "INT8",   // INT8 / INT4 / FP16 / MIXED
    val compressionRatio: Float = 4f,       // 相对 FP32 的压缩比
    val qualityLoss: Float = 0.02f,        // 预估质量损失 (0-1)
    val supportsGpu: Boolean = true,
    val supportsNpu: Boolean = false,
    val quantize: (modelPath: String, outputDir: String) -> String // 返回输出路径
) {
    fun run(modelPath: String, outputDir: String): String {
        Logger.i("Quantizer", "量化 [$name] $modelPath → $outputDir")
        return quantize(modelPath, outputDir)
    }
}

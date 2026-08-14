package com.localaipainter.core

import android.content.Context
import android.net.Uri

/**
 * 模型格式提供者 —— 插件式注册
 *
 * 新增模型格式的步骤：
 *   1. 实现 ModelFormatProvider（解析、转换、验证）
 *   2. 在 PluginRegistry.registerBuiltinModelFormats() 中调用 registerModelFormat()
 *
 * 支持：ONNX / SafeTensors / PyTorch ckpt / MNN / NCNN / DLA / GGML
 */
data class ModelFormatProvider(
    val extension: String,           // "onnx" / "safetensors" / "mnn"
    val displayName: String,         // 显示名
    val canLoadDirectly: Boolean,    // 是否可直接加载，还是需要转换
    val needsConversion: Boolean = !canLoadDirectly,
    val targetFormat: String? = if (needsConversion) "ONNX" else null,
    val minFileSizeMB: Long = 0,
    val maxFileSizeMB: Long = 4096,
    val supportedOps: Set<String> = emptySet(),   // 该格式支持的算子标签
    val validate: (filePath: String) -> Boolean = { true },
    val convertToOnnx: ((filePath: String, outputDir: String) -> String)? = null,
    val extractMetadata: (filePath: String) -> ModelMetadata = { ModelMetadata() }
) {
    fun isValid(path: String): Boolean {
        if (!validate(path)) return false
        val sizeMB = java.io.File(path).length() / (1024 * 1024)
        return sizeMB in minFileSizeMB..maxFileSizeMB
    }

    fun getMetadata(path: String): ModelMetadata = extractMetadata(path)
}

/**
 * 模型元信息（从模型文件头解析）
 */
data class ModelMetadata(
    val name: String = "Unknown",
    val version: String = "1.0",
    val modelType: String = "SD1.5",   // SD1.5 / SDXL / SD2.0 / LCM
    val architecture: String = "UNet",
    val paramCount: Long = 0,
    val recommendedPrecision: String = "FP16",
    val triggerWords: List<String> = emptyList(),
    val license: String = "Unknown",
    val source: String = "Unknown"
)

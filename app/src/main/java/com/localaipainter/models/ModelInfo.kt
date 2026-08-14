package com.localaipainter.models

/**
 * 模型信息 —— UI 层使用的轻量模型描述
 */
data class ModelInfo(
    val name: String,
    val path: String,
    val type: String,           // "SD1.5" / "SD2.1" / "SDXL" / "LCM" / "PixArt" / "Kolors"
    val precision: String = "FP16",  // "FP32" / "FP16" / "INT8" / "INT4"
    val backend: String = "CPU",    // "CPU" / "VULKAN" / "OPENGL" / "NPU" / "MNN" / "NCNN"
    val fileSizeMB: Long = 0L,
    val isFavorite: Boolean = false,
    val lastUsedAt: Long = 0L,
    val description: String = "",
)

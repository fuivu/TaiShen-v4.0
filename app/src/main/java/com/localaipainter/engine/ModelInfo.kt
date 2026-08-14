package com.localaipainter.engine

/**
 * 模型文件元信息
 */
data class ModelInfo(
    val name: String,
    val format: String,       // "SAFETENSORS" / "CKPT" / "ONNX" / "MNN"
    val fileSizeMB: Long,
    val isValid: Boolean
)

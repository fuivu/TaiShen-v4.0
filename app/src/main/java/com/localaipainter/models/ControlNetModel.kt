package com.localaipainter.models

/**
 * ControlNet 模型描述
 */
data class ControlNetModel(
    val name: String,
    val path: String,
    val type: String,           // "CANNY" / "OPENPOSE" / "DEPTH" / "SCRIBBLE" / "MLSD" / "SEG"
    val enabled: Boolean = true,
    val strength: Float = 1.0f,
    val preprocessor: String = "auto",  // 预处理方式
    val fileSizeMB: Long = 0L,
)

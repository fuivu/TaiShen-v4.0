package com.localaipainter.models

/**
 * LoRA 模型描述
 */
data class LoRAModel(
    val name: String,
    val path: String,
    val rank: Int = 16,
    val triggerWords: String = "",
    val weight: Float = 1.0f,
    val enabled: Boolean = true,
    val fileSizeMB: Long = 0L,
)

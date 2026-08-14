package com.localaipainter.data.entity

/**
 * LoRA 信息 — 用于 GenerationConfig 中的 loras 列表
 */
data class LoRAInfo(
    val name: String,
    val path: String,
    val weight: Float = 1.0f,
)

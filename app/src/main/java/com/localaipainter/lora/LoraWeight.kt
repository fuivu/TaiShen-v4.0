package com.localaipainter.lora

/**
 * LoRA 权重数据类
 */
data class LoraWeight(
    val name: String,          // LoRA 名称
    val path: String,          // 文件路径
    val scale: Float = 1.0f,   // 权重缩放
    val rank: Int = 4,         // LoRA rank
    val alpha: Float = 4f,     // LoRA alpha
    val targetModules: List<String> = listOf("to_q", "to_v", "to_k", "to_out")
)

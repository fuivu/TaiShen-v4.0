package com.localaipainter.data.dao

import androidx.room.Entity
import androidx.room.PrimaryKey

/**
 * 生成历史记录 — Room 实体（太神架构 v4.0）
 */
@Entity(tableName = "generation_history")
data class GenerationHistory(
    @PrimaryKey(autoGenerate = true)
    val id: Long = 0,

    // 时间
    val timestamp: Long = System.currentTimeMillis(),

    // 输入
    val prompt: String = "",
    val negativePrompt: String = "",
    val modelPath: String = "",
    val vaePath: String = "",

    // 参数
    val scheduler: String = "euler_a",
    val steps: Int = 30,
    val cfgScale: Float = 7.5f,
    val seed: Long = -1,
    val width: Int = 512,
    val height: Int = 512,
    val batchSize: Int = 1,
    val clipSkip: Int = 1,
    val denoisingStrength: Float = 1.0f,

    // LoRA
    val loraInfo: String = "",

    // ControlNet
    val controlNetType: String = "",
    val controlNetStrength: Float = 0f,

    // 后处理
    val upscaleFactor: Int = 1,
    val faceRestore: String = "",

    // 输出
    val outputPath: String = "",
    val outputCount: Int = 1,
    val generationTimeMs: Long = 0,

    // 性能
    val backend: String = "CPU",
    val threads: Int = 4,
    val powerMode: String = "BALANCED",

    // 用户标记
    val isFavorite: Boolean = false,
    val userRating: Int = 0,
    val userNote: String = "",
)

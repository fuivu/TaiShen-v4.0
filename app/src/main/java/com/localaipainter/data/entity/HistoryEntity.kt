package com.localaipainter.data.entity

import androidx.room.Entity
import androidx.room.PrimaryKey

@Entity(tableName = "history")
data class HistoryEntity(
    @PrimaryKey(autoGenerate = true) val id: Long = 0,
    val prompt: String,
    val negativePrompt: String = "",
    val modelId: Long,
    val steps: Int,
    val cfgScale: Float,
    val width: Int,
    val height: Int,
    val seed: Long,
    val scheduler: String,
    val elapsedMs: Long,
    val outputPath: String,
    val createdAt: Long = System.currentTimeMillis()
)

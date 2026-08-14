package com.localaipainter.data.entity

import androidx.room.Entity
import androidx.room.PrimaryKey

@Entity(tableName = "tasks")
data class TaskEntity(
    @PrimaryKey(autoGenerate = true) val id: Long = 0,
    val prompt: String,
    val negativePrompt: String = "",
    val modelId: Long,
    val steps: Int = 20,
    val cfgScale: Float = 7.5f,
    val width: Int = 512,
    val height: Int = 512,
    val seed: Long = -1,
    val scheduler: String = "EulerA",
    var status: String = "PENDING",
    var error: String? = null,
    var outputPath: String? = null,
    val createdAt: Long = System.currentTimeMillis()
)

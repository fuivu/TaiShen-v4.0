package com.localaipainter.data.entity

import androidx.room.Entity
import androidx.room.PrimaryKey

@Entity(tableName = "loras")
data class LoraEntity(
    @PrimaryKey(autoGenerate = true) val id: Long = 0,
    val name: String,
    val filePath: String,
    val triggerWords: String = "",
    val fileSize: Long,
    val verified: Boolean = false,
    val importedAt: Long = System.currentTimeMillis()
)

package com.localaipainter.data.entity

import androidx.room.Entity
import androidx.room.PrimaryKey

/**
 * 主模型（Checkpoint）数据库实体
 *
 * 太神架构 v4.0 字段说明：
 *   - isFavorite: 用户收藏标记，收藏的模型置顶展示
 *   - lastUsedAt: 最后使用时间，用于"最近使用"排序
 *   - description: 用户自定义备注
 *   - thumbnailPath: 模型封面缩略图路径
 */
@Entity(tableName = "models")
data class ModelEntity(
    @PrimaryKey(autoGenerate = true) val id: Long = 0,
    val name: String,
    val filePath: String,
    val type: String,                    // "SD1.5" / "SD2.1" / "SDXL" / "LCM" / "PixArt" / "Kolors"
    val backend: String,                 // "MNN" / "NCNN" / "QNN" / "VULKAN" / "ONNX" / "CPU"
    val precision: String,              // "FP32" / "FP16" / "INT8" / "INT4"
    val fileSize: Long,                 // 字节
    val verified: Boolean = false,      // 是否通过完整性校验
    val isFavorite: Boolean = false,    // ★ 收藏标记
    val lastUsedAt: Long = 0L,         // 最后使用时间戳（毫秒）
    val description: String = "",       // 用户备注
    val thumbnailPath: String = "",    // 封面缩略图路径
    val importedAt: Long = System.currentTimeMillis()
)

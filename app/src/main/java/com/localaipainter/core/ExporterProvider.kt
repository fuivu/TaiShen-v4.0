package com.localaipainter.core

import android.content.Context
import android.graphics.Bitmap
import android.net.Uri

/**
 * 导出器提供者 —— 插件式注册
 *
 * 新增导出格式的步骤：
 *   1. 实现导出逻辑
 *   2. 在 PluginRegistry.registerBuiltinExporters() 中调用 registerExporter()
 *
 * 支持：PNG / JPEG / WebP / AVIF / GIF 动画 / 视频导出
 */
data class ExporterProvider(
    val format: String,           // "PNG" / "JPEG" / "WEBP" / "GIF" / "MP4"
    val description: String,
    val mimeType: String,         // "image/png"
    val extension: String = format.lowercase(),
    val supportsTransparency: Boolean = false,
    val supportsAnimation: Boolean = false,
    val lossless: Boolean = true,
    val qualityRange: ClosedFloatingPointRange<Float> = 0.5f..1.0f,
    val defaultQuality: Float = 0.95f,
    val export: (context: Context, bitmap: Bitmap, quality: Float) -> Uri
) {
    fun runExport(context: Context, bitmap: Bitmap, quality: Float = defaultQuality): Uri {
        Logger.i("Exporter", "导出 [$format] quality=$quality")
        return export(context, bitmap, quality.coerceIn(qualityRange))
    }
}

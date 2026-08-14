package com.localaipainter.core

import android.content.Context
import android.graphics.Bitmap

/**
 * 图像后处理器提供者 —— 插件式注册
 *
 * 新增后处理步骤：
 *   1. 实现处理逻辑
 *   2. 在 PluginRegistry.registerBuiltinPostProcessors() 中调用 registerPostProcessor()
 *
 * 支持：高斯模糊 / AI 超分 / 人脸修复 / 色彩校正 / 锐化 / 风格迁移
 */
data class PostProcessorProvider(
    val name: String,
    val description: String,
    val category: PostProcessCategory = PostProcessCategory.ENHANCE,
    val supportsGpu: Boolean = true,
    val supportsCpu: Boolean = true,
    val qualityRange: ClosedFloatingPointRange<Float> = 0f..1f,
    val defaultQuality: Float = 0.5f,
    val process: (context: Context, bitmap: Bitmap, quality: Float) -> Bitmap
) {
    fun run(context: Context, bitmap: Bitmap, quality: Float = defaultQuality): Bitmap {
        Logger.i("PostProcess", "处理 [$name] quality=$quality")
        return process(context, bitmap, quality.coerceIn(qualityRange))
    }
}

enum class PostProcessCategory {
    ENHANCE,      // 增强：锐化、超分、降噪
    CORRECT,       // 校正：色彩、曝光、白平衡
    STYLIZE,      // 风格化：风格迁移、卡通化
    RESTORE,       // 修复：人脸修复、去模糊、去噪
    BLUR,          // 模糊：高斯、动感、径向
    TRANSFORM     // 变换：旋转、裁剪、透视
}

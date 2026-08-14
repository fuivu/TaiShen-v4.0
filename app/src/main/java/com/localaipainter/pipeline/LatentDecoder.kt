package com.localaipainter.pipeline

import android.graphics.Bitmap
import android.graphics.Color
import com.localaipainter.util.Logger

/**
 * 🎨 潜变量解码器 v2.0 —— 所有调度器共用
 *
 * 扩展点：
 *   - 新增后处理（gamma / 锐化 / 色彩映射）只需在本类加方法
 *   - 支持 HDR / Tonemapping / sRGB 转换
 *   - 支持自定义色彩空间（线性 / sRGB / Display-P3）
 */
object LatentDecoder {

    enum class ColorSpace { SRGB, LINEAR, DISPLAY_P3, HDR }

    private const val TAG = "LatentDecoder"

    // ============ 默认解码（sRGB） ============

    fun decode(latent: FloatArray, w: Int, h: Int): Bitmap {
        return decode(latent, w, h, ColorSpace.SRGB, gamma = 1.0f)
    }

    // ============ 高级解码 ============

    fun decode(
        latent: FloatArray,
        w: Int, h: Int,
        colorSpace: ColorSpace = ColorSpace.SRGB,
        gamma: Float = 1.0f,
        tonemap: Boolean = false
    ): Bitmap {
        val bm = Bitmap.createBitmap(w, h, Bitmap.Config.ARGB_8888)
        val pixels = IntArray(w * h)

        for (i in pixels.indices) {
            var r = latent[(i * 3) % latent.size]
            var g = latent[(i * 3 + 1) % latent.size]
            var b = latent[(i * 3 + 2) % latent.size]

            // 1. 归一化 [-1,1] → [0,1]
            r = (r * 0.5f + 0.5f).coerceIn(0f, 1f)
            g = (g * 0.5f + 0.5f).coerceIn(0f, 1f)
            b = (b * 0.5f + 0.5f).coerceIn(0f, 1f)

            // 2. Gamma 校正
            if (gamma != 1.0f) {
                r = r.pow(gamma)
                g = g.pow(gamma)
                b = b.pow(gamma)
            }

            // 3. Tonemapping（HDR → SDR）
            if (tonemap) {
                r = reinhard(r)
                g = reinhard(g)
                b = reinhard(b)
            }

            // 4. 色彩空间转换
            val (rr, gg, bb) = when (colorSpace) {
                ColorSpace.LINEAR -> {
                    // sRGB → Linear
                    Triple(srgbToLinear(r), srgbToLinear(g), srgbToLinear(b))
                }
                ColorSpace.DISPLAY_P3 -> {
                    // sRGB → Display-P3（简化矩阵）
                    val pr = r * 0.797f + g * 0.135f + b * 0.031f
                    val pg = r * 0.033f + g * 0.925f + b * 0.042f
                    val pb = r * 0.000f + g * 0.054f + b * 0.977f
                    Triple(pr.coerceIn(0f,1f), pg.coerceIn(0f,1f), pb.coerceIn(0f,1f))
                }
                else -> Triple(r, g, b)
            }

            val ri = (rr * 255f).toInt().coerceIn(0, 255)
            val gi = (gg * 255f).toInt().coerceIn(0, 255)
            val bi = (bb * 255f).toInt().coerceIn(0, 255)
            pixels[i] = Color.rgb(ri, gi, bi)
        }

        bm.setPixels(pixels, 0, w, 0, 0, w, h)
        Logger.v(TAG, "解码 ${w}x$h colorSpace=$colorSpace gamma=$gamma")
        return bm
    }

    // ============ 批量解码（用于批量生成） ============

    fun decodeBatch(latents: List<FloatArray>, w: Int, h: Int): List<Bitmap> {
        return latents.mapIndexed { idx, lat ->
            Logger.d(TAG, "批量解码 [$idx/${latents.size}]")
            decode(lat, w, h)
        }
    }

    // ============ 工具函数 ============

    private fun reinhard(x: Float): Float = x / (x + 1f)

    private fun srgbToLinear(x: Float): Float =
        if (x <= 0.04045f) x / 12.92f else ((x + 0.055f) / 1.055f).pow(2.4f)

    private fun Float.pow(e: Float): Float = kotlin.math.pow(this, e.toDouble()).toFloat()

    // ============ 调试用：导出 latent 统计 ============

    fun analyzeLatent(latent: FloatArray): String {
        if (latent.isEmpty()) return "空 latent"
        var min = Float.MAX_VALUE; var max = -Float.MAX_VALUE
        var sum = 0.0; var sumSq = 0.0
        for (v in latent) {
            if (v < min) min = v
            if (v > max) max = v
            sum += v; sumSq += v * v
        }
        val mean = (sum / latent.size).toFloat()
        val variance = (sumSq / latent.size - mean * mean).toFloat()
        val stddev = kotlin.math.sqrt(variance)
        return "min=$min max=$max mean=$mean stddev=$stddev"
    }
}

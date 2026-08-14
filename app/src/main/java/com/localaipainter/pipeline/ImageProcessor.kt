package com.localaipainter.pipeline

import android.content.Context
import android.graphics.*
import android.renderscript.*
import androidx.core.graphics.scale
import com.localaipainter.core.PostProcessorProvider
import com.localaipainter.core.PluginRegistry
import com.localaipainter.util.Logger

/**
 * 图像处理管线 —— 支持高斯模糊 / 缩放 / 调色板提取 / 后处理链
 *
 * 扩展点：注册新的 PostProcessorProvider 即可增加后处理算子，
 * 无需修改本类（开闭原则）。
 */
class ImageProcessor(private val context: Context) {

    companion object { private const val TAG = "ImageProcessor" }

    private var rs: RenderScript? = null

    init {
        try {
            if (RenderScript.isSupported()) {
                rs = RenderScript.create(context)
                Logger.i(TAG, "RenderScript 初始化成功（GPU 加速可用）")
            } else {
                Logger.w(TAG, "RenderScript 不支持，降级到 CPU 路径")
            }
        } catch (t: Throwable) {
            Logger.w(TAG, "RenderScript 初始化失败: ${t.message}")
        }
    }

    // ============ 高斯模糊 ============

    /**
     * 高斯模糊（RenderScript GPU 加速，天玑 Mali-G720 原生支持）
     * @param radius 模糊半径 1-25
     */
    fun gaussianBlur(bitmap: Bitmap, radius: Float = 15f): Bitmap {
        val r = radius.coerceIn(1f, 25f)
        // 优先走 RenderScript
        rs?.let { script ->
            try {
                val input = Allocation.createFromBitmap(script, bitmap)
                val output = Allocation.createTyped(script, input.type)
                val blur = ScriptIntrinsicBlur.create(script, Element.U8_4(script))
                blur.setRadius(r)
                blur.setInput(input)
                blur.forEach(output)
                val result = Bitmap.createBitmap(bitmap.width, bitmap.height, bitmap.config ?: Bitmap.Config.ARGB_8888)
                output.copyTo(result)
                input.destroy()
                output.destroy()
                blur.destroy()
                Logger.d(TAG, "高斯模糊完成 (RS) radius=$r")
                return result
            } catch (t: Throwable) {
                Logger.w(TAG, "RS 模糊失败，降级 CPU: ${t.message}")
            }
        }
        // CPU 降级
        return cpuGaussianBlur(bitmap, r)
    }

    private fun cpuGaussianBlur(src: Bitmap, radius: Float): Bitmap {
        val kernel = buildGaussianKernel(radius)
        val w = src.width; val h = src.height
        val inPx = IntArray(w * h); src.getPixels(inPx, 0, w, 0, 0, w, h)
        val tmp = IntArray(w * h); val out = IntArray(w * h)
        // 水平
        for (y in 0 until h) for (x in 0 until w) {
            var r=0f; var g=0f; var b=0f; var a=0f; var norm=0f
            for (k in kernel.indices) {
                val sx = (x + k - kernel.size/2).coerceIn(0, w-1)
                val p = inPx[y*w + sx]
                val w8 = kernel[k]
                r += ((p shr 16) and 0xFF) * w8; g += ((p shr 8) and 0xFF) * w8; b += (p and 0xFF) * w8; a += ((p shr 24) and 0xFF) * w8
                norm += w8
            }
            val i = y*w+x; tmp[i] = (a/norm).toInt().coerceIn(0,255) shl 24 or
                           ((r/norm).toInt().coerceIn(0,255) shl 16) or
                           ((g/norm).toInt().coerceIn(0,255) shl 8) or
                           (b/norm).toInt().coerceIn(0,255)
        }
        // 垂直
        val kHalf = kernel.size/2
        for (y in 0 until h) for (x in 0 until w) {
            var r=0f; var g=0f; var b=0f; var a=0f; var norm=0f
            for (k in kernel.indices) {
                val sy = (y + k - kHalf).coerceIn(0, h-1)
                val p = tmp[sy*w + x]
                val w8 = kernel[k]
                r += ((p shr 16) and 0xFF) * w8; g += ((p shr 8) and 0xFF) * w8; b += (p and 0xFF) * w8; a += ((p shr 24) and 0xFF) * w8
                norm += w8
            }
            out[y*w+x] = (a/norm).toInt().coerceIn(0,255) shl 24 or
                      ((r/norm).toInt().coerceIn(0,255) shl 16) or
                      ((g/norm).toInt().coerceIn(0,255) shl 8) or
                      (b/norm).toInt().coerceIn(0,255)
        }
        val result = Bitmap.createBitmap(w, h, src.config ?: Bitmap.Config.ARGB_8888)
        result.setPixels(out, 0, w, 0, 0, w, h)
        Logger.d(TAG, "高斯模糊完成 (CPU) radius=$radius")
        return result
    }

    private fun buildGaussianKernel(radius: Float): FloatArray {
        val r = radius.coerceAtLeast(1f)
        val size = (r * 3).toInt() * 2 + 1
        val kernel = FloatArray(size)
        val sigma = r / 3f
        val twoSigmaSq = 2 * sigma * sigma
        var sum = 0f
        for (i in kernel.indices) {
            val x = i - size/2
            kernel[i] = kotlin.math.exp(-(x*x).toFloat() / twoSigmaSq)
            sum += kernel[i]
        }
        for (i in kernel.indices) kernel[i] /= sum
        return kernel
    }

    // ============ 缩放 ============

    /**
     * 智能缩放：大图先预缩放再模糊，避免 RenderScript 输入过大
     * @param targetMax 目标最大边长
     */
    fun smartScale(bitmap: Bitmap, targetMax: Int = 1024): Bitmap {
        val maxSide = maxOf(bitmap.width, bitmap.height)
        if (maxSide <= targetMax) return bitmap
        val ratio = targetMax.toFloat() / maxSide
        val tw = (bitmap.width * ratio).toInt().coerceAtLeast(1)
        val th = (bitmap.height * ratio).toInt().coerceAtLeast(1)
        val scaled = bitmap.scale(tw, th)
        Logger.d(TAG, "缩放 ${bitmap.width}x${bitmap.height} → ${tw}x$th")
        return scaled
    }

    /**
     * 预缩放后模糊（用于背景图处理，性能最优）
     */
    fun blurForBackground(bitmap: Bitmap, radius: Float = 20f, maxSize: Int = 720): Bitmap {
        val scaled = smartScale(bitmap, maxSize)
        return gaussianBlur(scaled, radius)
    }

    // ============ 调色板提取 ============

    /**
     * 提取图像主色调，供 DynamicTheme 使用
     */
    fun extractDominantColor(bitmap: Bitmap): Int {
        val scaled = if (maxOf(bitmap.width, bitmap.height) > 128)
            bitmap.scale(64, 64) else bitmap
        val px = IntArray(scaled.width * scaled.height)
        scaled.getPixels(px, 0, scaled.width, 0, 0, scaled.width, scaled.height)
        val buckets = IntArray(16*16*16)
        var bestIdx = 0; var bestCount = 0
        for (p in px) {
            if ((p shr 24) and 0xFF < 128) continue
            val r = ((p shr 16) and 0xFF) shr 4
            val g = ((p shr 8) and 0xFF) shr 4
            val b = (p and 0xFF) shr 4
            val idx = (r shl 8) or (g shl 4) or b
            buckets[idx]++
            if (buckets[idx] > bestCount) { bestCount = buckets[idx]; bestIdx = idx }
        }
        val rr = ((bestIdx shr 8) and 0xFF) shl 4
        val gg = ((bestIdx shr 4) and 0xFF) shl 4
        val bb = (bestIdx and 0xFF) shl 4
        Logger.d(TAG, "主色调: #${"%02x%02x%02x".format(rr,gg,bb)}")
        return 0xFF000000.toInt() or (rr shl 16) or (gg shl 8) or bb
    }

    /**
     * 提取多色板（Material You 风格）
     */
    fun extractPalette(bitmap: Bitmap): List<Int> {
        val scaled = if (maxOf(bitmap.width, bitmap.height) > 256)
            bitmap.scale(128, 128) else bitmap
        val px = IntArray(scaled.width * scaled.height)
        scaled.getPixels(px, 0, scaled.width, 0, 0, scaled.width, scaled.height)
        val colors = px.filter { ((it shr 24) and 0xFF) > 128 }.distinct().take(8)
        Logger.d(TAG, "提取色板 ${colors.size} 色")
        return colors
    }

    // ============ 后处理链 ============

    /**
     * 执行注册的后处理链
     */
    fun applyPostProcessors(bitmap: Bitmap, names: List<String>, quality: Float = 0.5f): Bitmap {
        var result = bitmap
        for (name in names) {
            PluginRegistry.getPostProcessor(name)?.let { provider ->
                result = provider.run(context, result, quality)
                Logger.d(TAG, "后处理: $name")
            }
        }
        return result
    }

    // ============ 蒙版 / 融合（Inpainting） ============

    fun bitmapToLatent(bitmap: Bitmap, width: Int, height: Int): FloatArray {
        val scaled = Bitmap.createScaledBitmap(bitmap, width / 8, height / 8, true)
        val px = IntArray(scaled.width * scaled.height)
        scaled.getPixels(px, 0, scaled.width, 0, 0, scaled.width, scaled.height)
        val latent = FloatArray(px.size * 3)
        for (i in px.indices) {
            latent[i*3]   = ((px[i] shr 16) and 0xFF) / 127.5f - 1f
            latent[i*3+1] = ((px[i] shr 8) and 0xFF) / 127.5f - 1f
            latent[i*3+2] = (px[i] and 0xFF) / 127.5f - 1f
        }
        return latent
    }

    fun blendLatents(init: FloatArray, noise: FloatArray, strength: Float): FloatArray {
        val f = strength.coerceIn(0.05f, 1f)
        return FloatArray(init.size) { i -> init[i] * (1-f) + noise[i] * f }
    }

    fun applyMask(latent: FloatArray, mask: Bitmap, original: FloatArray, w: Int, h: Int): FloatArray {
        val scaled = Bitmap.createScaledBitmap(mask, w/8, h/8, true)
        val px = IntArray(scaled.width * scaled.height)
        scaled.getPixels(px, 0, scaled.width, 0, 0, scaled.width, scaled.height)
        val result = FloatArray(latent.size)
        var idx = 0
        for (i in px.indices) {
            val m = ((px[i] and 0xFF) / 255f).coerceIn(0f, 1f)
            val inv = 1f - m
            result[idx]   = latent[idx]   * m + original[idx]   * inv; idx++
            result[idx]   = latent[idx]   * m + original[idx]   * inv; idx++
            result[idx]   = latent[idx]   * m + original[idx]   * inv; idx++
            result[idx]   = latent[idx]   * m + original[idx]   * inv; idx++
        }
        return result
    }

    fun release() {
        try { rs?.destroy(); rs = null } catch (_: Throwable) {}
    }
}

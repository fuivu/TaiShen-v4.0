package com.localaipainter.quantize

import kotlin.math.abs
import kotlin.math.ln
import kotlin.math.roundToInt

/**
 * KL 散度量化 - INT8
 * 通过最小化原始分布和量化分布的 KL 散度来寻找最优裁剪范围
 */
class KLQuantizer(
    private val numBins: Int = 2048,
    private val numQuantBins: Int = 256
) : LocalQuantizer {

    override val type: QuantizationType = QuantizationType.INT8

    override fun quantize(weights: FloatArray): QuantizedTensor {
        if (weights.isEmpty()) {
            return QuantizedTensor(ByteArray(0), 1f, 0, QuantizationType.INT8, intArrayOf(0))
        }

        // 1. 构建直方图
        val min = weights.minOrNull() ?: 0f
        val max = weights.maxOrNull() ?: 0f
        val range = (max - min).coerceAtLeast(1e-8f)
        val binWidth = range / numBins

        val hist = IntArray(numBins)
        for (w in weights) {
            val bin = ((w - min) / binWidth).toInt().coerceIn(0, numBins - 1)
            hist[bin]++
        }

        // 2. 寻找最优裁剪阈值 (最小化 KL 散度)
        var bestThreshold = max
        var minKL = Float.MAX_VALUE
        val step = range / 100f

        for (t in 0..100) {
            val threshold = min + t * step
            val kl = computeKL(weights, threshold)
            if (kl < minKL) {
                minKL = kl
                bestThreshold = threshold
            }
        }

        // 3. 使用最优阈值量化
        val scale = (2f * bestThreshold).coerceAtLeast(1e-8f) / 255f
        val quantized = ByteArray(weights.size) { i ->
            val q = (weights[i] / scale).roundToInt().coerceIn(-128, 127)
            q.toByte()
        }

        return QuantizedTensor(quantized, scale, 0, QuantizationType.INT8, intArrayOf(weights.size))
    }

    override fun dequantize(tensor: QuantizedTensor): FloatArray {
        if (tensor.data.isEmpty()) return FloatArray(0)
        return FloatArray(tensor.data.size) { i ->
            tensor.data[i].toFloat() * tensor.scale
        }
    }

    override fun bits(): Int = 8

    private fun computeKL(data: FloatArray, threshold: Float): Float {
        if (data.isEmpty()) return 0f
        var kl = 0f
        val binWidth = (2f * threshold).coerceAtLeast(1e-8f) / numQuantBins
        val qHist = FloatArray(numQuantBins)

        for (d in data) {
            val clipped = d.coerceIn(-threshold, threshold)
            val bin = ((clipped + threshold) / binWidth).toInt().coerceIn(0, numQuantBins - 1)
            qHist[bin] += 1f
        }

        val total = data.size.toFloat()
        val uniformProb = 1f / numQuantBins
        for (i in qHist.indices) {
            val p = qHist[i] / total
            if (p > 0f) {
                kl += p * ln(p / uniformProb)
            }
        }
        return kl
    }
}

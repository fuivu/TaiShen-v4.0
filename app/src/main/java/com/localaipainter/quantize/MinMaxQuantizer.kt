package com.localaipainter.quantize

import kotlin.math.roundToInt

/**
 * Min-Max 线性量化 - INT8
 * 公式: q = round((fp32 - min) / (max - min) * 255) - 128
 */
class MinMaxQuantizer : LocalQuantizer {
    override val type: QuantizationType = QuantizationType.INT8

    override fun quantize(weights: FloatArray): QuantizedTensor {
        var min = Float.MAX_VALUE
        var max = Float.MIN_VALUE
        for (w in weights) {
            if (w < min) min = w
            if (w > max) max = w
        }
        val scale = if (max > min) (max - min) / 255f else 1f
        val zeroPoint = 0

        val quantized = ByteArray(weights.size) { i ->
            val q = ((weights[i] - min) / scale).roundToInt().coerceIn(-128, 127)
            q.toByte()
        }

        return QuantizedTensor(quantized, scale, zeroPoint, QuantizationType.INT8, intArrayOf(weights.size))
    }

    override fun dequantize(tensor: QuantizedTensor): FloatArray {
        return FloatArray(tensor.data.size) { i ->
            tensor.data[i].toFloat() * tensor.scale + tensor.zeroPoint
        }
    }

    override fun bits(): Int = 8
}

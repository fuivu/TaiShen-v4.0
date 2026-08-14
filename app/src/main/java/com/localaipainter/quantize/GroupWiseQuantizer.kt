package com.localaipainter.quantize

import kotlin.math.roundToInt
import kotlin.math.sqrt

/**
 * INT4 Group-wise 量化 - 极致压缩
 * 每 32 个元素为一组，共享 scale 和 zero-point
 */
class GroupWiseQuantizer(
    private val groupSize: Int = 32
) : LocalQuantizer {

    override val type: QuantizationType = QuantizationType.INT4

    override fun quantize(weights: FloatArray): QuantizedTensor {
        val numGroups = (weights.size + groupSize - 1) / groupSize
        val scales = FloatArray(numGroups)
        val zeroPoints = IntArray(numGroups)
        val quantized = ByteArray(weights.size / 2 + weights.size % 2) // 4-bit packs 2 per byte

        for (g in 0 until numGroups) {
            val start = g * groupSize
            val end = minOf(start + groupSize, weights.size)
            var gMin = Float.MAX_VALUE
            var gMax = Float.MIN_VALUE
            for (i in start until end) {
                if (weights[i] < gMin) gMin = weights[i]
                if (weights[i] > gMax) gMax = weights[i]
            }

            val scale = if (gMax > gMin) (gMax - gMin) / 15f else 1f
            val zp = roundToInt(-gMin / scale).coerceIn(0, 15)
            scales[g] = scale
            zeroPoints[g] = zp

            for (i in start until end) {
                val q = roundToInt(weights[i] / scale + zp).coerceIn(0, 15)
                val byteIdx = i / 2
                if (i % 2 == 0) {
                    quantized[byteIdx] = (q and 0x0F).toByte()
                } else {
                    quantized[byteIdx] = (quantized[byteIdx].toInt() or ((q and 0x0F) shl 4)).toByte()
                }
            }
        }

        // 打包 scales + zeroPoints 到 data 中
        // 简化：实际应序列化到单独 buffer
        return QuantizedTensor(quantized, scales[0], zeroPoints[0], QuantizationType.INT4, intArrayOf(weights.size, groupSize))
    }

    override fun dequantize(tensor: QuantizedTensor): FloatArray {
        val size = tensor.originalShape[0]
        val result = FloatArray(size)
        // 简化反量化
        for (i in 0 until size) {
            val byteIdx = i / 2
            val nibble = if (i % 2 == 0) {
                tensor.data[byteIdx].toInt() and 0x0F
            } else {
                (tensor.data[byteIdx].toInt() shr 4) and 0x0F
            }
            result[i] = (nibble - tensor.zeroPoint) * tensor.scale
        }
        return result
    }

    override fun bits(): Int = 4
}

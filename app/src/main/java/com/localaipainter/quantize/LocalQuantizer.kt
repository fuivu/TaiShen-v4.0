package com.localaipainter.quantize

import com.localaipainter.core.QuantizationType

/**
 * 本地量化器接口
 */
interface LocalQuantizer {
    val type: QuantizationType

    /**
     * 量化 FP32 权重为低精度
     */
    fun quantize(weights: FloatArray): QuantizedTensor

    /**
     * 反量化回 FP32 (用于验证)
     */
    fun dequantize(tensor: QuantizedTensor): FloatArray

    /**
     * 量化精度 (比特)
     */
    fun bits(): Int
}

data class QuantizedTensor(
    val data: ByteArray,
    val scale: Float,
    val zeroPoint: Int,
    val type: QuantizationType,
    val originalShape: IntArray
) {
    override fun equals(other: Any?): Boolean {
        if (this === other) return true
        if (javaClass != other?.javaClass) return false
        other as QuantizedTensor
        return data.contentEquals(other.data) && scale == other.scale
    }

    override fun hashCode(): Int = data.contentHashCode() + scale.hashCode()
}

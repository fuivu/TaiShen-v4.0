package com.localaipainter.pipeline

import com.localaipainter.engine.InferenceEngine
import com.localaipainter.lora.LoraWeight

/**
 * UNet 去噪网络封装
 */
class Unet(private val engine: InferenceEngine) {

    private var loaded = false

    fun load(modelPath: String) {
        nativeLoad(modelPath)
        loaded = true
    }

    fun predictNoise(
        latents: FloatArray,
        timestep: Int,
        condEmbedding: FloatArray,
        uncondEmbedding: FloatArray,
        cfgScale: Float,
        loras: List<LoraWeight> = emptyList()
    ): FloatArray {
        if (!loaded) error("Unet not loaded")

        // CFG: 组合条件和无条件预测
        val condNoise = nativePredict(latents, timestep, condEmbedding, loras.toTypedArray())
        val uncondNoise = nativePredict(latents, timestep, uncondEmbedding, emptyArray())

        // CFG 公式: noise = uncond + scale * (cond - uncond)
        return FloatArray(condNoise.size) { i ->
            uncondNoise[i] + cfgScale * (condNoise[i] - uncondNoise[i])
        }
    }

    fun release() {
        if (loaded) {
            nativeRelease()
            loaded = false
        }
    }

    private external fun nativeLoad(path: String)
    private external fun nativePredict(
        latents: FloatArray, timestep: Int,
        embedding: FloatArray, loras: Array<LoraWeight>
    ): FloatArray
    private external fun nativeRelease()
}

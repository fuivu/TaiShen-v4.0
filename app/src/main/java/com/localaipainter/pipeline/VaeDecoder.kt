package com.localaipainter.pipeline

import com.localaipainter.engine.InferenceEngine

/**
 * VAE Decoder - 将潜变量解码为 RGB 图像
 */
class VaeDecoder(private val engine: InferenceEngine) {

    private var loaded = false

    fun load(modelPath: String) {
        nativeLoad(modelPath)
        loaded = true
    }

    fun decode(latents: FloatArray): FloatArray {
        if (!loaded) error("VAE not loaded")
        return nativeDecode(latents)
    }

    fun release() {
        if (loaded) {
            nativeRelease()
            loaded = false
        }
    }

    private external fun nativeLoad(path: String)
    private external fun nativeDecode(latents: FloatArray): FloatArray
    private external fun nativeRelease()
}

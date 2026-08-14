package com.localaipainter.pipeline

import com.localaipainter.engine.InferenceEngine

/**
 * CLIP Text Encoder 封装
 */
class TextEncoder(private val engine: InferenceEngine) {

    private var loaded = false

    fun load(modelPath: String) {
        // 通过 JNI 调用原生推理
        nativeLoad(modelPath)
        loaded = true
    }

    fun encode(text: String): FloatArray {
        if (!loaded) error("TextEncoder not loaded")
        return nativeEncode(text)
    }

    fun encodeBatch(texts: List<String>): Array<FloatArray> {
        return texts.map { encode(it) }.toTypedArray()
    }

    fun release() {
        if (loaded) {
            nativeRelease()
            loaded = false
        }
    }

    private external fun nativeLoad(path: String)
    private external fun nativeEncode(text: String): FloatArray
    private external fun nativeRelease()
}

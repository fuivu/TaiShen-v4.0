package com.localaipainter.engine

import kotlinx.coroutines.channels.awaitClose
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.callbackFlow
import kotlinx.coroutines.flow.flowOn
import kotlinx.coroutines.Dispatchers

/**
 * 进度适配器 — 将 JNI 回调桥接为 Kotlin Flow
 */
class ProgressAdapter : JNIBridge.ProgressCallback {

    data class ProgressUpdate(
        val step: Int,
        val totalSteps: Int,
        val progress: Float,  // 0.0 ~ 1.0
        val previewPath: String?,
    )

    data class CompleteEvent(
        val success: Boolean,
        val outputPath: String?,
        val message: String?,
    )

    data class ErrorEvent(
        val code: Int,
        val message: String?,
    )

    private val _events = callbackFlow<Any> {
        // 这个 flow 由 onStep/onComplete/onError 驱动
        awaitClose { /* 清理 */ }
    }

    val events: Flow<Any> = _events.flowOn(Dispatchers.Default)

    // 当前进度缓存
    @Volatile var currentStep: Int = 0
        private set

    @Volatile var totalSteps: Int = 0
        private set

    @Volatile var progress: Float = 0f
        private set

    @Volatile var lastPreviewPath: String? = null
        private set

    @Volatile var isComplete: Boolean = false
        private set

    @Volatile var isSuccess: Boolean = false
        private set

    @Volatile var outputPath: String? = null
        private set

    @Volatile var errorMessage: String? = null
        private set

    // ============ JNI 回调实现 ============

    override fun onStep(step: Int, totalSteps: Int, progress: Float, previewPath: String?) {
        this.currentStep = step
        this.totalSteps = totalSteps
        this.progress = progress.coerceIn(0f, 1f)
        this.lastPreviewPath = previewPath
    }

    override fun onComplete(success: Boolean, outputPath: String?, message: String?) {
        this.isComplete = true
        this.isSuccess = success
        this.outputPath = outputPath
        this.errorMessage = message
    }

    override fun onError(errorCode: Int, message: String?) {
        this.isComplete = true
        this.isSuccess = false
        this.errorMessage = "Error $errorCode: ${message ?: "Unknown"}"
    }

    // ============ 工具方法 ============

    fun reset() {
        currentStep = 0
        totalSteps = 0
        progress = 0f
        lastPreviewPath = null
        isComplete = false
        isSuccess = false
        outputPath = null
        errorMessage = null
    }

    fun progressPercent(): Int = (progress * 100).toInt()

    fun isRunning(): Boolean = !isComplete && progress < 1f
}

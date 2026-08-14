package com.localaipainter.xunji

import android.util.Log

/**
 * ══════════════════════════════════════════════════════════════
 *  迅集 (XunJi) — 量化执行器  v4.0 TaiShen
 *  精度切换 · 权重管理 · 性能统计 · 降级链执行
 * ══════════════════════════════════════════════════════════════
 */
class XunJiQuantExecutor(private val monitor: XunJiMemoryMonitor) {

    private var currentLevel: QuantLevel = QuantLevel.FP16
    private var prefs: XunJiPrefs = XunJiPrefs()
    private val stats = mutableMapOf<QuantLevel, PerfStat>()
    private var oomCount = 0

    data class PerfStat(
        var totalRuns: Int = 0,
        var totalMs: Long = 0,
        var oomCount: Int = 0,
        var successCount: Int = 0
    ) {
        val avgMs: Double get() = if (totalRuns > 0) totalMs.toDouble() / totalRuns else 0.0
        val oomRate: Double get() = if (totalRuns > 0) oomCount.toDouble() / totalRuns else 0.0
    }

    // ─── 公开 API ────────────────────────────────────────

    fun setPrecision(level: QuantLevel) {
        currentLevel = level
        nativeSetPrecision(level.name)
        Log.i(TAG, "XunJi precision → ${level.label}")
    }

    fun getPrecision(): QuantLevel = currentLevel

    fun setPrefs(p: XunJiPrefs) { prefs = p }

    fun getPrefs(): XunJiPrefs = prefs

    /** 加载量化模型权重 */
    fun loadQuantizedModel(path: String, level: QuantLevel = currentLevel): Long {
        return nativeLoadModel(path, level.name)
    }

    /** 释放权重句柄 */
    fun freeHandle(handle: Long) { nativeFreeHandle(handle) }

    /** 执行一次推理并统计 */
    fun runInference(handle: Long, input: FloatArray): FloatArray? {
        val t0 = System.currentTimeMillis()
        return try {
            val out = nativeRun(handle, input)
            val dt = System.currentTimeMillis() - t0
            recordSuccess(currentLevel, dt)
            out
        } catch (e: OutOfMemoryError) {
            recordOom(currentLevel)
            handleOom()
            null
        }
    }

    fun getStats(level: QuantLevel): PerfStat = stats.getOrPut(level) { PerfStat() }

    fun getAllStats(): Map<QuantLevel, PerfStat> = stats.toMap()

    fun resetStats() { stats.clear(); oomCount = 0 }

    fun oomLocked(): Boolean = prefs.oomLock && oomCount >= OOM_LOCK_THRESHOLD

    // ─── 内部 ────────────────────────────────────────────

    private fun recordSuccess(lvl: QuantLevel, ms: Long) {
        val s = stats.getOrPut(lvl) { PerfStat() }
        s.totalRuns++; s.totalMs += ms; s.successCount++
    }

    private fun recordOom(lvl: QuantLevel) {
        val s = stats.getOrPut(lvl) { PerfStat() }
        s.totalRuns++; s.oomCount++
        oomCount++
    }

    private fun handleOom() {
        Log.w(TAG, "OOM at ${currentLevel.label}, oomCount=$oomCount")
        if (oomLocked()) {
            setPrecision(prefs.oomLockLevel)
            Log.e(TAG, "OOM locked → ${prefs.oomLockLevel.label}")
            return
        }
        // 降级链
        val order = listOf(QuantLevel.INT8, QuantLevel.INT4, QuantLevel.INT2)
        val idx = order.indexOf(currentLevel)
        if (idx in 0 until order.size - 1) setPrecision(order[idx + 1])
    }

    // ─── JNI ─────────────────────────────────────────────

    private external fun nativeSetPrecision(level: String)
    private external fun nativeLoadModel(path: String, level: String): Long
    private external fun nativeFreeHandle(handle: Long)
    private external fun nativeRun(handle: Long, input: FloatArray): FloatArray

    companion object {
        private const val TAG = "XunJiExecutor"
        private const val OOM_LOCK_THRESHOLD = 3
        init { System.loadLibrary("xunji") }
    }
}

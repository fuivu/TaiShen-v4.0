package com.localaipainter.xunji

import android.content.Context
import android.content.SharedPreferences
import org.json.JSONObject

/**
 * ══════════════════════════════════════════════════════════════
 *  迅集 (XunJi) — 调度器  v4.0 TaiShen
 *  自动决策 · 降级链 · OOM 锁定 · 持久化
 * ══════════════════════════════════════════════════════════════
 */
class XunJiScheduler(private val ctx: Context, private val monitor: XunJiMemoryMonitor) {

    private val executor = XunJiQuantExecutor(monitor)
    private val prefs: SharedPreferences =
        ctx.getSharedPreferences("xunji_prefs", Context.MODE_PRIVATE)

    // 降级链：从高到低
    private val fallbackChain = listOf(
        QuantLevel.FP16, QuantLevel.INT8, QuantLevel.INT4, QuantLevel.INT2
    )

    init { loadPersistedPrefs() }

    // ─── 核心：自动选择最优量化等级 ───────────────────────

    fun autoSelect(): QuantLevel {
        val snap = monitor.sample()
        val saved = executor.getPrefs()

        // 1. OOM 锁定优先
        if (executor.oomLocked()) {
            Log.w(TAG, "OOM locked → ${saved.oomLockLevel.label}")
            return saved.oomLockLevel
        }

        // 2. 用户手动指定
        val manual = saved.manualLevel
        if (manual != null && isFeasible(manual, snap)) {
            return manual
        }

        // 3. Native 自动选择
        val nativePick = nativeAutoSelect(
            snap.totalKb, snap.availableKb,
            monitor.gpuTotalMB().toLong() * 1024, 0L
        )
        val nativeLevel = QuantLevel.values().firstOrNull { it.name == nativePick }
        if (nativeLevel != null && isFeasible(nativeLevel, snap)) return nativeLevel

        // 4. Kotlin 启发式降级
        val recommended = monitor.recommendedLevel(snap)
        for (lvl in buildFallbackChain(recommended)) {
            if (isFeasible(lvl, snap)) return lvl
        }
        return QuantLevel.INT2
    }

    fun applyAuto(): QuantLevel {
        val lvl = autoSelect()
        executor.setPrecision(lvl)
        return lvl
    }

    // ─── 降级链 ────────────────────────────────────────────

    fun buildFallbackChain(preferred: QuantLevel): List<QuantLevel> {
        val idx = fallbackChain.indexOf(preferred)
        return if (idx >= 0) fallbackChain.subList(idx, fallbackChain.size) else fallbackChain
    }

    /** 执行推理，遇 OOM 自动沿降级链重试 */
    fun runWithFallback(handle: Long, input: FloatArray): FloatArray? {
        val result = executor.runInference(handle, input)
        if (result != null) return result
        // OOM → 降级重试
        for (lvl in buildFallbackChain(executor.getPrecision()).drop(1)) {
            Log.w(TAG, "Falling back → ${lvl.label}")
            executor.setPrecision(lvl)
            val h2 = executor.loadQuantizedModel("default", lvl)
            if (h2 != 0L) {
                val retry = executor.runInference(h2, input)
                executor.freeHandle(h2)
                if (retry != null) return retry
            }
        }
        return null
    }

    // ─── 可行性校验 ────────────────────────────────────────

    fun isFeasible(level: QuantLevel, snap: MemSnapshot = monitor.current()): Boolean {
        val budget = snap.safeBudgetKb()
        val modelKb = estimateModelKb(level)
        val margin = budget - modelKb
        return margin > budget * 0.15  // 至少留 15%
    }

    private fun estimateModelKb(level: QuantLevel): Long {
        // 典型 SD1.5 UNet ~1.2GB FP32 → 按比例估算
        val baseFp32Mb = 1200
        return (baseFp32Mb * level.memRatio).toLong() * 1024
    }

    // ─── 用户偏好持久化 ────────────────────────────────────

    fun savePrefs(p: XunJiPrefs) {
        executor.setPrefs(p)
        prefs.edit().apply {
            putString("manual_level", p.manualLevel?.name)
            putBoolean("allow_fallback", p.allowFallback)
            putString("min_level", p.minLevel.name)
            putString("max_level", p.maxLevel.name)
            putBoolean("oom_lock", p.oomLock)
            putString("oom_lock_level", p.oomLockLevel.name)
            apply()
        }
    }

    private fun loadPersistedPrefs() {
        val p = XunJiPrefs(
            manualLevel = prefs.getString("manual_level", null)?.let { QuantLevel.valueOf(it) },
            allowFallback = prefs.getBoolean("allow_fallback", true),
            minLevel = QuantLevel.valueOf(prefs.getString("min_level", "INT2")!!),
            maxLevel = QuantLevel.valueOf(prefs.getString("max_level", "FP32")!!),
            oomLock = prefs.getBoolean("oom_lock", false),
            oomLockLevel = QuantLevel.valueOf(prefs.getString("oom_lock_level", "INT2")!!)
        )
        executor.setPrefs(p)
    }

    // ─── 状态查询 ────────────────────────────────────────────

    fun statusJson(): String {
        val s = monitor.current()
        val obj = JSONObject().apply {
            put("level", executor.getPrecision().name)
            put("mem_used_ratio", s.usedRatio)
            put("mem_available_kb", s.availableKb)
            put("gpu_total_mb", monitor.gpuTotalMB())
            put("gpu_method", monitor.gpuMethod())
            put("oom_locked", executor.oomLocked())
            put("confidence_r2", monitor.confidenceR2())
        }
        return obj.toString(2)
    }

    // ─── JNI ───────────────────────────────────────────────

    private external fun nativeAutoSelect(
        totalKb: Long, availKb: Long, gpuTotalKb: Long, gpuAvailKb: Long
    ): String

    companion object {
        private const val TAG = "XunJiScheduler"
        init { System.loadLibrary("xunji") }
    }
}

package com.localaipainter.hunyuan

import android.util.Log
import org.json.JSONObject
import java.util.concurrent.atomic.AtomicInteger

/**
 * ═════════════════════════════════════════════════════════════
 *  混元 (HunYuan) — 核心调度引擎  v4.0 TaiShen
 *  自动算法选择 · 防抖动 · 降级链 · 自定义配置 CRUD
 * ═════════════════════════════════════════════════════════════
 */
class HunYuanEngine(ctx: Context) {

    private val monitor = HunYuanPerformanceMonitor(ctx)
    private val prefs = ctx.getSharedPreferences("hunyuan_prefs", Context.MODE_PRIVATE)

    // 自定义配置库
    private val customConfigs = mutableListOf<CustomAlgoConfig>()
    private var activeCustom: CustomAlgoConfig? = null

    // 当前策略
    private var strategy: ScheduleStrategy = ScheduleStrategy.ADAPTIVE

    // 稳定性控制：连续 N 次才切换
    private var lastPick: AlgoType? = null
    private var stableCount = 0

    // 统计
    private val runCount = IntArray(AlgoType.count())
    private val runTimeMs = DoubleArray(AlgoType.count())
    private val failCount = IntArray(AlgoType.count())
    private var totalRuns = AtomicInteger(0)

    // 降级链
    private val fallbackOrder = listOf(
        AlgoType.NPU_HARDWARE, AlgoType.SPECULATIVE, AlgoType.FP16_HALF,
        AlgoType.FUSED_OPS, AlgoType.SPARSE_ATTN, AlgoType.INT8_QUANT,
        AlgoType.MIXED_PRECISION, AlgoType.INT4_EXTREME,
        AlgoType.CASCADE_SR, AlgoType.FP32_FULL
    )

    // 默认自定义配置
    init {
        loadCustomConfigs()
        if (customConfigs.isEmpty()) {
            customConfigs.add(CustomAlgoConfig.default())
            saveCustomConfigs()
        }
    }

    // ─── 核心：自动选择 ────────────────────────────────────

    fun autoPick(): AlgoType {
        val snap = monitor.sample()
        val ranked = sortByStrategy(strategy, snap)

        // 稳定性控制：连续 3 次才切换
        val top = ranked.first()
        if (top != lastPick) {
            stableCount++
            if (stableCount >= STABILITY_THRESHOLD) {
                lastPick = top
                stableCount = 0
            }
        } else stableCount = 0

        // 校验可行性
        val chosen = lastPick ?: top
        return if (isFeasible(chosen, snap)) chosen else fallback(chosen, snap)
    }

    private fun sortByStrategy(s: ScheduleStrategy, snap: PerfSnapshot): List<AlgoType> {
        val list = AlgoType.values().toMutableList()
        list.sortByDescending { score(it, s, snap) }
        return list
    }

    private fun score(a: AlgoType, s: ScheduleStrategy, snap: PerfSnapshot): Float {
        return when (s) {
            ScheduleStrategy.MAX_SPEED  -> a.speedTier.toFloat()
            ScheduleStrategy.MAX_QUALITY -> a.qualityScore * 5f
            ScheduleStrategy.POWER_SAVE  -> a.powerEff * 5f
            ScheduleStrategy.BALANCED    -> {
                val tempFactor = if (snap.cpuTempC > 70) 0.3f else if (snap.cpuTempC > 55) 0.6f else 1.0f
                a.qualityScore * tempFactor + a.speedTier * 0.3f + a.powerEff * tempFactor
            }
            ScheduleStrategy.ADAPTIVE   -> {
                var sc = a.qualityScore * 1.5f
                if (snap.batteryPct < 20f) sc += a.powerEff * 2f
                if (snap.cpuTempC > 65f) sc -= a.speedTier * 0.3f
                if (snap.memUsedRatio > 0.8f) sc -= (5 - a.speedTier) * 0.5f
                if (snap.charging) sc += 0.5f
                sc
            }
            ScheduleStrategy.CUSTOM -> {
                val order = activeCustom?.algoOrder ?: customConfigs.first().algoOrder
                // 按自定义顺序给分
                order.indexOf(a).toFloat()
            }
        }
    }

    private fun isFeasible(a: AlgoType, snap: PerfSnapshot): Boolean {
        // 内存检查
        val memOk = when (a) {
            AlgoType.FP32_FULL -> snap.availMemKb > 2000_000L
            AlgoType.CASCADE_SR -> snap.availMemKb > 1500_000L
            AlgoType.MIXED_PRECISION -> snap.availMemKb > 1000_000L
            AlgoType.INT4_EXTREME -> snap.availMemKb > 300_000L
            else -> snap.availMemKb > 500_000L
        }
        // 温度检查
        val tempOk = when (a) {
            AlgoType.FP32_FULL, AlgoType.CASCADE_SR -> snap.cpuTempC < 60f
            else -> snap.cpuTempC < 80f
        }
        // 电量检查
        val battOk = if (a.powerEff < 0.4f) snap.batteryPct > 15f else true
        return memOk && tempOk && battOk
    }

    private fun fallback(from: AlgoType, snap: PerfSnapshot): AlgoType {
        val idx = fallbackOrder.indexOf(from).coerceAtLeast(0)
        for (i in idx + 1 until fallbackOrder.size) {
            if (isFeasible(fallbackOrder[i], snap)) return fallbackOrder[i]
        }
        return AlgoType.INT4_EXTREME
    }

    // ─── 执行 ────────────────────────────────────────────────

    fun execute(input: FloatArray): FloatArray? {
        val algo = autoPick()
        val t0 = System.currentTimeMillis()
        return try {
            val out = nativeExecute(handle, input) ?: return null
            val dt = System.currentTimeMillis() - t0
            recordRun(algo, dt, success = true)
            out
        } catch (e: OutOfMemoryError) {
            recordRun(algo, 0, success = false)
            // 降级重试
            for (fb in fallbackOrder.dropWhile { it != algo }.drop(1)) {
                Log.w(TAG, "HunYuan fallback → ${fb.label}")
                val h2 = fb.id
                val retry = nativeExecute(handle, input)
                if (retry != null) { recordRun(fb, 0, success = true); return retry }
            }
            null
        }
    }

    // ─── 策略管理 ────────────────────────────────────────────

    fun setStrategy(s: ScheduleStrategy) {
        strategy = s
        lastPick = null
        stableCount = 0
        prefs.edit().putInt("strategy_id", s.id).apply()
        Log.i(TAG, "HunYuan strategy → ${s.label}")
    }

    fun getStrategy(): ScheduleStrategy = strategy

    fun setCustomConfig(name: String) {
        customConfigs.firstOrNull { it.name == name }?.let {
            activeCustom = it
            strategy = ScheduleStrategy.CUSTOM
            prefs.edit().putString("active_custom", name).apply()
        }
    }

    fun addCustomConfig(cfg: CustomAlgoConfig) {
        customConfigs.removeAll { it.name == cfg.name }
        customConfigs.add(cfg)
        saveCustomConfigs()
    }

    fun removeCustomConfig(name: String) {
        customConfigs.removeAll { it.name == name }
        if (activeCustom?.name == name) activeCustom = null
        saveCustomConfigs()
    }

    fun listCustomConfigs(): List<CustomAlgoConfig> = customConfigs.toList()

    fun getActiveCustom(): CustomAlgoConfig? = activeCustom

    // ─── 统计 ────────────────────────────────────────────────

    fun getStatsJson(): String {
        val obj = JSONObject().apply {
            put("strategy", strategy.label)
            put("total_runs", totalRuns.get())
            put("per_algo", JSONObject().apply {
                for (a in AlgoType.values()) {
                    put(a.name, JSONObject().apply {
                        put("runs", runCount[a.id])
                        put("avg_ms", if (runCount[a.id] > 0) runTimeMs[a.id] / runCount[a.id] else 0.0)
                        put("fails", failCount[a.id])
                    })
                }
            })
        }
        return obj.toString(2)
    }

    fun resetStats() {
        for (i in runCount.indices) { runCount[i] = 0; runTimeMs[i] = 0.0; failCount[i] = 0 }
        totalRuns.set(0)
    }

    // ─── 与迅集联动 ────────────────────────────────────────

    fun currentQuantLevel(): Int {
        // 根据当前算法推荐量化等级
        val algo = lastPick ?: autoPick()
        return when (algo) {
            AlgoType.FP32_FULL -> 32
            AlgoType.CASCADE_SR -> 32
            AlgoType.MIXED_PRECISION -> 16
            AlgoType.FP16_HALF -> 16
            AlgoType.NPU_HARDWARE -> 8
            AlgoType.INT8_QUANT -> 8
            AlgoType.INT4_EXTREME -> 4
            else -> 8
        }
    }

    // ─── 内部 ────────────────────────────────────────────────

    private fun recordRun(algo: AlgoType, ms: Long, success: Boolean) {
        val id = algo.id
        if (success) { runCount[id]++; runTimeMs[id] += ms; totalRuns.incrementAndGet() }
        else failCount[id]++
    }

    private fun saveCustomConfigs() {
        val arr = org.json.JSONArray()
        customConfigs.forEach { arr.put(it.toJson()) }
        prefs.edit().putString("custom_configs_json", arr.toString()).apply()
    }

    private fun loadCustomConfigs() {
        val json = prefs.getString("custom_configs_json", null) ?: return
        try {
            val arr = org.json.JSONArray(json)
            for (i in 0 until arr.length()) {
                val o = arr.getJSONObject(i)
                val name = o.optString("name", "config_$i")
                val minB = o.optInt("min_battery", 15)
                val maxT = o.optInt("max_temp", 75)
                val npu = o.optBoolean("prefer_npu", false)
                // 简化：用默认顺序
                customConfigs.add(CustomAlgoConfig(name, CustomAlgoConfig.default().algoOrder, minB, maxT, npu))
            }
        } catch (_: Exception) {}
        val active = prefs.getString("active_custom", null)
        activeCustom = customConfigs.firstOrNull { it.name == active }
        val sid = prefs.getInt("strategy_id", ScheduleStrategy.ADAPTIVE.id)
        strategy = ScheduleStrategy.fromId(sid)
    }

    // ─── JNI 句柄 ──────────────────────────────────────────

    private var handle: Long = 0L

    init { handle = nativeInit() }

    private external fun nativeInit(): Long
    private external fun nativeShutdown(handle: Long)
    private external fun nativeSetStrategy(handle: Long, name: String)
    private external fun nativeGetStrategy(handle: Long): String
    private external fun nativeSetCustomOrder(handle: Long, order: IntArray)
    private external fun nativeSetDeviceCap(handle: Long, cores: Int, freq: Float, ramGb: Int, npu: Boolean, gpu: Boolean, vulkan: Boolean, socTier: Int)
    private external fun nativeFeedPerf(handle: Long, cpuU: Float, cpuT: Float, memR: Float, batP: Float, batT: Float, charging: Boolean, gpuU: Float, availKb: Long)
    private external fun nativePick(handle: Long): Int
    private external fun nativeExecute(handle: Long, input: FloatArray): FloatArray?
    private external fun nativeStatus(handle: Long): String
    private external fun nativeResetStats(handle: Long)
    private external fun nativeAlgoCount(): Int
    private external fun nativeAlgoName(id: Int): String
    private external fun nativeAlgoQuality(id: Int): Float
    private external fun nativeAlgoSpeed(id: Int): Float
    private external fun nativeAlgoPower(id: Int): Float
    private external fun nativeSort(handle: Long, strategyName: String): IntArray

    companion object {
        private const val TAG = "HunYuanEngine"
        private const val STABILITY_THRESHOLD = 3
        init { System.loadLibrary("hunyuan") }
    }
}

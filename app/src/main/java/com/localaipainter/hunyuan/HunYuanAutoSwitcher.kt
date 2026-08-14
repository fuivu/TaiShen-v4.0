package com.localaipainter.hunyuan

import kotlinx.coroutines.*

/**
 * ═════════════════════════════════════════════════════════════
 *  混元 (HunYuan) — 自动切换守护器  v4.0 TaiShen
 *  后台协程 · 自适应采样间隔 · 阈值触发
 * ═════════════════════════════════════════════════════════════
 */
class HunYuanAutoSwitcher(private val engine: HunYuanEngine) {

    private var job: Job? = null
    private var intervalMs = 5000L

    /** 启动后台监控 */
    fun start(scope: CoroutineScope) {
        if (job?.isActive == true) return
        job = scope.launch {
            while (isActive) {
                val snap = engine.monitor.sample()
                // 自适应采样间隔
                intervalMs = when {
                    snap.cpuTempC > 75f -> 2000L
                    snap.memUsedRatio > 0.85f -> 3000L
                    snap.batteryPct < 15f -> 10000L
                    else -> 5000L
                }
                // 触发自动选择
                val pick = engine.autoPick()
                // 如果温度/内存临界，强制降级
                if (snap.cpuTempC > 80f || snap.memUsedRatio > 0.9f) {
                    // 降级到最节能
                    engine.setStrategy(ScheduleStrategy.POWER_SAVE)
                }
                delay(intervalMs)
            }
        }
    }

    fun stop() { job?.cancel() }

    fun isRunning(): Boolean = job?.isActive == true
}

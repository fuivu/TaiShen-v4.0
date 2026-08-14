package com.localaipainter.core

import android.content.Context
import android.os.BatteryManager
import android.os.Build
import com.localaipainter.engine.DeviceDetector
import com.localaipainter.util.Logger
import kotlin.math.exp
import kotlin.math.min
import java.util.ArrayDeque

/**
 * DevicePowerScore — 太神架构核心算法
 *
 * 动态推理路径决策引擎。
 * 每秒计算一次设备综合算力评分，决定推理路径：
 *
 *   Score ≥ 0.85 (旗舰机)    → LOCAL_INT4_VULKAN 纯本地自研架构
 *   0.4 ≤ Score < 0.85 (中端) → HYBRID_PIPELINE   本地轻量+云端精调
 *   Score < 0.4 (旧设备/弱网) → CLOUD_NATIVE      100%云端协议
 *
 * 评分维度（等权重归一化后加权）：
 *   1. 硬件算力  (SoC 等级 + GPU 等级 + NPU 可用性)
 *   2. 网络 RTT   (延迟越低分越高)
 *   3. 电池温度  (过高触发降频，分数衰减)
 *   4. 剩余电量  (低于 15% 触发节能模式)
 *   5. 当前负载  (CPU/Memory 压力)
 *
 * 设计哲学：架构即法则。硬件是执行者，太神是立法者。
 */
class DevicePowerScore(private val context: Context) {

    companion object {
        private const val TAG = "DevicePowerScore"

        // ─── 路径常量 ────────────────────────────
        const val PATH_LOCAL_VULKAN = "LOCAL_INT4_VULKAN"
        const val PATH_HYBRID = "HYBRID_PIPELINE"
        const val PATH_CLOUD = "CLOUD_NATIVE"

        // ─── 阈值 ────────────────────────────────
        const val THRESHOLD_FLAGSHIP = 0.85f
        const val THRESHOLD_MID = 0.4f

        // ─── SoC 算力基准分（参考值，越高越强）──
        private val SOC_SCORES: Map<String, Float> = mapOf(
            // 骁龙旗舰
            "Qualcomm Snapdragon 8 Gen 3" to 1.0f,
            "Qualcomm Snapdragon 8 Gen 2" to 0.88f,
            "Qualcomm Snapdragon 8+ Gen 1" to 0.82f,
            "Qualcomm Snapdragon 8 Gen 1" to 0.75f,
            // 天玑旗舰
            "MediaTek Dimensity 9400" to 0.98f,
            "MediaTek Dimensity 9200" to 0.90f,
            "MediaTek Dimensity 9000" to 0.85f,
            "MediaTek Dimensity 8400" to 0.82f,
            "MediaTek Dimensity 8300" to 0.78f,
            // 中端
            "Qualcomm Snapdragon 7 Gen 3" to 0.65f,
            "Qualcomm Snapdragon 7s Gen 2" to 0.58f,
            "MediaTek Dimensity 7200" to 0.60f,
            "MediaTek Dimensity 7000" to 0.55f,
            // 旧设备/入门
            "Qualcomm Snapdragon 695" to 0.38f,
            "Qualcomm Snapdragon 665" to 0.25f,
            "MediaTek Dimensity 700" to 0.30f,
            "MediaTek Helio G85" to 0.22f,
        )

        // ─── 评分历史（用于趋势分析）────────────
        private const val HISTORY_SIZE = 30  // 30秒滑动窗口
    }

    // ─── 内部状态 ──────────────────────────────
    private val socDetector = DeviceDetector(context)
    private val scoreHistory = ArrayDeque<Float>(HISTORY_SIZE)
    private var lastNetworkRttMs: Float = 50f  // 默认假设良好网络
    private var consecutiveHighScoreDays: Int = 0

    /**
     * 计算当前 DevicePowerScore (0.0 ~ 1.0)
     *
     * @param networkRttMs 最近一次网络 RTT（毫秒），-1 表示未知
     * @return 综合评分
     */
    fun compute(networkRttMs: Float = -1f): Float {
        // 1. 硬件算力分 (0~1)
        val hwScore = computeHardwareScore()

        // 2. 网络分 (0~1, 越高越好)
        val netScore = computeNetworkScore(
            if (networkRttMs > 0) networkRttMs else lastNetworkRttMs
        )

        // 3. 电池温度分 (0~1)
        val tempScore = computeBatteryTempScore()

        // 4. 电量分 (0~1)
        val batteryScore = computeBatteryLevelScore()

        // 5. 负载分 (0~1, 越高表示压力越小)
        val loadScore = computeLoadScore()

        // ─── 加权融合 ────────────────────────────
        // 硬件权重最高（太神架构核心：本地算力优先）
        val weights = floatArrayOf(0.35f, 0.20f, 0.15f, 0.15f, 0.15f)
        val scores = floatArrayOf(hwScore, netScore, tempScore, batteryScore, loadScore)

        var finalScore = 0f
        for (i in scores.indices) {
            finalScore += weights[i] * scores[i]
        }

        // ─── 滑动平均（抑制抖动）────────────────
        scoreHistory.addLast(finalScore)
        if (scoreHistory.size > HISTORY_SIZE) {
            scoreHistory.removeFirst()
        }
        val smoothed = scoreHistory.average().toFloat()

        Logger.d(TAG, String.format(
            "Score=%.3f (hw=%.2f net=%.2f temp=%.2f bat=%.2f load=%.2f)",
            smoothed, hwScore, netScore, tempScore, batteryScore, loadScore
        ))

        return smoothed.coerceIn(0f, 1f)
    }

    /**
     * 根据评分决定推理路径
     */
    fun decidePath(score: Float = compute()): String {
        return when {
            score >= THRESHOLD_FLAGSHIP -> PATH_LOCAL_VULKAN
            score >= THRESHOLD_MID -> PATH_HYBRID
            else -> PATH_CLOUD
        }
    }

    /**
     * 获取路径的人类可读描述（统一 UI 展示）
     * 所有设备显示"极速响应中"，不暴露性能差异
     */
    fun getPathLabel(path: String): String {
        return when (path) {
            PATH_LOCAL_VULKAN -> "⚡ 太神引擎极速响应中"
            PATH_HYBRID -> "⚡ 太神引擎极速响应中"
            PATH_CLOUD -> "⚡ 太神引擎极速响应中"
            else -> "⚡ 太神引擎极速响应中"
        }
    }

    /**
     * 获取设备信息标签（统一显示"太神AI引擎 4.0"）
     * 红线2：所有机型均显示统一身份
     */
    fun getEngineIdentity(): String = "太神 AI 引擎 4.0"

    /**
     * 记录网络 RTT（由网络模块定期调用）
     */
    fun reportNetworkRtt(rttMs: Float) {
        lastNetworkRttMs = rttMs
    }

    /**
     * 检查是否触发 LegacyKiller 条件
     * 连续 7 天 Score > 0.85 → 触发自毁协议
     */
    fun shouldTriggerLegacyKiller(consecutiveDays: Int): Boolean {
        return consecutiveDays >= 7
    }

    // ═════════════════════════════════════════════
    //  子评分函数
    // ═════════════════════════════════════════════

    private fun computeHardwareScore(): Float {
        val deviceInfo = socDetector.detect()
        val baseScore = SOC_SCORES[deviceInfo.chipset] ?: 0.35f

        // NPU 加成
        val npuBonus = if (deviceInfo.npuAvailable) 0.08f else 0f

        // Vulkan 加成（太神架构核心加速路径）
        val vulkanBonus = if (deviceInfo.hasVulkan) 0.05f else 0f

        // 内存加成
        val ramBonus = when {
            deviceInfo.totalRAMGB >= 12 -> 0.05f
            deviceInfo.totalRAMGB >= 8 -> 0.03f
            deviceInfo.totalRAMGB >= 6 -> 0.02f
            else -> 0f
        }

        return (baseScore + npuBonus + vulkanBonus + ramBonus).coerceIn(0f, 1f)
    }

    private fun computeNetworkScore(rttMs: Float): Float {
        // RTT 越低分越高，>500ms 视为弱网
        return when {
            rttMs < 20f -> 1.0f
            rttMs < 50f -> 0.9f
            rttMs < 100f -> 0.75f
            rttMs < 200f -> 0.55f
            rttMs < 500f -> 0.35f
            else -> 0.1f
        }
    }

    private fun computeBatteryTempScore(): Float {
        return try {
            val bm = context.getSystemService(Context.BATTERY_SERVICE) as? BatteryManager
            val tempC = bm?.getIntProperty(BatteryManager.BATTERY_PROPERTY_TEMPERATURE)?.div(10f) ?: 30f
            when {
                tempC < 35f -> 1.0f       // 凉爽
                tempC < 40f -> 0.8f       // 温热
                tempC < 45f -> 0.5f       // 偏热，可能降频
                tempC < 50f -> 0.2f       // 很热
                else -> 0.05f              // 危险温度
            }
        } catch (_: Exception) { 0.7f }
    }

    private fun computeBatteryLevelScore(): Float {
        return try {
            val bm = context.getSystemService(Context.BATTERY_SERVICE) as? BatteryManager
            // 优先用 BATTERY_PROPERTY_CAPACITY（Android 8.0+）
            val capacity = bm?.getIntProperty(BatteryManager.BATTERY_PROPERTY_CAPACITY)
            val level = if (capacity != null && capacity > 0) {
                capacity
            } else {
                // Fallback: 通过 Intent 获取电量百分比
                val intent = context.registerReceiver(null, android.content.IntentFilter(android.content.Intent.ACTION_BATTERY_CHANGED))
                val levelRaw = intent?.getIntExtra(android.os.BatteryManager.EXTRA_LEVEL, 80) ?: 80
                val scale = intent?.getIntExtra(android.os.BatteryManager.EXTRA_SCALE, 100) ?: 100
                (levelRaw * 100 / scale.coerceAtLeast(1)).coerceIn(0, 100)
            }
            when {
                level > 50 -> 1.0f
                level > 30 -> 0.8f
                level > 15 -> 0.5f          // 触发节能
                else -> 0.2f                // 极度省电模式
            }
        } catch (_: Exception) { 0.7f }
    }

    private fun computeLoadScore(): Float {
        // 基于可用内存和 CPU 核心使用率估算
        return try {
            val activityManager = context.getSystemService(Context.ACTIVITY_SERVICE) as android.app.ActivityManager
            val mi = android.app.ActivityManager.MemoryInfo()
            activityManager.getMemoryInfo(mi)
            val availMemGB = mi.availMem / (1024.0 * 1024.0 * 1024.0)
            val totalMemGB = mi.totalMem / (1024.0 * 1024.0 * 1024.0)
            val memRatio = if (totalMemGB > 0) (availMemGB / totalMemGB).toFloat() else 0.5f

            // 内存充足 + 核心数多 = 低负载潜力
            val coreBonus = min(Runtime.getRuntime().availableProcessors() / 8f, 1f) * 0.2f
            (memRatio * 0.8f + coreBonus).coerceIn(0f, 1f)
        } catch (_: Exception) { 0.5f }
    }

    /**
     * 获取详细评分报告（用于日志/调试）
     */
    fun getDetailedReport(): String {
        val score = compute()
        val path = decidePath(score)
        return """
            ════════════════════════════════════════
              DevicePowerScore Report
            ════════════════════════════════════════
              Score:    ${String.format("%.3f", score)}
              Path:     $path
              Identity: ${getEngineIdentity()}
            ════════════════════════════════════════
        """.trimIndent()
    }
}

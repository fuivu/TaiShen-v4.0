package com.localaipainter.xunji

/**
 * ═══════════════════════════════════════════════════════════════
 *  迅集 (XunJi) — 量化调度辅助架构  v4.0 TaiShen
 *  全局配置中心 · 7 级量化枚举 · 内存快照 · 降级策略
 * ═══════════════════════════════════════════════════════════════
 */

/** 量化精度等级 — 从最高精度到极限压缩 */
enum class QuantLevel(val bits: Int, val label: String, val memRatio: Float) {
    FP32(32, "FP32 全精度", 1.0f),
    FP16(16, "FP16 半精度", 0.5f),
    BF16(16, "BF16 脑半精度", 0.5f),
    INT8( 8, "INT8 量化",    0.25f),
    INT4( 4, "INT4 量化",    0.125f),
    INT2( 2, "INT2 极限",    0.0625f),
    FP8 ( 8, "FP8 (E4M3)",  0.25f);

    /** 是否为整数量化 */
    fun isInteger(): Boolean = when(this){ INT8->true; INT4->true; INT2->true; else->false }
    /** 是否为浮点格式 */
    fun isFloat(): Boolean = when(this){ FP32->true; FP16->true; BF16->true; FP8->true; else->false }
    /** 字节数 */
    fun bytesPerElem(): Int = bits / 8
}

/** 内存压力等级 */
enum class MemPressure { LOW, MEDIUM, HIGH, CRITICAL }

/** 内存快照 — 单次采集结果 */
data class MemSnapshot(
    val totalKb: Long,
    val freeKb: Long,
    val availableKb: Long,
    val buffersKb: Long,
    val cachedKb: Long,
    val gpuTotalMb: Int = 0,
    val gpuAvailableMb: Int = 0,
    val gpuQueryMethod: String = "unknown",
    val timestampMs: Long = System.currentTimeMillis()
) {
    val usedKb: Long get() = totalKb - availableKb
    val usedRatio: Float get() = if (totalKb > 0) usedKb.toFloat() / totalKb else 0f
    val pressure: MemPressure
        get() = when {
            availableKb < totalKb * 0.05f -> MemPressure.CRITICAL
            availableKb < totalKb * 0.10f -> MemPressure.HIGH
            availableKb < totalKb * 0.20f -> MemPressure.MEDIUM
            else -> MemPressure.LOW
        }
    fun safeBudgetKb(reserveMb: Int = 200): Long = maxOf(0L, availableKb - reserveMb * 1024L)
}

/** 用户偏好 — 手动指定量化等级 */
data class XunJiPrefs(
    val manualLevel: QuantLevel? = null,   // null = 自动
    val allowFallback: Boolean = true,
    val minLevel: QuantLevel = QuantLevel.INT2,
    val maxLevel: QuantLevel = QuantLevel.FP32,
    val oomLock: Boolean = false,
    val oomLockLevel: QuantLevel = QuantLevel.INT2
)

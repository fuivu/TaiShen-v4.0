package com.localaipainter.hunyuan

/**
 * ═════════════════════════════════════════════════════════════
 *  混元 (HunYuan) — 配置  v4.0 TaiShen
 *  10 种算法枚举 · 6 种策略 · 性能快照 · 自定义配置
 * ═════════════════════════════════════════════════════════════
 */

/** 10 种推理算法 */
enum class AlgoType(val id: Int, val label: String, val speedTier: Int) {
    FP32_FULL(0, "FP32 全精度", 1),
    FP16_HALF(1, "FP16 半精度", 3),
    INT8_QUANT(2, "INT8 量化", 4),
    SPARSE_ATTN(3, "稀疏注意力", 4),
    SPECULATIVE(4, "投机采样", 5),
    FUSED_OPS(5, "算子融合", 4),
    MIXED_PRECISION(6, "分层混合精度", 3),
    INT4_EXTREME(7, "INT4 极限量化", 5),
    NPU_HARDWARE(8, "NPU 硬件加速", 5),
    CASCADE_SR(9, "联级超分", 2);

    companion object {
        fun fromId(id: Int) = values().firstOrNull { it.id == id } ?: FP16_HALF
        fun count() = values().size
    }
}

/** 6 种调度策略 */
enum class ScheduleStrategy(val id: Int, val label: String) {
    MAX_SPEED(0, "极速优先"),
    MAX_QUALITY(1, "画质优先"),
    POWER_SAVE(2, "省电模式"),
    BALANCED(3, "均衡模式"),
    ADAPTIVE(4, "自适应"),
    CUSTOM(5, "自定义");

    companion object {
        fun fromId(id: Int) = values().firstOrNull { it.id == id } ?: ADAPTIVE
    }
}

/** 性能快照 — 每次采集 */
data class PerfSnapshot(
    val cpuUsage: Float = 0f,       // 0..1
    val cpuTempC: Float = 50f,
    val memUsedRatio: Float = 0.5f,  // 0..1
    val batteryPct: Float = 80f,     // 0..100
    val batteryTempC: Float = 35f,
    val charging: Boolean = false,
    val gpuUsage: Float = 0.3f,      // 0..1
    val availMemKb: Long = 4_000_000L,
    val timestampMs: Long = System.currentTimeMillis()
)

/** 设备能力 */
data class DeviceCap(
    val cpuCores: Int = Runtime.getRuntime().availableProcessors(),
    val cpuFreqGhz: Float = 2.0f,
    val ramGb: Int = 8,
    val hasNpu: Boolean = false,
    val hasGpu: Boolean = true,
    val hasVulkan: Boolean = true,
    val socTier: Int = 2 // 1=low 2=mid 3=high 4=flagship
)

/** 自定义配置 */
data class CustomAlgoConfig(
    val name: String,
    val algoOrder: List<AlgoType>,  // 用户自选优先级
    val minBatteryPct: Int = 15,
    val maxTempC: Int = 75,
    val preferNpu: Boolean = false
) {
    fun toJson(): String {
        val arr = algoOrder.joinToString(",") { "\"${it.name}\"" }
        return """{"name":"$name","algos":[$arr],"min_battery":$minBatteryPct,"max_temp":$maxTempC,"prefer_npu":$preferNpu}"""
    }

    companion object {
        fun default() = CustomAlgoConfig(
            name = "我的配置",
            algoOrder = listOf(
                AlgoType.NPU_HARDWARE, AlgoType.FP16_HALF, AlgoType.SPECULATIVE,
                AlgoType.FUSED_OPS, AlgoType.SPARSE_ATTN, AlgoType.INT8_QUANT,
                AlgoType.MIXED_PRECISION, AlgoType.INT4_EXTREME,
                AlgoType.CASCADE_SR, AlgoType.FP32_FULL
            )
        )
    }
}

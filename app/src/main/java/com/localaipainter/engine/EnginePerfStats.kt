package com.localaipainter.engine

/**
 * 引擎性能统计快照
 */
data class EnginePerfStats(
    val totalInferences: Long,
    val totalTimeMs: Long,
    val avgTimeMs: Double,
    val peakMemoryMB: Long,
    val cacheHitRate: Float
)

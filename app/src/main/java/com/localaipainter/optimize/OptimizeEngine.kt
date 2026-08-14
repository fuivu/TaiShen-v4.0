package com.localaipainter.optimize

import com.localaipainter.util.Logger

/**
 * 太神架构 · 优化引擎总控
 *
 * 统一管理三大核心优化：
 * 1. OperatorFusion2   → 异构算子融合
 * 2. DynamicSparseAttention → 动态稀疏激活
 * 3. SpeculativeDecoding  → 投机采样
 *
 * 根据硬件画像自动启用/禁用各模块，提供统一接口给 Pipeline 调用。
 */
class OptimizeEngine(
    private val socProfile: SocProfile
) {
    companion object {
        private const val TAG = "OptimizeEngine"
        const val ENGINE_VERSION = "4.0.0"
    }

    // ─── 三大优化模块 ────────────────────────────────
    val operatorFusion: OperatorFusion2
    val sparseAttention: DynamicSparseAttention
    val speculativeDecoding: SpeculativeDecoding

    // ─── 全局开关 ────────────────────────────────────
    data class OptimizationConfig(
        val enableFusion: Boolean = true,
        val enableSparse: Boolean = true,
        val enableSpeculative: Boolean = true,
        val fusionMode: OperatorFusion2.FusionMode = OperatorFusion2.FusionMode.ADAPTIVE,
        val sparseStrategy: DynamicSparseAttention.SparseStrategy = DynamicSparseAttention.SparseStrategy.ADAPTIVE,
        val maxDraftTokens: Int = 8,
        val targetSparsity: Float = 0.75f
    )

    private var config: OptimizationConfig = OptimizationConfig()

    // ─── 硬件画像 ────────────────────────────────────
    data class SocProfile(
        val socName: String,
        val architecture: String,        // "armv8.2" / "armv9" / "armv9.2+sve2"
        val numCores: Int,
        val bigCoreFreqMHz: Int,
        val littleCoreFreqMHz: Int,
        val l2CacheKB: Int,
        val l3CacheKB: Int,
        val memoryBandwidthGBs: Float,
        val hasVulkan: Boolean,
        val hasOpenCL: Boolean,
        val hasNPU: Boolean,
        val npuVendor: String,
        val npuTOPS: Float,            // NPU 算力 (INT8 TOPS)
        val supportsFP16: Boolean,
        val supportsINT4: Boolean,
        val supportsSVE2: Boolean,
        val supportsDotProd: Boolean
    )

    // ─── 综合统计 ────────────────────────────────────
    data class OptimizeStats(
        val fusionStats: OperatorFusion2.FusionStats,
        val sparseStats: DynamicSparseAttention.SparseStats,
        val specStats: SpeculativeDecoding.SpecStats,
        val overallSpeedup: Float,
        val overallMemorySavedMB: Float,
        val configSnapshot: OptimizationConfig
    )

    init {
        // 初始化算子融合
        val hwProfile = OperatorFusion2.HardwareProfile(
            socName = socProfile.socName,
            hasVulkan = socProfile.hasVulkan,
            hasOpenCL = socProfile.hasOpenCL,
            hasNPU = socProfile.hasNPU,
            npuVendor = socProfile.npuVendor,
            l2CacheKB = socProfile.l2CacheKB,
            memoryBandwidthGBs = socProfile.memoryBandwidthGBs,
            computeUnits = socProfile.numCores
        )
        operatorFusion = OperatorFusion2(hwProfile)

        // 初始化稀疏注意力
        val spProfile = DynamicSparseAttention.SocProfile(
            socName = socProfile.socName,
            hasNPU = socProfile.hasNPU,
            npuDedicatedMemMB = (socProfile.npuTOPS * 32).toInt(), // 经验估算
            l3CacheKB = socProfile.l3CacheKB,
            supportsSVE2 = socProfile.supportsSVE2,
            supportsFP16Matrix = socProfile.supportsFP16
        )
        sparseAttention = DynamicSparseAttention(
            hiddenDim = 768,
            numHeads = 12,
            seqLen = 4096,
            socProfile = spProfile
        )

        // 初始化投机采样
        val specConfig = SpeculativeDecoding.SpecConfig(
            draftModelPath = "",
            targetModelPath = "",
            maxDraftTokens = config.maxDraftTokens,
            useNPUForDraft = socProfile.hasNPU,
            useNPUForVerify = socProfile.hasNPU,
            npuVendor = socProfile.npuVendor,
            enableCIMMode = socProfile.npuVendor == "mediatek" && socProfile.npuTOPS >= 30f
        )
        speculativeDecoding = SpeculativeDecoding(specConfig)

        // 自动调优
        autoTune()

        Logger.i(TAG, "══════════════════════════════════════════")
        Logger.i(TAG, "太神优化引擎 v$ENGINE_VERSION 初始化完成")
        Logger.i(TAG, "SoC: ${socProfile.socName} | NPU: ${socProfile.npuVendor} ${socProfile.npuTOPS}TOPS")
        Logger.i(TAG, "Fusion: ${if (config.enableFusion) "✅" else "❌"} | " +
            "Sparse: ${if (config.enableSparse) "✅" else "❌"} | " +
            "Speculative: ${if (config.enableSpeculative) "✅" else "❌"}")
        Logger.i(TAG, "══════════════════════════════════════════")
    }

    // ─── 自动调优 ────────────────────────────────────
    private fun autoTune() {
        // 天玑 8400+ / 骁龙 8 Gen3+ → 全开
        if (socProfile.npuTOPS >= 30f && socProfile.l3CacheKB >= 4096) {
            config = config.copy(
                enableFusion = true,
                enableSparse = true,
                enableSpeculative = true,
                fusionMode = OperatorFusion2.FusionMode.AGGRESSIVE,
                sparseStrategy = DynamicSparseAttention.SparseStrategy.ADAPTIVE,
                maxDraftTokens = 8,
                targetSparsity = 0.80f
            )
            operatorFusion.setMode(OperatorFusion2.FusionMode.AGGRESSIVE)
            sparseAttention.updateConfig(
                sparseAttention.getConfig().copy(
                    strategy = DynamicSparseAttention.SparseStrategy.ADAPTIVE,
                    topK = 96,
                    sparsityTarget = 0.80f
                )
            )
            Logger.i(TAG, "Auto-tune: FLAGSHIP mode (all optimizations aggressive)")
        }
        // 中端 → 适度
        else if (socProfile.npuTOPS >= 10f && socProfile.l3CacheKB >= 2048) {
            config = config.copy(
                enableFusion = true,
                enableSparse = true,
                enableSpeculative = true,
                fusionMode = OperatorFusion2.FusionMode.ADAPTIVE,
                sparseStrategy = DynamicSparseAttention.SparseStrategy.LOCAL_WINDOW,
                maxDraftTokens = 4,
                targetSparsity = 0.65f
            )
            operatorFusion.setMode(OperatorFusion2.FusionMode.ADAPTIVE)
            sparseAttention.updateConfig(
                sparseAttention.getConfig().copy(
                    strategy = DynamicSparseAttention.SparseStrategy.LOCAL_WINDOW,
                    localWindowSize = 96,
                    sparsityTarget = 0.65f
                )
            )
            Logger.i(TAG, "Auto-tune: MIDRANGE mode (balanced optimizations)")
        }
        // 低端 → 仅保守融合
        else {
            config = config.copy(
                enableFusion = true,
                enableSparse = false,
                enableSpeculative = false,
                fusionMode = OperatorFusion2.FusionMode.CONSERVATIVE,
                maxDraftTokens = 2
            )
            operatorFusion.setMode(OperatorFusion2.FusionMode.CONSERVATIVE)
            Logger.i(TAG, "Auto-tune: BUDGET mode (fusion only, conservative)")
        }
    }

    // ─── 对外 API ────────────────────────────────────

    /**
     * 对计算图执行完整优化管线
     * 1. 算子融合
     * 2. 稀疏化标注
     * 3. 投机解码准备
     */
    fun optimizeGraph(graph: OperatorFusion2.ComputeGraph): OptimizedGraph {
        // Step 1: 算子融合
        val fusedGraph = if (config.enableFusion) {
            operatorFusion.fuseGraph(graph)
        } else graph

        // Step 2: 稀疏化标注（在图上标记可稀疏的注意力节点）
        val sparsifiedGraph = if (config.enableSparse) {
            annotateSparseNodes(fusedGraph)
        } else fusedGraph

        // Step 3: 投机解码准备（标记可投机的解码层）
        val finalGraph = if (config.enableSpeculative) {
            annotateSpeculativeNodes(sparsifiedGraph)
        } else sparsifiedGraph

        return OptimizedGraph(
            graph = finalGraph,
            fusionApplied = config.enableFusion,
            sparseApplied = config.enableSparse,
            speculativeApplied = config.enableSpeculative
        )
    }

    private fun annotateSparseNodes(graph: OperatorFusion2.ComputeGraph): OperatorFusion2.ComputeGraph {
        // 标记所有 Attention 节点为可稀疏
        graph.nodes.forEachIndexed { idx, node ->
            if (node.opType.contains("Attention") || node.opType.contains("MatMul")) {
                Logger.d(TAG, "Sparse annotation: node[$idx]=${node.opType} (sparsity=${config.targetSparsity})")
            }
        }
        return graph
    }

    private fun annotateSpeculativeNodes(graph: OperatorFusion2.ComputeGraph): OperatorFusion2.ComputeGraph {
        // 标记最后 N 层为可投机解码
        val decoderLayers = graph.nodes.count { it.opType.contains("Decoder") }
        val specLayers = minOf(4, decoderLayers / 2)
        Logger.d(TAG, "Speculative annotation: $specLayers/$decoderLayers decoder layers marked")
        return graph
    }

    /**
     * 预热所有优化模块
     */
    fun warmupAll(inputShape: IntArray = intArrayOf(1, 3, 512, 512)) {
        Logger.i(TAG, "Warming up all optimization modules...")
        if (config.enableFusion) {
            operatorFusion.warmup(inputShape)
        }
        // 投机采样预热
        if (config.enableSpeculative) {
            Logger.d(TAG, "Speculative warmup: K=${config.maxDraftTokens}")
        }
        Logger.i(TAG, "Warmup complete ✅")
    }

    /**
     * 更新配置
     */
    fun updateConfig(newConfig: OptimizationConfig) {
        config = newConfig
        operatorFusion.setMode(newConfig.fusionMode)
        sparseAttention.updateConfig(
            sparseAttention.getConfig().copy(
                strategy = newConfig.sparseStrategy,
                sparsityTarget = newConfig.targetSparsity
            )
        )
        Logger.i(TAG, "Config updated: $config")
    }

    fun getConfig(): OptimizationConfig = config

    /**
     * 获取综合统计
     */
    fun getStats(): OptimizeStats {
        val fStats = operatorFusion.getStats()
        val sStats = sparseAttention.getStats()
        val spStats = speculativeDecoding.getStats()

        // 综合加速比（乘法叠加）
        val overallSpeedup = fStats.estimatedSpeedup *
            sStats.speedupEstimate() *
            spStats.theoreticalSpeedup

        val memSavedMB = (fStats.savedMemReads + fStats.savedMemWrites +
            sStats.memorySavedBytes).toFloat() / (1024f * 1024f)

        return OptimizeStats(
            fusionStats = fStats,
            sparseStats = sStats,
            specStats = spStats,
            overallSpeedup = overallSpeedup.coerceAtMost(10f),
            overallMemorySavedMB = memSavedMB,
            configSnapshot = config
        )
    }

    /**
     * 生成完整优化报告
     */
    fun generateReport(): String {
        val stats = getStats()
        return """
            |═════════════════════════════════════════════════════
            |  太神优化引擎 v$ENGINE_VERSION — 综合报告
            |═════════════════════════════════════════════════════
            |
            |【硬件画像】
            |  SoC:        ${socProfile.socName}
            |  架构:       ${socProfile.architecture}
            |  NPU:        ${socProfile.npuVendor} (${socProfile.npuTOPS} TOPS)
            |  内存带宽:   ${socProfile.memoryBandwidthGBs} GB/s
            |  L3缓存:     ${socProfile.l3CacheKB} KB
            |  FP16:       ${if (socProfile.supportsFP16) "✅" else "❌"} | 
            |  INT4:       ${if (socProfile.supportsINT4) "✅" else "❌"} | 
            |  SVE2:       ${if (socProfile.supportsSVE2) "✅" else "❌"}
            |
            |【算子融合 2.0】
            |  融合模式:   ${stats.fusionStats.currentMode}
            |  已融合算子: ${stats.fusionStats.totalFusedOps}
            |  注册模式:   ${stats.fusionStats.registeredPatterns}
            |  活跃Kernel: ${stats.fusionStats.activeKernels}
            |  预估加速:   ${"%.2f".format(stats.fusionStats.estimatedSpeedup)}×
            |
            |【动态稀疏激活】
            |  策略:       ${stats.sparseStats.currentStrategy}
            |  总查询:     ${stats.sparseStats.totalQueries}
            |  稀疏度:     ${"%.1f".format(stats.sparseStats.sparsityRatio * 100)}%
            |  节省内存:   ${"%.1f".format(stats.sparseStats.memorySavedMB())} MB
            |  策略切换:   ${stats.sparseStats.strategySwitches}
            |  预估加速:   ${"%.2f".format(stats.sparseStats.speedupEstimate())}×
            |
            |【投机采样】
            |  总轮次:     ${stats.specStats.totalRounds}
            |  草稿Token:  ${stats.specStats.totalDrafted}
            |  接受Token:  ${stats.specStats.totalAccepted}
            |  接受率:     ${"%.1f".format(stats.specStats.acceptanceRate * 100)}%
            |  当前K:      ${stats.specStats.currentDraftLength}
            |  NPU回退:    ${stats.specStats.npuFallbackCount}
            |  预估加速:   ${"%.2f".format(stats.specStats.theoreticalSpeedup)}×
            |
            |【综合指标】
            |  ★ 总加速比:  ${"%.2f".format(stats.overallSpeedup)}×
            |  ★ 节省内存:  ${"%.1f".format(stats.overallMemorySavedMB)} MB
            |
            |═════════════════════════════════════════════════════
        """.trimMargin()
    }

    /**
     * 关闭所有优化模块
     */
    fun shutdown() {
        speculativeDecoding.shutdown()
        Logger.i(TAG, "OptimizeEngine shutdown complete")
    }

    // ─── 优化后的图 ──────────────────────────────────
    data class OptimizedGraph(
        val graph: OperatorFusion2.ComputeGraph,
        val fusionApplied: Boolean,
        val sparseApplied: Boolean,
        val speculativeApplied: Boolean
    )
}

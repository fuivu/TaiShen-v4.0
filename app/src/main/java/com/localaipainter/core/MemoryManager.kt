package com.localaipainter.core

import android.util.Log

/**
 * 4 级内存管理策略
 */
class MemoryManager(private val totalRamMB: Long) {

    enum class Strategy {
        BALANCED,       // 均衡模式 - 全精度 + 缓存
        CONSERVATIVE,   // 保守模式 - INT8 + 限制并发
        AGGRESSIVE,     // 极限压缩 - INT4 + 最小缓存
        MINIMAL         // 最低配置 - CPU only + 超低分辨率
    }

    private var currentStrategy: Strategy = Strategy.BALANCED
    private val tensorCache = LruTensorCache(maxSizeMB = (totalRamMB / 8).toInt())
    private val allocatedTensors = mutableMapOf<String, Long>()

    init {
        // 根据 RAM 自动选择初始策略
        currentStrategy = when {
            totalRamMB >= 8192 -> Strategy.BALANCED
            totalRamMB >= 4096 -> Strategy.CONSERVATIVE
            totalRamMB >= 2048 -> Strategy.AGGRESSIVE
            else -> Strategy.MINIMAL
        }
        Log.i(TAG, "Memory strategy: $currentStrategy (RAM: ${totalRamMB}MB)")
    }

    fun getStrategy(): Strategy = currentStrategy

    fun setStrategy(strategy: Strategy) {
        currentStrategy = strategy
        adjustCacheSize()
    }

    fun getMaxImageSize(): Pair<Int, Int> {
        return when (currentStrategy) {
            Strategy.BALANCED -> Pair(768, 768)
            Strategy.CONSERVATIVE -> Pair(512, 512)
            Strategy.AGGRESSIVE -> Pair(384, 384)
            Strategy.MINIMAL -> Pair(256, 256)
        }
    }

    fun getRecommendedQuantization(): QuantizationType {
        return when (currentStrategy) {
            Strategy.BALANCED -> QuantizationType.NONE
            Strategy.CONSERVATIVE -> QuantizationType.INT8
            Strategy.AGGRESSIVE -> QuantizationType.INT4
            Strategy.MINIMAL -> QuantizationType.INT4
        }
    }

    fun getMaxSteps(): Int {
        return when (currentStrategy) {
            Strategy.BALANCED -> 30
            Strategy.CONSERVATIVE -> 20
            Strategy.AGGRESSIVE -> 15
            Strategy.MINIMAL -> 10
        }
    }

    fun allocateTensor(name: String, sizeMB: Long): Boolean {
        val available = getAvailableRamMB()
        if (available < sizeMB + 200) { // 保留 200MB 系统余量
            // 尝试降级
            if (currentStrategy != Strategy.MINIMAL) {
                val nextLevel = Strategy.values()[currentStrategy.ordinal + 1]
                Log.w(TAG, "Memory low (${available}MB), downgrading to $nextLevel")
                setStrategy(nextLevel)
                return false // 需要重试
            }
            return false
        }
        allocatedTensors[name] = sizeMB
        return true
    }

    fun freeTensor(name: String) {
        allocatedTensors.remove(name)
    }

    fun getCache(): LruTensorCache = tensorCache

    fun getMemoryReport(): String {
        val used = allocatedTensors.values.sum()
        return "Strategy: $currentStrategy | Used: ${used}MB | Available: ${getAvailableRamMB()}MB | Cached: ${tensorCache.sizeMB()}MB"
    }

    private fun adjustCacheSize() {
        val cacheMB = when (currentStrategy) {
            Strategy.BALANCED -> (totalRamMB / 6).toInt()
            Strategy.CONSERVATIVE -> (totalRamMB / 10).toInt()
            Strategy.AGGRESSIVE -> (totalRamMB / 16).toInt()
            Strategy.MINIMAL -> (totalRamMB / 32).toInt()
        }
        tensorCache.resize(cacheMB)
    }

    private fun getAvailableRamMB(): Long {
        val runtime = Runtime.getRuntime()
        val usedMB = (runtime.totalMemory() - runtime.freeMemory()) / (1024 * 1024)
        return totalRamMB - usedMB
    }

    companion object {
        private const val TAG = "MemoryManager"
    }
}

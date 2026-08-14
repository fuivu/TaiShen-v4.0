package com.localaipainter.memory

import java.util.LinkedHashMap

/**
 * LRU 张量缓存 —— 限制最大显存/内存占用，超出时淘汰最久未使用的张量
 */
class LruTensorCache(private val maxSizeMB: Int) {
    private val cache = object : LinkedHashMap<String, FloatArray>(16, 0.75f, true) {
        override fun removeEldestEntry(eldest: MutableMap.MutableEntry<String, FloatArray>): Boolean {
            return currentSizeBytes() > maxSizeMB * 1024 * 1024
        }
    }

    @Synchronized
    fun put(key: String, tensor: FloatArray) {
        cache[key] = tensor
    }

    @Synchronized
    fun get(key: String): FloatArray? = cache[key]

    @Synchronized
    fun remove(key: String) {
        cache.remove(key)
    }

    @Synchronized
    fun clear() {
        cache.clear()
    }

    @Synchronized
    fun currentSizeMB(): Int = (currentSizeBytes() / (1024 * 1024)).toInt()

    private fun currentSizeBytes(): Long {
        var total = 0L
        for (arr in cache.values) {
            total += arr.size * 4L
        }
        return total
    }
}

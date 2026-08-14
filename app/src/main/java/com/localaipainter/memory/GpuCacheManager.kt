package com.localaipainter.memory

import android.content.Context
import android.content.SharedPreferences
import com.localaipainter.util.Logger

class GpuCacheManager(context: Context) {
    private val prefs: SharedPreferences = context.getSharedPreferences("gpu_cache", Context.MODE_PRIVATE)

    var enabled: Boolean
        get() = prefs.getBoolean("enabled", true)
        set(v) {
            prefs.edit().putBoolean("enabled", v).apply()
            nativeSetEnabled(v)
            Logger.i("GpuCache", "enabled=$v")
        }

    var cacheSizeMB: Int
        get() = prefs.getInt("cache_size_mb", 512)
        set(v) {
            prefs.edit().putInt("cache_size_mb", v).apply()
            nativeSetSize(v)
            Logger.i("GpuCache", "size=${v}MB")
        }

    val hitRate: Float get() = nativeGetHitRate()
    val usedMB: Int get() = nativeGetUsed()
    val totalMB: Int get() = nativeGetTotal()

    init {
        if (enabled) {
            nativeInit(cacheSizeMB)
            Logger.i("GpuCache", "init ${cacheSizeMB}MB")
        }
    }

    fun clear() { nativeClear(); Logger.i("GpuCache", "cleared") }
    fun release() { nativeRelease() }

    private external fun nativeInit(sizeMB: Int)
    private external fun nativeSetEnabled(e: Boolean)
    private external fun nativeSetSize(sizeMB: Int)
    private external fun nativeGetHitRate(): Float
    private external fun nativeGetUsed(): Int
    private external fun nativeGetTotal(): Int
    private external fun nativeClear()
    private external fun nativeRelease()
}

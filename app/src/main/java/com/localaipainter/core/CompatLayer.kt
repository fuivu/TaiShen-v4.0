package com.localaipainter.core

import android.content.Context
import android.os.Build
import android.os.VibrationEffect
import android.os.Vibrator
import android.os.VibratorManager
import androidx.core.content.ContextCompat

/**
 * 🛡️ 兼容性抽象层 —— 屏蔽 Android 版本差异
 *
 * 所有"不同版本 API 行为不同"的逻辑集中在这里，
 * 业务代码只调用本层接口，不写版本判断。
 *
 * 适配范围：Android 8.0 (API 26) ~ Android 15 (API 35)
 */
object CompatLayer {

    // ============ 振动（API 26 前后差异） ============

    fun vibrate(context: Context, ms: Long) {
        try {
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.S) {
                val vm = context.getSystemService(VibratorManager::class.java)
                vm?.vibrate(VibrationEffect.createOneShot(ms, VibrationEffect.DEFAULT_AMPLITUDE))
            } else {
                @Suppress("DEPRECATION")
                val v = context.getSystemService(Context.VIBRATOR_SERVICE) as? Vibrator
                v?.vibrate(VibrationEffect.createOneShot(ms, VibrationEffect.DEFAULT_AMPLITUDE))
            }
        } catch (t: Throwable) { /* 忽略振动失败 */ }
    }

    // ============ 通知渠道（API 26+） ============

    fun ensureNotificationChannel(
        context: Context,
        channelId: String,
        name: String,
        importance: Int = android.app.NotificationManager.IMPORTANCE_LOW
    ) {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
            val nm = context.getSystemService(android.app.NotificationManager::class.java)
            if (nm?.getNotificationChannel(channelId) == null) {
                val ch = android.app.NotificationChannel(channelId, name, importance)
                ch.description = name
                ch.setShowBadge(false)
                nm?.createNotificationChannel(ch)
            }
        }
    }

    // ============ 存储权限（API 33+ 细分） ============

    fun getStoragePermissions(): Array<String> {
        return if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU) {
            arrayOf(android.Manifest.permission.READ_MEDIA_IMAGES)
        } else if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
            arrayOf() // Scoped Storage，无需权限
        } else {
            arrayOf(
                android.Manifest.permission.READ_EXTERNAL_STORAGE,
                android.Manifest.permission.WRITE_EXTERNAL_STORAGE
            )
        }
    }

    // ============ 前台服务类型（API 34+ 强制） ============

    fun getForegroundServiceType(): Int {
        return if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.UPSIDE_DOWN_CAKE) {
            android.content.pm.ServiceInfo.FOREGROUND_SERVICE_TYPE_DATA_SYNC or
            android.content.pm.ServiceInfo.FOREGROUND_SERVICE_TYPE_MEDIA_PROCESSING
        } else {
            0
        }
    }

    // ============ 动态取色（API 31+） ============

    fun supportsDynamicColor(): Boolean =
        Build.VERSION.SDK_INT >= Build.VERSION_CODES.S

    // ============ 共享元素过渡（API 21+） ============

    fun supportsSharedElementTransition(): Boolean =
        Build.VERSION.SDK_INT >= Build.VERSION_CODES.LOLLIPOP

    // ============ 大屏/折叠屏适配 ============

    fun isLargeScreen(context: Context): Boolean {
        val metrics = context.resources.displayMetrics
        val widthDp = metrics.widthPixels / metrics.density
        return widthDp >= 600
    }

    fun isTablet(context: Context): Boolean {
        val metrics = context.resources.displayMetrics
        val widthDp = metrics.widthPixels / metrics.density
        return widthDp >= 840
    }

    // ============ 内存信息（API 级别差异） ============

    fun getMemoryInfo(context: Context): CompatMemoryInfo {
        val am = context.getSystemService(Context.ACTIVITY_SERVICE) as android.app.ActivityManager
        val mi = android.app.ActivityManager.MemoryInfo()
        am.getMemoryInfo(mi)
        return CompatMemoryInfo(
            totalMemMB = mi.totalMem / (1024 * 1024),
            availMemMB = mi.availMem / (1024 * 1024),
            lowMemory = mi.lowMemory,
            thresholdMB = mi.threshold / (1024 * 1024),
            isLowRamDevice = am.isLowRamDevice
        )
    }

    // ============ 硬件加速 ============

    fun setHardwareAccelerated(flags: Int): Int {
        var f = flags or android.view.WindowManager.LayoutParams.FLAG_HARDWARE_ACCELERATED
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
            // API 26+ 支持更细粒度的硬件加速控制
            f = f or 0x00001000 // FLAG_DRAWS_SYSTEM_BAR_BACKGROUNDS 已存在
        }
        return f
    }
}

data class CompatMemoryInfo(
    val totalMemMB: Long,
    val availMemMB: Long,
    val lowMemory: Boolean,
    val thresholdMB: Long,
    val isLowRamDevice: Boolean
) {
    val usedPercent: Float get() = ((totalMemMB - availMemMB).toFloat() / totalMemMB).coerceIn(0f, 1f)
}

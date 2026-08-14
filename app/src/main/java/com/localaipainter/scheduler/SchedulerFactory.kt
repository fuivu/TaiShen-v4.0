package com.localaipainter.scheduler

import com.localaipainter.core.SchedulerCategory
import com.localaipainter.core.SchedulerProvider
import com.localaipainter.pipeline.Scheduler
import com.localaipainter.pipeline.SchedulerRegistry

/**
 * SchedulerFactory —— 太神架构 v4.0 调度器工厂
 *
 * 统一入口，封装 SchedulerRegistry 的创建逻辑。
 * 调用方只需 import com.localaipainter.scheduler.SchedulerFactory，
 * 无需关心底层注册细节。
 *
 * 使用：
 *   val scheduler = SchedulerFactory.create("LCM", steps=6, cfg=1.0f, w=512, h=512)
 */
object SchedulerFactory {

    // ============ 创建 ============

    /**
     * 按类型名创建调度器
     */
    fun create(type: String, steps: Int, cfg: Float, w: Int, h: Int): Scheduler {
        return SchedulerRegistry.create(type, steps, cfg, w, h)
    }

    /**
     * 按 SchedulerProvider 创建
     */
    fun create(provider: SchedulerProvider, steps: Int, cfg: Float, w: Int, h: Int): Scheduler {
        return provider.create(steps, cfg, w, h)
    }

    // ============ 查询 ============

    fun getMeta(type: String): SchedulerProvider =
        SchedulerRegistry.getMeta(type)

    fun getAll(): List<SchedulerProvider> =
        SchedulerRegistry.ALL

    fun getFast(): List<SchedulerProvider> =
        SchedulerRegistry.getFast()

    fun getBalanced(): List<SchedulerProvider> =
        SchedulerRegistry.getBalanced()

    fun getQuality(): List<SchedulerProvider> =
        SchedulerRegistry.getQuality()

    fun getByCategory(cat: SchedulerCategory): List<SchedulerProvider> =
        SchedulerRegistry.getByCategory(cat)

    // ============ 便捷方法 ============

    /**
     * 获取推荐的默认调度器（LCM 极速）
     */
    fun createDefault(steps: Int = 6, cfg: Float = 1.0f, w: Int = 512, h: Int = 512): Scheduler {
        return create("LCM", steps, cfg, w, h)
    }

    /**
     * 根据设备能力推荐调度器
     * - 高算力 → DDIM (高质量)
     * - 中算力 → DPM++2M (均衡)
     * - 低算力 → LCM (极速)
     */
    fun createRecommended(powerScore: Float, steps: Int, cfg: Float, w: Int, h: Int): Scheduler {
        val type = when {
            powerScore >= 0.85f -> "DDIM"
            powerScore >= 0.5f  -> "DPM++2M"
            else                  -> "LCM"
        }
        return create(type, steps, cfg, w, h)
    }
}

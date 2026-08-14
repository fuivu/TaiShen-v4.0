package com.localaipainter.core

import com.localaipainter.pipeline.Scheduler

/**
 * 调度器提供者 —— 插件式注册
 *
 * 新增调度器的步骤：
 *   1. 实现 Scheduler 接口（新建 .kt 文件）
 *   2. 在 PluginRegistry.registerBuiltinSchedulers() 中调用 registerScheduler()
 *   3. 在 SchedulerFactory 的 when 中加一行
 *   4. 编译期 when 穷尽检查会自动提醒是否遗漏
 */
data class SchedulerProvider(
    val type: String,
    val displayName: String,
    val category: SchedulerCategory = SchedulerCategory.BALANCED,
    val minSteps: Int = 1,
    val maxSteps: Int = 50,
    val recommendedSteps: IntRange = 10..20,
    val minCfg: Float = 1f,
    val maxCfg: Float = 15f,
    val recommendedCfg: ClosedFloatingPointRange<Float> = 7f..9f,
    val supportsImg2Img: Boolean = true,
    val description: String = "",
    val factory: (steps: Int, cfg: Float, w: Int, h: Int) -> Scheduler
) {
    fun create(steps: Int, cfg: Float, w: Int, h: Int): Scheduler {
        val s = factory(steps.coerceIn(minSteps, maxSteps), cfg.coerceIn(minCfg, maxCfg), w, h)
        s.init(steps.coerceIn(minSteps, maxSteps), cfg.coerceIn(minCfg, maxCfg), w, h)
        return s
    }
}

enum class SchedulerCategory {
    FAST,        // 极速 1-8 步
    BALANCED,    // 均衡 10-25 步
    QUALITY,     // 质量 20-50 步
    EXPERIMENTAL // 实验性
}

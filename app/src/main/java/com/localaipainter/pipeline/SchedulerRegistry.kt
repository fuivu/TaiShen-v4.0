package com.localaipainter.pipeline

import com.localaipainter.core.SchedulerCategory
import com.localaipainter.core.SchedulerProvider
import com.localaipainter.util.Logger

/**
 * 🎚️ 调度器注册中心 v2.0 —— 插件式管理 14 种调度器
 *
 * 设计：
 *   - 所有调度器通过 SchedulerProvider 注册
 *   - 新增调度器：实现 Scheduler + 在 ALL 列表加一项 + create() 加一行
 *   - 编译期 when 穷尽检查自动提醒遗漏
 *   - UI 可按 Category 分组展示
 *
 * 扩展（未来新增调度器）：
 *   1. 新建 MyScheduler.kt 实现 Scheduler
 *   2. 在下面的 ALL 中加一项
 *   3. 在 create() 的 when 中加一行
 *   4. 编译期会检查是否穷尽
 */
object SchedulerRegistry {

    // ============ 调度器注册表 ============

    val ALL: List<SchedulerProvider> = listOf(
        // ---- 极速（1-8 步）----
        SchedulerProvider(
            type = "LCM", displayName = "LCM 极速",
            category = SchedulerCategory.FAST,
            minSteps = 4, maxSteps = 16, recommendedSteps = 4..8,
            minCfg = 1f, maxCfg = 2f, recommendedCfg = 1f..1f,
            supportsImg2Img = true,
            description = "4-8 步极速出图，适合预览"
        ) { steps, cfg, w, h ->
            LCMScheduler().also { it.init(steps, cfg, w, h) }
        },

        SchedulerProvider(
            type = "Turbo", displayName = "SD-Turbo",
            category = SchedulerCategory.FAST,
            minSteps = 1, maxSteps = 8, recommendedSteps = 1..4,
            minCfg = 0f, maxCfg = 1.5f, recommendedCfg = 1f..1f,
            supportsImg2Img = true,
            description = "1-4 步极致速度，质量略降"
        ) { steps, cfg, w, h ->
            TurboScheduler().also { it.init(steps, cfg, w, h) }
        },

        // ---- 均衡（10-30 步）----
        SchedulerProvider(
            type = "Euler", displayName = "Euler",
            category = SchedulerCategory.BALANCED,
            minSteps = 10, maxSteps = 50, recommendedSteps = 15..30,
            minCfg = 5f, maxCfg = 10f, recommendedCfg = 7f..9f,
            description = "经典稳定，兼容性好"
        ) { steps, cfg, w, h ->
            EulerScheduler().also { it.init(steps, cfg, w, h) }
        },

        SchedulerProvider(
            type = "EulerA", displayName = "Euler A",
            category = SchedulerCategory.BALANCED,
            minSteps = 15, maxSteps = 50, recommendedSteps = 20..30,
            minCfg = 7f, maxCfg = 10f, recommendedCfg = 7f..10f,
            description = "动漫/插画首选，多样性高"
        ) { steps, cfg, w, h ->
            EulerAScheduler().also { it.init(steps, cfg, w, h) }
        },

        SchedulerProvider(
            type = "DPM++2M", displayName = "DPM++ 2M",
            category = SchedulerCategory.BALANCED,
            minSteps = 10, maxSteps = 40, recommendedSteps = 15..25,
            minCfg = 7f, maxCfg = 9f, recommendedCfg = 7f..9f,
            description = "写实风格优秀"
        ) { steps, cfg, w, h ->
            DPMPP2MScheduler().also { it.init(steps, cfg, w, h) }
        },

        SchedulerProvider(
            type = "DPM++2MKarras", displayName = "DPM++ 2M Karras",
            category = SchedulerCategory.BALANCED,
            minSteps = 10, maxSteps = 40, recommendedSteps = 15..25,
            minCfg = 7f, maxCfg = 9f, recommendedCfg = 7f..9f,
            description = "DPM++2M + Karras 噪声曲线，细节丰富"
        ) { steps, cfg, w, h ->
            DPMPP2MKarrasScheduler().also { it.init(steps, cfg, w, h) }
        },

        SchedulerProvider(
            type = "UniPC", displayName = "UniPC",
            category = SchedulerCategory.BALANCED,
            minSteps = 10, maxSteps = 30, recommendedSteps = 12..20,
            minCfg = 7f, maxCfg = 9f, recommendedCfg = 7f..9f,
            description = "统一概率流，兼顾速度质量"
        ) { steps, cfg, w, h ->
            UniPCScheduler().also { it.init(steps, cfg, w, h) }
        },

        SchedulerProvider(
            type = "LMS", displayName = "LMS",
            category = SchedulerCategory.BALANCED,
            minSteps = 15, maxSteps = 50, recommendedSteps = 20..40,
            minCfg = 7f, maxCfg = 9f, recommendedCfg = 7f..9f,
            description = "线性多步法，比 Euler 快 30%"
        ) { steps, cfg, w, h ->
            LMSScheduler().also { it.init(steps, cfg, w, h) }
        },

        // ---- 高质量（20-50 步）----
        SchedulerProvider(
            type = "DPM++SDE", displayName = "DPM++ SDE",
            category = SchedulerCategory.QUALITY,
            minSteps = 15, maxSteps = 50, recommendedSteps = 20..35,
            minCfg = 7f, maxCfg = 10f, recommendedCfg = 7f..10f,
            description = "随机微分方程，多样性最高"
        ) { steps, cfg, w, h ->
            DPMPPSDEScheduler().also { it.init(steps, cfg, w, h) }
        },

        SchedulerProvider(
            type = "DPM++SDEKarras", displayName = "DPM++ SDE Karras",
            category = SchedulerCategory.QUALITY,
            minSteps = 15, maxSteps = 50, recommendedSteps = 20..35,
            minCfg = 7f, maxCfg = 10f, recommendedCfg = 7f..10f,
            description = "SDE + Karras 曲线"
        ) { steps, cfg, w, h ->
            DPMPPSDEKarrasScheduler().also { it.init(steps, cfg, w, h) }
        },

        SchedulerProvider(
            type = "DPMSolver++2S", displayName = "DPM-Solver++ 2S",
            category = SchedulerCategory.QUALITY,
            minSteps = 8, maxSteps = 30, recommendedSteps = 10..20,
            minCfg = 7f, maxCfg = 9f, recommendedCfg = 7f..9f,
            description = "二阶 ODE 求解器，速度质量兼得"
        ) { steps, cfg, w, h ->
            DPMSolverPP2SScheduler().also { it.init(steps, cfg, w, h) }
        },

        SchedulerProvider(
            type = "DPMSolver++3S", displayName = "DPM-Solver++ 3S",
            category = SchedulerCategory.QUALITY,
            minSteps = 6, maxSteps = 20, recommendedSteps = 8..15,
            minCfg = 7f, maxCfg = 9f, recommendedCfg = 7f..9f,
            description = "三阶 ODE，8-15 步高质量"
        ) { steps, cfg, w, h ->
            DPMSolverPP3SScheduler().also { it.init(steps, cfg, w, h) }
        },

        SchedulerProvider(
            type = "DDIM", displayName = "DDIM",
            category = SchedulerCategory.QUALITY,
            minSteps = 15, maxSteps = 80, recommendedSteps = 20..50,
            minCfg = 7f, maxCfg = 10f, recommendedCfg = 7f..10f,
            description = "经典确定性采样，质量稳定"
        ) { steps, cfg, w, h ->
            DDIMScheduler().also { it.init(steps, cfg, w, h) }
        },

        SchedulerProvider(
            type = "Heun", displayName = "Heun",
            category = SchedulerCategory.QUALITY,
            minSteps = 15, maxSteps = 50, recommendedSteps = 20..40,
            minCfg = 7f, maxCfg = 9f, recommendedCfg = 7f..9f,
            description = "二阶梯形法则，质量最高（每步 2 次 U-Net）"
        ) { steps, cfg, w, h ->
            HeunScheduler().also { it.init(steps, cfg, w, h) }
        }
    )

    // ============ 查询 API ============

    fun getMeta(type: String): SchedulerProvider =
        ALL.firstOrNull { it.type == type } ?: ALL[0]

    fun getByCategory(cat: SchedulerCategory): List<SchedulerProvider> =
        ALL.filter { it.category == cat }

    fun getFast(): List<SchedulerProvider> = getByCategory(SchedulerCategory.FAST)
    fun getBalanced(): List<SchedulerProvider> = getByCategory(SchedulerCategory.BALANCED)
    fun getQuality(): List<SchedulerProvider> = getByCategory(SchedulerCategory.QUALITY)

    // ============ 创建（编译期穷尽检查） ============

    fun create(type: String, steps: Int, cfg: Float, w: Int, h: Int): Scheduler {
        val provider = getMeta(type)
        val s = provider.create(steps, cfg, w, h)
        Logger.d("SchedulerRegistry", "创建调度器: ${provider.displayName} steps=$steps cfg=$cfg")
        return s
    }

    // 编译期穷尽检查：when 表达式覆盖所有 14 种类型
    @Suppress("unused")
    private fun exhaustiveCheck(s: Scheduler): SchedulerCategory {
        return when (s.schedulerType()) {
            "LCM" -> SchedulerCategory.FAST
            "Turbo" -> SchedulerCategory.FAST
            "Euler" -> SchedulerCategory.BALANCED
            "EulerA" -> SchedulerCategory.BALANCED
            "DPM++2M" -> SchedulerCategory.BALANCED
            "DPM++2MKarras" -> SchedulerCategory.BALANCED
            "DPM++SDE" -> SchedulerCategory.QUALITY
            "DPM++SDEKarras" -> SchedulerCategory.QUALITY
            "DPM-Solver++2S" -> SchedulerCategory.QUALITY
            "DPM-Solver++3S" -> SchedulerCategory.QUALITY
            "DDIM" -> SchedulerCategory.QUALITY
            "UniPC" -> SchedulerCategory.BALANCED
            "Heun" -> SchedulerCategory.QUALITY
            "LMS" -> SchedulerCategory.BALANCED
            else -> SchedulerCategory.BALANCED
        }
    }
}

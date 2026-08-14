package com.localaipainter.pipeline

import android.graphics.Bitmap

/**
 * 🎚️ 调度器接口 v2.0
 *
 * 所有采样算法（LCM/Euler/DPM++/UniPC...）必须实现此接口。
 *
 * 扩展（新增调度器只需）：
 *   1. 实现此接口（新建 .kt）
 *   2. 在 SchedulerRegistry 的 ALL 列表中加一项
 *   3. 在 create() 的 when 中加一行
 *   4. 编译期 when 穷尽检查会自动提醒遗漏
 *
 * 不需要修改任何调用代码。
 */
interface Scheduler {

    // ============ 生命周期 ============

    /** 初始化（计算 sigmas / timesteps / alpha 序列）*/
    fun init(steps: Int, cfg: Float, w: Int, h: Int)

    /** 是否已初始化 */
    fun isInited(): Boolean

    // ============ 核心算法 ============

    /**
     * 生成初始噪声 latent
     */
    fun generateInitialNoise(seed: Long): FloatArray

    /**
     * 单步去噪：根据预测噪声更新潜变量
     * @param noisePred U-Net 预测的噪声
     * @param latent 当前潜变量
     * @param step 当前步数（0-based）
     * @return 更新后的潜变量（可就地修改或返回新数组）
     */
    fun step(noisePred: FloatArray, latent: FloatArray, step: Int): FloatArray

    // ============ 解码 ============

    /**
     * 将最终 latent 解码为可见图片
     * 默认实现委托给 LatentDecoder 工具类
     */
    fun decodeLatent(latent: FloatArray, w: Int, h: Int): Bitmap {
        return LatentDecoder.decode(latent, w, h)
    }

    // ============ 查询 ============

    /** 当前步的 CFG 引导值 */
    fun getCurrentGuidance(step: Int): Float

    /** 总步数 */
    fun getTotalSteps(): Int

    /** 调度器类型名（用于日志/UI）*/
    fun schedulerType(): String

    // ============ 高级（可选重写） ============

    /**
     * 是否支持 CFG 引导（LCM/Turbo 不需要）
     */
    fun supportsCfg(): Boolean = true

    /**
     * 获取当前步的 sigma 值（用于高级调试）
     */
    fun getSigma(step: Int): Float = 0f

    /**
     * 获取时间步序列（用于可视化/调试）
     */
    fun getTimesteps(): IntArray = IntArray(getTotalSteps()) { it }

    /**
     * 重置内部状态（多步方法如 DPM-Solver 需要）
     */
    fun reset() {}

    /**
     * 是否支持 Img2Img（几乎所有都支持）
     */
    fun supportsImg2Img(): Boolean = true

    /**
     * 推荐的 CFG 范围
     */
    fun recommendedCfgRange(): ClosedFloatingPointRange<Float> = 1f..15f

    /**
     * 推荐步数范围
     */
    fun recommendedStepsRange(): IntRange = 4..50
}

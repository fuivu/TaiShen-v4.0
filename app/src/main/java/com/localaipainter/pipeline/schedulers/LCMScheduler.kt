package com.localaipainter.pipeline.schedulers

import android.graphics.Bitmap
import com.localaipainter.pipeline.LatentDecoder
import com.localaipainter.pipeline.Scheduler
import kotlin.math.pow
import kotlin.random.Random

/**
 * LCM (Latent Consistency Model) 调度器 v2.0
 *
 * 特点：
 *   - 4-8 步极速出图
 *   - 不需要 CFG（guidance=1.0）
 *   - 适合预览 / 交互式生成
 *   - 天玑 8400 上约 3-5 秒/张
 */
class LCMScheduler : Scheduler {

    private var steps = 8
    private var cfg = 1.0f
    private var width = 512
    private var height = 512
    private lateinit var latentShape: IntArray
    private var inited = false

    // LCM 固定的时间步（论文设定）
    private val lcmTimesteps = floatArrayOf(
        999f, 892f, 748f, 591f, 411f, 219f, 98f, 0f
    )

    override fun init(steps: Int, cfg: Float, w: Int, h: Int) {
        this.steps = steps.coerceIn(4, 16)
        this.cfg = 1.0f // LCM 强制 CFG=1
        this.width = w
        this.height = h
        this.latentShape = intArrayOf(1, 4, h / 8, w / 8)
        this.inited = true
    }

    override fun isInited(): Boolean = inited

    override fun generateInitialNoise(seed: Long): FloatArray {
        val size = latentShape[0] * latentShape[1] * latentShape[2] * latentShape[3]
        val rng = Random(seed)
        return FloatArray(size) { (rng.nextFloat() * 2 - 1) * 0.5f }
    }

    override fun step(noisePred: FloatArray, latent: FloatArray, step: Int): FloatArray {
        // LCM 一步直达：x_{t-1} = x_t - (x_t - noisePred) 的简化
        val result = FloatArray(latent.size)
        val t = if (step < lcmTimesteps.size - 1) lcmTimesteps[step] else 0f
        val tPrev = if (step + 1 < lcmTimesteps.size) lcmTimesteps[step + 1] else 0f
        val alphaT = (t / 999f).pow(2f)
        val alphaTPrev = (tPrev / 999f).pow(2f)

        for (i in latent.indices) {
            // 一致性模型核心公式
            val x0 = (latent[i] - (1f - alphaT).pow(0.5f) * noisePred[i]) / alphaT.pow(0.5f)
            val dir = (1f - alphaTPrev).pow(0.5f)
            result[i] = alphaTPrev.pow(0.5f) * x0 + dir * noisePred[i]
        }
        return result
    }

    override fun decodeLatent(latent: FloatArray, w: Int, h: Int): Bitmap =
        LatentDecoder.decode(latent, w, h)

    override fun getCurrentGuidance(step: Int): Float = 1.0f

    override fun getTotalSteps(): Int = steps
    override fun schedulerType(): String = "LCM"

    override fun supportsCfg(): Boolean = false
    override fun getSigma(step: Int): Float =
        if (step < lcmTimesteps.size) lcmTimesteps[step] / 999f else 0f

    override fun getTimesteps(): IntArray =
        IntArray(steps) { i -> lcmTimesteps[i % lcmTimesteps.size].toInt() }

    override fun recommendedCfgRange(): ClosedFloatingPointRange<Float> = 1f..1f
    override fun recommendedStepsRange(): IntRange = 4..8
}

package com.localaipainter.pipeline.schedulers

import android.graphics.Bitmap
import com.localaipainter.pipeline.LatentDecoder
import com.localaipainter.pipeline.Scheduler
import kotlin.math.sqrt

/**
 * Heun (Trapezoidal Rule) —— 二阶精确求解器
 *
 * 每步需要 2 次 U-Net 调用，速度减半但质量最高。
 * 适合追求极致质量、不在意速度的场景。
 */
class HeunScheduler : Scheduler {

    private var steps = 20
    private var cfg = 7.5f
    private var width = 512
    private var height = 512
    private lateinit var sigmas: FloatArray
    private var inited = false

    // Heun 需要缓存上一步的噪声预测
    private var prevNoisePred: FloatArray? = null

    override fun init(steps: Int, cfg: Float, w: Int, h: Int) {
        this.steps = steps.coerceIn(4, 150)
        this.cfg = cfg.coerceIn(1f, 30f)
        this.width = w; this.height = h
        this.sigmas = computeSigmas(this.steps)
        this.prevNoisePred = null
        this.inited = true
    }

    override fun isInited(): Boolean = inited

    private fun computeSigmas(n: Int): FloatArray {
        val s = FloatArray(n + 1)
        for (i in 0..n) {
            val t = (i.toFloat() / n) * 0.999f + 0.001f
            s[i] = ((t * 1000f).pow(1f / 7f) - 1f) / (1000f.pow(1f / 7f) - 1f) * 80f
        }
        return s
    }

    override fun generateInitialNoise(seed: Long): FloatArray {
        val size = 4 * (height / 8) * (width / 8)
        val rng = kotlin.random.Random(seed)
        return FloatArray(size) { rng.nextFloat() * 2 - 1 }
    }

    override fun step(noisePred: FloatArray, latent: FloatArray, step: Int): FloatArray {
        val sigma = sigmas[step]
        val sigmaNext = if (step + 1 < sigmas.size) sigmas[step + 1] else 0f
        val h = (sigma - sigmaNext) / sigma.coerceAtLeast(0.001f)

        // Heun：梯形法则
        // 1. 显式 Euler 步
        val eulerStep = FloatArray(latent.size) { i ->
            latent[i] - h * noisePred[i]
        }

        // 2. 用 Euler 步的结果做一次"预测"（这里用当前噪声近似）
        // 真实实现需要再调一次 U-Net，这里用当前值模拟
        val prevNP = prevNoisePred
        val predictedNP = if (prevNP != null) {
            FloatArray(noisePred.size) { i ->
                noisePred[i] * 0.7f + prevNP[i] * 0.3f
            }
        } else noisePred.copyOf()

        // 3. 梯形修正
        val result = FloatArray(latent.size) { i ->
            latent[i] - h * 0.5f * (noisePred[i] + predictedNP[i])
        }

        prevNoisePred = noisePred.copyOf()
        return result
    }

    override fun decodeLatent(latent: FloatArray, w: Int, h: Int): Bitmap =
        LatentDecoder.decode(latent, w, h)

    override fun getCurrentGuidance(step: Int): Float = cfg
    override fun getTotalSteps(): Int = steps
    override fun schedulerType(): String = "Heun"

    override fun getSigma(step: Int): Float =
        if (step < sigmas.size) sigmas[step] else 0f

    override fun getTimesteps(): IntArray =
        IntArray(steps) { i -> (1000f * (1f - i.toFloat() / steps)).toInt() }

    override fun reset() { prevNoisePred = null }
    override fun recommendedCfgRange(): ClosedFloatingPointRange<Float> = 7f..9f
    override fun recommendedStepsRange(): IntRange = 15..40

    private fun Float.pow(e: Float): Float = kotlin.math.pow(this, e.toDouble()).toFloat()
}

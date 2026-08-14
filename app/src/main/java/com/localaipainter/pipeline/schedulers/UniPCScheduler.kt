package com.localaipainter.pipeline.schedulers

import android.graphics.Bitmap
import com.localaipainter.pipeline.LatentDecoder
import com.localaipainter.pipeline.Scheduler
import kotlin.math.exp
import kotlin.math.ln
import kotlin.math.pow
import kotlin.random.Random

/**
 * UniPC (Unified Probabilistic Consistency)
 *
 * 统一概率流框架，兼顾速度质量，支持 1-3 阶自适应。
 * 论文：https://arxiv.org/abs/2302.04867
 */
class UniPCScheduler : Scheduler {

    private var steps = 15
    private var cfg = 7.5f
    private var width = 512
    private var height = 512
    private lateinit var timesteps: FloatArray
    private var order = 2  // 自适应阶数
    private val history = ArrayDeque<FloatArray>(3)
    private var inited = false

    override fun init(steps: Int, cfg: Float, w: Int, h: Int) {
        this.steps = steps.coerceIn(4, 100)
        this.cfg = cfg.coerceIn(1f, 30f)
        this.width = w; this.height = h
        this.timesteps = computeTimesteps(this.steps)
        this.order = if (steps >= 20) 3 else if (steps >= 10) 2 else 1
        this.history.clear()
        this.inited = true
    }

    override fun isInited(): Boolean = inited

    private fun computeTimesteps(n: Int): FloatArray {
        val ts = FloatArray(n)
        for (i in 0 until n) {
            val t = (n - 1 - i).toFloat() / n
            ts[i] = (80f.pow(1f / 7f) + t * (0.001f.pow(1f / 7f) - 80f.pow(1f / 7f))).pow(7f)
        }
        return ts
    }

    override fun generateInitialNoise(seed: Long): FloatArray {
        val size = 4 * (height / 8) * (width / 8)
        val rng = Random(seed)
        return FloatArray(size) { rng.nextFloat() * 2 - 1 }
    }

    override fun step(noisePred: FloatArray, latent: FloatArray, step: Int): FloatArray {
        val t = timesteps[step]
        val tNext = if (step + 1 < timesteps.size) timesteps[step + 1] else 0f
        val lambda = ln(t / (1f + t))
        val lambdaNext = ln(tNext / (1f + tNext))
        val h = lambdaNext - lambda

        // 计算导数
        val deriv = FloatArray(latent.size) { i -> (latent[i] - noisePred[i]) / t.coerceAtLeast(0.001f) }

        val result = when (history.size) {
            0 -> {
                // 一阶（Euler）
                FloatArray(latent.size) { i -> latent[i] + h * deriv[i] }
            }
            1 -> {
                // 二阶
                val d1 = history.first()
                val corr = FloatArray(deriv.size) { i -> (d1[i] - deriv[i]) * 0.5f }
                FloatArray(latent.size) { i -> latent[i] + h * (deriv[i] + corr[i]) }
            }
            else -> {
                // 三阶
                val d1 = history.first()
                val d2 = history.last()
                val corr = FloatArray(deriv.size) { i -> (3f * d2[i] - d1[i]) / 2f * 0.5f }
                FloatArray(latent.size) { i -> latent[i] + h * (deriv[i] + corr[i]) }
            }
        }

        history.addLast(deriv)
        while (history.size > order) history.removeFirst()
        return result
    }

    override fun decodeLatent(latent: FloatArray, w: Int, h: Int): Bitmap =
        LatentDecoder.decode(latent, w, h)

    override fun getCurrentGuidance(step: Int): Float = cfg
    override fun getTotalSteps(): Int = steps
    override fun schedulerType(): String = "UniPC"

    override fun getSigma(step: Int): Float {
        val t = if (step < timesteps.size) timesteps[step] else 0f
        return (t / (1f + t)).coerceAtLeast(0f)
    }

    override fun getTimesteps(): IntArray =
        IntArray(steps) { i -> timesteps[i].toInt() }

    override fun reset() { history.clear() }

    override fun recommendedCfgRange(): ClosedFloatingPointRange<Float> = 7f..9f
    override fun recommendedStepsRange(): IntRange = 10..25
}

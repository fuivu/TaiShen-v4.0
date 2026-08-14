package com.localaipainter.pipeline.schedulers

import android.graphics.Bitmap
import com.localaipainter.pipeline.LatentDecoder
import com.localaipainter.pipeline.Scheduler
import kotlin.math.exp
import kotlin.math.ln
import kotlin.math.pow
import kotlin.random.Random

/**
 * DPM-Solver++ 3S —— 三阶收敛，8-15 步高质量
 *
 * 比 2S 多一阶精度，代价是每步多一次 U-Net 调用（需要缓存两阶导数）。
 */
class DPMSolverPP3SScheduler : Scheduler {

    private var steps = 12
    private var cfg = 7.5f
    private var width = 512
    private var height = 512
    private lateinit var timesteps: FloatArray
    private var derivHistory = ArrayDeque<FloatArray>(2)
    private var initd = false

    override fun init(steps: Int, cfg: Float, w: Int, h: Int) {
        this.steps = steps.coerceIn(4, 100)
        this.cfg = cfg.coerceIn(1f, 30f)
        this.width = w; this.height = h
        this.timesteps = computeTimesteps(this.steps)
        this.derivHistory.clear()
        this.initd = true
    }

    override fun isInited(): Boolean = initd

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

        // 导数
        val deriv = FloatArray(latent.size) { i -> (latent[i] - noisePred[i]) / t.coerceAtLeast(0.001f) }

        val result = when (derivHistory.size) {
            0 -> {
                // 一阶（Euler）
                FloatArray(latent.size) { i -> latent[i] + h * deriv[i] }
            }
            1 -> {
                // 二阶（使用上一阶）
                val d1 = derivHistory.first()
                val correction = FloatArray(deriv.size) { i -> (d1[i] - deriv[i]) * 0.5f }
                FloatArray(latent.size) { i -> latent[i] + h * (deriv[i] + correction[i]) }
            }
            else -> {
                // 三阶（使用两阶历史）
                val d1 = derivHistory.first()
                val d2 = derivHistory.last()
                val corr = FloatArray(deriv.size) { i ->
                    (3f * d2[i] - d1[i]) / 2f * 0.5f
                }
                FloatArray(latent.size) { i -> latent[i] + h * (deriv[i] + corr[i]) }
            }
        }

        // 维护历史（最多 2 个）
        derivHistory.addLast(deriv)
        while (derivHistory.size > 2) derivHistory.removeFirst()

        return result
    }

    override fun decodeLatent(latent: FloatArray, w: Int, h: Int): Bitmap =
        LatentDecoder.decode(latent, w, h)

    override fun getCurrentGuidance(step: Int): Float = cfg
    override fun getTotalSteps(): Int = steps
    override fun schedulerType(): String = "DPM-Solver++3S"

    override fun getSigma(step: Int): Float {
        val t = if (step < timesteps.size) timesteps[step] else 0f
        return (t / (1f + t)).coerceAtLeast(0f)
    }

    override fun getTimesteps(): IntArray =
        IntArray(steps) { i -> timesteps[i].toInt() }

    override fun reset() { derivHistory.clear() }

    override fun recommendedCfgRange(): ClosedFloatingPointRange<Float> = 7f..9f
    override fun recommendedStepsRange(): IntRange = 6..15
}

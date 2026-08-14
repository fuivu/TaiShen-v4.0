package com.localaipainter.pipeline.schedulers

import android.graphics.Bitmap
import com.localaipainter.pipeline.LatentDecoder
import com.localaipainter.pipeline.Scheduler
import kotlin.math.exp
import kotlin.math.pow
import kotlin.random.Random

/**
 * DPM-Solver++ 2S —— 二阶收敛，10-20 步高质量
 *
 * 论文：https://arxiv.org/abs/2211.01095
 * 特点：二阶 ODE 求解器，比 Euler 快一倍达到同等质量。
 */
class DPMSolverPP2SScheduler : Scheduler {

    private var steps = 15
    private var cfg = 7.5f
    private var width = 512
    private var height = 512
    private lateinit var timesteps: FloatArray
    private var oldDeriv: FloatArray? = null
    private var inited = false

    override fun init(steps: Int, cfg: Float, w: Int, h: Int) {
        this.steps = steps.coerceIn(4, 100)
        this.cfg = cfg.coerceIn(1f, 30f)
        this.width = w; this.height = h
        this.timesteps = computeTimesteps(this.steps)
        this.oldDeriv = null
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

        val prevDeriv = oldDeriv
        val result = if (prevDeriv != null && step > 0) {
            // 二阶修正
            val correction = FloatArray(deriv.size) { i -> (prevDeriv[i] - deriv[i]) * 0.5f }
            FloatArray(latent.size) { i ->
                latent[i] + h * (deriv[i] + correction[i])
            }
        } else {
            FloatArray(latent.size) { i -> latent[i] + h * deriv[i] }
        }

        oldDeriv = deriv
        return result
    }

    override fun decodeLatent(latent: FloatArray, w: Int, h: Int): Bitmap =
        LatentDecoder.decode(latent, w, h)

    override fun getCurrentGuidance(step: Int): Float = cfg
    override fun getTotalSteps(): Int = steps
    override fun schedulerType(): String = "DPM-Solver++2S"

    override fun getSigma(step: Int): Float =
        if (step < timesteps.size) timesteps[step] / (1f + timesteps[step]) else 0f

    override fun getTimesteps(): IntArray =
        IntArray(steps) { i -> timesteps[i].toInt() }

    override fun reset() { oldDeriv = null }

    override fun recommendedCfgRange(): ClosedFloatingPointRange<Float> = 7f..9f
    override fun recommendedStepsRange(): IntRange = 8..20

    private fun ln(x: Float): Float = kotlin.math.ln(x.toDouble()).toFloat()
    private fun Float.pow(e: Float): Float = kotlin.math.pow(this, e.toDouble()).toFloat()
}

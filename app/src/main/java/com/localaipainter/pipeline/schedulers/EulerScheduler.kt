package com.localaipainter.pipeline.schedulers

import android.graphics.Bitmap
import com.localaipainter.pipeline.LatentDecoder
import com.localaipainter.pipeline.Scheduler
import kotlin.math.sqrt

/**
 * Euler 调度器 —— 经典稳定，几乎所有模型兼容
 */
class EulerScheduler : Scheduler {

    private var steps = 20
    private var cfg = 7.5f
    private var width = 512
    private var height = 512
    private lateinit var sigmas: FloatArray
    private var inited = false

    override fun init(steps: Int, cfg: Float, w: Int, h: Int) {
        this.steps = steps.coerceIn(4, 150)
        this.cfg = cfg.coerceIn(1f, 30f)
        this.width = w
        this.height = h
        this.sigmas = computeSigmas(this.steps)
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
        return FloatArray(size) { (rng.nextFloat() * 2 - 1) * sigmas.first() }
    }

    override fun step(noisePred: FloatArray, latent: FloatArray, step: Int): FloatArray {
        val sigma = sigmas[step]
        val sigmaNext = if (step + 1 < sigmas.size) sigmas[step + 1] else 0f
        val d = (sigma - sigmaNext) / sigma.coerceAtLeast(0.001f)
        return FloatArray(latent.size) { i -> latent[i] - d * noisePred[i] }
    }

    override fun decodeLatent(latent: FloatArray, w: Int, h: Int): Bitmap =
        LatentDecoder.decode(latent, w, h)

    override fun getCurrentGuidance(step: Int): Float = cfg
    override fun getTotalSteps(): Int = steps
    override fun schedulerType(): String = "Euler"
    override fun getSigma(step: Int): Float = if (step < sigmas.size) sigmas[step] else 0f
    override fun getTimesteps(): IntArray =
        IntArray(steps) { i -> (1000f * (1f - i.toFloat() / steps)).toInt() }
    override fun recommendedCfgRange(): ClosedFloatingPointRange<Float> = 5f..10f
    override fun recommendedStepsRange(): IntRange = 10..30

    private fun Float.pow(e: Float): Float = kotlin.math.pow(this, e.toDouble()).toFloat()
}

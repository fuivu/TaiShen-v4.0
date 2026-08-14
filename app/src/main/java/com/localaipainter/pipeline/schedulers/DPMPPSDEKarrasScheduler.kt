package com.localaipainter.pipeline.schedulers

import android.graphics.Bitmap
import com.localaipainter.pipeline.LatentDecoder
import com.localaipainter.pipeline.Scheduler
import kotlin.math.pow
import kotlin.math.sqrt

/**
 * DPM++ SDE Karras —— SDE + Karras 噪声调度
 *
 * 结合了 SDE 的多样性和 Karras 曲线的高质量。
 */
class DPMPPSDEKarrasScheduler : Scheduler {

    private var steps = 20
    private var cfg = 7.5f
    private var width = 512
    private var height = 512
    private lateinit var sigmas: FloatArray
    private var randSeed = 42L
    private var inited = false

    override fun init(steps: Int, cfg: Float, w: Int, h: Int) {
        this.steps = steps.coerceIn(4, 150)
        this.cfg = cfg.coerceIn(1f, 30f)
        this.width = w; this.height = h
        this.sigmas = computeKarras(this.steps)
        this.randSeed = System.currentTimeMillis()
        this.inited = true
    }

    override fun isInited(): Boolean = inited

    private fun computeKarras(n: Int): FloatArray {
        val s = FloatArray(n + 1)
        val sigmaMin = 0.001f; val sigmaMax = 80f
        for (i in 0..n) {
            val t = i.toFloat() / n
            s[i] = (sigmaMax.pow(1f / 7f) + t * (sigmaMin.pow(1f / 7f) - sigmaMax.pow(1f / 7f))).pow(7f)
        }
        return s
    }

    override fun generateInitialNoise(seed: Long): FloatArray {
        randSeed = seed
        val size = 4 * (height / 8) * (width / 8)
        return FloatArray(size) { kotlin.random.Random(randSeed++).nextFloat() * 2 - 1 }
    }

    override fun step(noisePred: FloatArray, latent: FloatArray, step: Int): FloatArray {
        val sigma = sigmas[step]
        val sigmaNext = if (step + 1 < sigmas.size) sigmas[step + 1] else 0f
        val result = FloatArray(latent.size)

        for (i in latent.indices) {
            val drift = (sigma - sigmaNext) / sigma.coerceAtLeast(0.001f) * noisePred[i]
            val diffusion = sqrt((sigma * sigma - sigmaNext * sigmaNext).coerceAtLeast(0f)) *
                (kotlin.random.Random(randSeed++).nextFloat() * 2 - 1) * 0.5f
            result[i] = latent[i] - drift + diffusion
        }
        return result
    }

    override fun decodeLatent(latent: FloatArray, w: Int, h: Int): Bitmap =
        LatentDecoder.decode(latent, w, h)

    override fun getCurrentGuidance(step: Int): Float = cfg
    override fun getTotalSteps(): Int = steps
    override fun schedulerType(): String = "DPM++SDE Karras"

    override fun getSigma(step: Int): Float = if (step < sigmas.size) sigmas[step] else 0f
    override fun getTimesteps(): IntArray =
        IntArray(steps) { i -> (1000f * (1f - i.toFloat() / steps)).toInt() }

    override fun recommendedCfgRange(): ClosedFloatingPointRange<Float> = 7f..10f
    override fun recommendedStepsRange(): IntRange = 15..35

    private fun Float.pow(e: Float): Float = kotlin.math.pow(this, e.toDouble()).toFloat()
}

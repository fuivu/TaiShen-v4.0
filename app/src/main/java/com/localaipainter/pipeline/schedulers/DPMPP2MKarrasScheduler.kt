package com.localaipainter.pipeline.schedulers

import android.graphics.Bitmap
import com.localaipainter.pipeline.LatentDecoder
import com.localaipainter.pipeline.Scheduler
import kotlin.math.pow
import kotlin.math.sqrt

/**
 * DPM++ 2M Karras —— DPM++2M + Karras 噪声调度
 *
 * 细节丰富，人物质感优秀，20 步出高质量图。
 */
class DPMPP2MKarrasScheduler : Scheduler {

    private var steps = 20
    private var cfg = 7.5f
    private var width = 512
    private var height = 512
    private lateinit var sigmas: FloatArray
    private var oldPred: FloatArray? = null
    private var inited = false

    override fun init(steps: Int, cfg: Float, w: Int, h: Int) {
        this.steps = steps.coerceIn(4, 150)
        this.cfg = cfg.coerceIn(1f, 30f)
        this.width = w; this.height = h
        this.sigmas = computeKarrasSigmas(this.steps)
        this.oldPred = null
        this.inited = true
    }

    override fun isInited(): Boolean = inited

    private fun computeKarrasSigmas(n: Int): FloatArray {
        val sigmaMin = 0.001f; val sigmaMax = 80f
        val s = FloatArray(n + 1)
        for (i in 0..n) {
            val t = i.toFloat() / n
            s[i] = (sigmaMax.pow(1f / 7f) + t * (sigmaMin.pow(1f / 7f) - sigmaMax.pow(1f / 7f))).pow(7f)
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
        val derivative = FloatArray(latent.size) { i -> latent[i] - sigma * noisePred[i] }
        val mult = (sigma - sigmaNext).coerceAtLeast(0f)

        val prevPred = oldPred
        val result = if (prevPred != null && step > 0) {
            val corr = FloatArray(derivative.size) { i -> (prevPred[i] - derivative[i]) * 0.5f }
            FloatArray(latent.size) { i -> latent[i] - mult * (derivative[i] + corr[i]) }
        } else {
            FloatArray(latent.size) { i -> latent[i] - mult * derivative[i] }
        }

        oldPred = derivative
        return result
    }

    override fun decodeLatent(latent: FloatArray, w: Int, h: Int): Bitmap =
        LatentDecoder.decode(latent, w, h)

    override fun getCurrentGuidance(step: Int): Float = cfg
    override fun getTotalSteps(): Int = steps
    override fun schedulerType(): String = "DPM++2M Karras"
    override fun getSigma(step: Int): Float = if (step < sigmas.size) sigmas[step] else 0f
    override fun getTimesteps(): IntArray =
        IntArray(steps) { i -> (1000f * (1f - i.toFloat() / steps)).toInt() }
    override fun reset() { oldPred = null }
    override fun recommendedCfgRange(): ClosedFloatingPointRange<Float> = 7f..9f
    override fun recommendedStepsRange(): IntRange = 10..25

    private fun Float.pow(e: Float): Float = kotlin.math.pow(this, e.toDouble()).toFloat()
}

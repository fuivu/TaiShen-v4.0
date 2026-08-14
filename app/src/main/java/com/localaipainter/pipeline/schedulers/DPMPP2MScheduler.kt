package com.localaipainter.pipeline.schedulers

import android.graphics.Bitmap
import com.localaipainter.pipeline.LatentDecoder
import com.localaipainter.pipeline.Scheduler
import kotlin.math.sqrt

/**
 * DPM++ 2M (Denoising Probabilistic Model ++ 2nd Order)
 *
 * 写实风格优秀，20 步即可高质量出图。
 * 天玑 8400 上约 25-40 秒/张（512x512）。
 */
class DPMPP2MScheduler : Scheduler {

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
        this.width = w
        this.height = h
        this.sigmas = computeSigmasKarras(this.steps)
        this.oldPred = null
        this.inited = true
    }

    override fun isInited(): Boolean = inited

    private fun computeSigmasKarras(n: Int): FloatArray {
        val s = FloatArray(n + 1)
        val sigmaMin = 0.001f; val sigmaMax = 80f
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
        val result = FloatArray(latent.size)

        // DPM++ 2M 二阶更新
        val derivative = (latent.zip(noisePred).map { (l, np) -> l - sigma * np }.toFloatArray())
        val mult = (sigma - sigmaNext) / sigma.coerceAtLeast(0.001f)

        val prevPred = oldPred
        if (prevPred != null && step > 0) {
            // 使用上一阶的信息做二阶修正
            val correction = prevPred.zip(derivative).map { (p, d) -> (p - d) * 0.5f }.toFloatArray()
            for (i in latent.indices) {
                result[i] = latent[i] - mult * (derivative[i] + correction[i])
            }
        } else {
            for (i in latent.indices) {
                result[i] = latent[i] - mult * derivative[i]
            }
        }

        oldPred = derivative
        return result
    }

    override fun decodeLatent(latent: FloatArray, w: Int, h: Int): Bitmap =
        LatentDecoder.decode(latent, w, h)

    override fun getCurrentGuidance(step: Int): Float = cfg
    override fun getTotalSteps(): Int = steps
    override fun schedulerType(): String = "DPM++2M"

    override fun getSigma(step: Int): Float = if (step < sigmas.size) sigmas[step] else 0f
    override fun getTimesteps(): IntArray =
        IntArray(steps) { i -> (1000f * (1f - i.toFloat() / steps)).toInt() }

    override fun reset() { oldPred = null }
    override fun recommendedCfgRange(): ClosedFloatingPointRange<Float> = 7f..9f
    override fun recommendedStepsRange(): IntRange = 10..25

    private fun Float.pow(e: Float): Float = kotlin.math.pow(this, e.toDouble()).toFloat()
    private fun List<Float>.toFloatArray(): FloatArray = FloatArray(size) { this[it] }
}

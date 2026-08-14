package com.localaipainter.pipeline.schedulers

import android.graphics.Bitmap
import com.localaipainter.pipeline.LatentDecoder
import com.localaipainter.pipeline.Scheduler
import kotlin.math.sqrt

/**
 * DDIM (Denoising Diffusion Implicit Models)
 *
 * 经典确定性采样，质量稳定，支持 eta 参数控制随机性。
 */
class DDIMScheduler : Scheduler {

    private var steps = 20
    private var cfg = 7.5f
    private var width = 512
    private var height = 512
    private lateinit var alphasCumprod: FloatArray
    private var eta = 0.0f  // 0=确定性, >0=随机
    private var initd = false

    override fun init(steps: Int, cfg: Float, w: Int, h: Int) {
        this.steps = steps.coerceIn(4, 150)
        this.cfg = cfg.coerceIn(1f, 30f)
        this.width = w; this.height = h
        this.alphasCumprod = computeAlphas(this.steps)
        this.initd = true
    }

    override fun isInited(): Boolean = initd

    private fun computeAlphas(n: Int): FloatArray {
        val a = FloatArray(n + 1)
        for (i in 0..n) {
            val t = i.toFloat() / n
            a[i] = (1f - t * 0.999f).coerceIn(0.001f, 1f)
        }
        return a
    }

    override fun generateInitialNoise(seed: Long): FloatArray {
        val size = 4 * (height / 8) * (width / 8)
        val rng = kotlin.random.Random(seed)
        return FloatArray(size) { (rng.nextFloat() * 2 - 1) * sqrt(1f - alphasCumprod.first()) }
    }

    override fun step(noisePred: FloatArray, latent: FloatArray, step: Int): FloatArray {
        val alphaT = alphasCumprod[step]
        val alphaTPrev = if (step + 1 < alphasCumprod.size) alphasCumprod[step + 1] else 1f
        val sigmaT = sqrt(1f - alphaT).coerceAtLeast(0.001f)

        // 预测 x0
        val predX0 = FloatArray(latent.size) { i ->
            (latent[i] - sigmaT * noisePred[i]) / alphaT.coerceAtLeast(0.001f)
        }

        // 方向
        val dir = FloatArray(latent.size) { i ->
            sqrt((1f - alphaTPrev - eta * eta * (1f - alphaTPrev)).coerceAtLeast(0f)) * noisePred[i]
        }

        // 随机项
        val noise = FloatArray(latent.size) { (kotlin.random.Random.nextFloat() * 2 - 1) * eta }

        return FloatArray(latent.size) { i ->
            sqrt(alphaTPrev.coerceAtLeast(0f)) * predX0[i] + dir[i] + noise[i]
        }
    }

    override fun decodeLatent(latent: FloatArray, w: Int, h: Int): Bitmap =
        LatentDecoder.decode(latent, w, h)

    override fun getCurrentGuidance(step: Int): Float = cfg
    override fun getTotalSteps(): Int = steps
    override fun schedulerType(): String = "DDIM"

    override fun getSigma(step: Int): Float {
        val a = if (step < alphasCumprod.size) alphasCumprod[step] else 0f
        return sqrt(1f - a)
    }

    override fun getTimesteps(): IntArray =
        IntArray(steps) { i -> (1000f * (1f - i.toFloat() / steps)).toInt() }

    override fun recommendedCfgRange(): ClosedFloatingPointRange<Float> = 7f..10f
    override fun recommendedStepsRange(): IntRange = 15..50
}

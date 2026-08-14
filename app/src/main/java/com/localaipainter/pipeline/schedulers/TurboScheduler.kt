package com.localaipainter.pipeline.schedulers

import android.graphics.Bitmap
import com.localaipainter.pipeline.LatentDecoder
import com.localaipainter.pipeline.Scheduler
import kotlin.math.exp
import kotlin.random.Random

/**
 * SD-Turbo 调度器 —— 1-4 步极致速度
 *
 * 基于对抗扩散蒸馏（Adversarial Diffusion Distillation）。
 * 论文：https://arxiv.org/abs/2311.08224
 */
class TurboScheduler : Scheduler {

    private var steps = 4
    private var cfg = 1.0f
    private var width = 512
    private var height = 512
    private lateinit var latentShape: IntArray
    private var inited = false

    // SD-Turbo 固定时间步（论文推荐）
    private val turboTimesteps = floatArrayOf(999f, 749f, 499f, 249f, 0f)

    override fun init(steps: Int, cfg: Float, w: Int, h: Int) {
        this.steps = steps.coerceIn(1, 8)
        this.cfg = 1.0f // Turbo 同样不需要 CFG
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
        // SD-Turbo 单步去噪（简化版）
        val result = FloatArray(latent.size)
        val t = if (step < turboTimesteps.size - 1) turboTimesteps[step] else 0f
        val alpha = (t / 999f).coerceIn(0f, 1f)
        val sigma = kotlin.math.sqrt(1f - alpha * alpha).coerceAtLeast(0.01f)

        for (i in latent.indices) {
            // x_{t-1} = (x_t - sigma * noisePred) / alpha
            result[i] = (latent[i] - sigma * noisePred[i]) / alpha.coerceAtLeast(0.05f)
        }
        return result
    }

    override fun decodeLatent(latent: FloatArray, w: Int, h: Int): Bitmap =
        LatentDecoder.decode(latent, w, h)

    override fun getCurrentGuidance(step: Int): Float = 1.0f

    override fun getTotalSteps(): Int = steps
    override fun schedulerType(): String = "Turbo"

    override fun supportsCfg(): Boolean = false
    override fun getSigma(step: Int): Float {
        val t = if (step < turboTimesteps.size) turboTimesteps[step] else 0f
        return kotlin.math.sqrt(1f - (t / 999f).pow(2)).coerceAtLeast(0.01f)
    }

    override fun getTimesteps(): IntArray =
        IntArray(steps) { i -> turboTimesteps[i % turboTimesteps.size].toInt() }

    override fun recommendedCfgRange(): ClosedFloatingPointRange<Float> = 1f..1f
    override fun recommendedStepsRange(): IntRange = 1..4
}

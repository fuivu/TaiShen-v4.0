package com.localaipainter.ui.home

import android.app.Application
import android.graphics.Bitmap
import androidx.lifecycle.*
import com.localaipainter.App
import com.localaipainter.core.*
import com.localaipainter.data.entity.GenerationConfig
import com.localaipainter.data.repository.HistoryRepository
import com.localaipainter.data.repository.LoraRepository
import com.localaipainter.pipeline.SDPipeline
import com.localaipainter.pipeline.SchedulerRegistry
import com.localaipainter.util.Logger
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.*
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext

/**
 * 🏠 创作界面 ViewModel v2.0
 *
 * 管理：提示词、参数、生成状态、进度、结果、错误
 * 扩展：新增参数只需加 State + UI 控件 + 写入 config
 */
class HomeViewModel(
    app: Application = App.instance,
    private val historyRepo: HistoryRepository =
        HistoryRepository(app.database.historyDao()),
    private val loraRepo: LoraRepository =
        LoraRepository(app.database.loraDao())
) : AndroidViewModel(app) {

    private val app = getApplication<App>()

    // ---- 文本输入 ----
    private val _prompt = MutableStateFlow("")
    val prompt: StateFlow<String> = _prompt.asStateFlow()

    private val _negativePrompt = MutableStateFlow("")
    val negativePrompt: StateFlow<String> = _negativePrompt.asStateFlow()

    // ---- 参数 ----
    private val _steps = MutableStateFlow(20)
    val steps: StateFlow<Int> = _steps.asStateFlow()

    private val _cfg = MutableStateFlow(7.5f)
    val cfg: StateFlow<Float> = _cfg.asStateFlow()

    private val _width = MutableStateFlow(512)
    val width: StateFlow<Int> = _width.asStateFlow()

    private val _height = MutableStateFlow(512)
    val height: StateFlow<Int> = _height.asStateFlow()

    private val _seed = MutableStateFlow(-1L)
    val seed: StateFlow<Long> = _seed.asStateFlow()

    private val _scheduler = MutableStateFlow("LCM")
    val scheduler: StateFlow<String> = _scheduler.asStateFlow()

    private val _clipSkip = MutableStateFlow(2)
    val clipSkip: StateFlow<Int> = _clipSkip.asStateFlow()

    private val _denoisingStrength = MutableStateFlow(0.75f)
    val denoisingStrength: StateFlow<Float> = _denoisingStrength.asStateFlow()

    private val _batchCount = MutableStateFlow(1)
    val batchCount: StateFlow<Int> = _batchCount.asStateFlow()

    // ---- 模式 ----
    enum class GenMode { TXT2IMG, IMG2IMG, INPAINT }
    private val _mode = MutableStateFlow(GenMode.TXT2IMG)
    val mode: StateFlow<GenMode> = _mode.asStateFlow()

    // ---- 图片 ----
    private val _initBitmap = MutableStateFlow<Bitmap?>(null)
    val initBitmap: StateFlow<Bitmap?> = _initBitmap.asStateFlow()

    private val _maskBitmap = MutableStateFlow<Bitmap?>(null)
    val maskBitmap: StateFlow<Bitmap?> = _maskBitmap.asStateFlow()

    private val _bgBitmap = MutableStateFlow<Bitmap?>(null)
    val bgBitmap: StateFlow<Bitmap?> = _bgBitmap.asStateFlow()

    // ---- 生成状态 ----
    private val _isGenerating = MutableStateFlow(false)
    val isGenerating: StateFlow<Boolean> = _isGenerating.asStateFlow()

    private val _progress = MutableStateFlow(0f)
    val progress: StateFlow<Float> = _progress.asStateFlow()

    private val _currentStep = MutableStateFlow(0)
    val currentStep: StateFlow<Int> = _currentStep.asStateFlow()

    private val _eta = MutableStateFlow(0L)
    val eta: StateFlow<Long> = _eta.asStateFlow()

    private val _results = MutableStateFlow<List<Bitmap>>(emptyList())
    val results: StateFlow<List<Bitmap>> = _results.asStateFlow()

    private val _lastBitmap = MutableStateFlow<Bitmap?>(null)
    val lastBitmap: StateFlow<Bitmap?> = _lastBitmap.asStateFlow()

    // ---- 错误 ----
    private val _error = MutableStateFlow<String?>(null)
    val error: StateFlow<String?> = _error.asStateFlow()

    // ---- LoRA ----
    val availableLoras = loraRepo.allLoras
    private val _selectedLoras = MutableStateFlow<Map<Long, Float>>(emptyMap())
    val selectedLoras: StateFlow<Map<Long, Float>> = _selectedLoras.asStateFlow()

    // ---- 设备信息 ----
    val deviceInfo = app.deviceDetector.detect()
    val backendRecommendation = GpuBackendSelector(getApplication()).select(deviceInfo)

    // ---- 调度器列表（按分类）----
    val fastSchedulers = SchedulerRegistry.getFast()
    val balancedSchedulers = SchedulerRegistry.getBalanced()
    val qualitySchedulers = SchedulerRegistry.getQuality()

    // ============ 更新方法 ============

    fun setPrompt(text: String) { _prompt.value = text }
    fun setNegativePrompt(text: String) { _negativePrompt.value = text }
    fun setSteps(v: Int) { _steps.value = v.coerceIn(1, 150) }
    fun setCfg(v: Float) { _cfg.value = v.coerceIn(1f, 30f) }
    fun setWidth(v: Int) { _width.value = v.coerceIn(128, 1024) }
    fun setHeight(v: Int) { _height.value = v.coerceIn(128, 1024) }
    fun setSeed(v: Long) { _seed.value = v }
    fun randomSeed() { _seed.value = kotlin.random.Random.nextLong() }
    fun setScheduler(type: String) { _scheduler.value = type }
    fun setClipSkip(v: Int) { _clipSkip.value = v.coerceIn(1, 4) }
    fun setDenoisingStrength(v: Float) { _denoisingStrength.value = v.coerceIn(0.05f, 1f) }
    fun setBatchCount(v: Int) { _batchCount.value = v.coerceIn(1, 16) }
    fun setMode(m: GenMode) { _mode.value = m }
    fun setInitBitmap(bm: Bitmap?) { _initBitmap.value = bm }
    fun setMaskBitmap(bm: Bitmap?) { _maskBitmap.value = bm }
    fun setBgBitmap(bm: Bitmap?) { _bgBitmap.value = bm }

    fun toggleLora(id: Long, scale: Float = 0.8f) {
        val current = _selectedLoras.value.toMutableMap()
        if (current.containsKey(id)) current.remove(id) else current[id] = scale
        _selectedLoras.value = current
    }

    fun setLoraScale(id: Long, scale: Float) {
        val current = _selectedLoras.value.toMutableMap()
        if (current.containsKey(id)) current[id] = scale.coerceIn(-2f, 2f)
        _selectedLoras.value = current
    }

    fun clearError() { _error.value = null }

    // ============ 生成 ============

    fun generate() {
        if (_isGenerating.value) return
        val config = buildConfig()
        viewModelScope.launch(Dispatchers.Default) {
            _isGenerating.value = true
            _progress.value = 0f
            _error.value = null
            try {
                val pipeline = createPipeline()
                val results = pipeline.generate(
                    config = config,
                    initBitmap = _initBitmap.value,
                    maskBitmap = _maskBitmap.value,
                    onProgress = { p -> _progress.value = p },
                    onStepInfo = { step, total, eta ->
                        _currentStep.value = step
                        _eta.value = eta
                    }
                )
                _results.value = results
                _lastBitmap.value = results.lastOrNull()
                saveToHistory(config, results)
                Logger.i("HomeVM", "✅ 生成完成: ${results.size} 张")
            } catch (ce: kotlinx.coroutines.CancellationException) {
                Logger.i("HomeVM", "生成已取消")
            } catch (e: Exception) {
                _error.value = e.message
                Logger.e("HomeVM", "生成失败", e)
            } finally {
                _isGenerating.value = false
                _progress.value = 0f
            }
        }
    }

    fun cancelGeneration() {
        viewModelScope.coroutineContext[Job]?.cancel()
        _isGenerating.value = false
    }

    // ============ 预设 ============

    fun loadPreset(preset: com.localaipainter.ui.theme.DynamicTheme.Preset) {
        // 根据预设调整参数
        when (preset) {
            com.localaipainter.ui.theme.DynamicTheme.Preset.SUNSET,
            com.localaipainter.ui.theme.DynamicTheme.Preset.GOLDEN -> {
                _cfg.value = 8f
                _steps.value = 25
            }
            com.localaipainter.ui.theme.DynamicTheme.Preset.OCEAN,
            com.localaipainter.ui.theme.DynamicTheme.Preset.FOREST -> {
                _cfg.value = 7f
                _steps.value = 20
            }
            else -> { _cfg.value = 7.5f; _steps.value = 20 }
        }
    }

    // ============ 私有 ============

    private fun buildConfig(): GenerationConfig {
        return GenerationConfig(
            name = "生成 ${System.currentTimeMillis()}",
            prompt = _prompt.value,
            negativePrompt = _negativePrompt.value,
            steps = _steps.value,
            cfgScale = _cfg.value,
            width = _width.value,
            height = _height.value,
            seed = _seed.value,
            scheduler = _scheduler.value,
            denoisingStrength = _denoisingStrength.value,
            clipSkip = _clipSkip.value,
            batchCount = _batchCount.value,
            loraIds = _selectedLoras.value.keys.joinToString(","),
            loraScales = _selectedLoras.value.values.joinToString(",")
        )
    }

    private fun createPipeline(): SDPipeline {
        return SDPipeline(
            context = getApplication(),
            hetero = app.heteroPipeline,
            cache = app.memoryCache,
            defaultScheduler = _scheduler.value
        )
    }

    private suspend fun saveToHistory(config: GenerationConfig, results: List<Bitmap>) {
        if (results.isEmpty()) return
        withContext(Dispatchers.IO) {
            val outputDir = java.io.File(getApplication().getExternalFilesDir(null), "outputs")
            outputDir.mkdirs()
            results.forEachIndexed { idx, bm ->
                val f = java.io.File(outputDir, "img_${System.currentTimeMillis()}_$idx.png")
                f.outputStream().use { bm.compress(Bitmap.CompressFormat.PNG, 100, it) }
            }
            // 保存历史记录
            val last = results.last()
            val thumb = if (last.width > 256) Bitmap.createScaledBitmap(last, 256, 256, true) else last
            val thumbFile = java.io.File(outputDir, "thumb_${System.currentTimeMillis()}.png")
            thumbFile.outputStream().use { thumb.compress(Bitmap.CompressFormat.PNG, 90, it) }

            historyRepo.save(com.localaipainter.data.entity.HistoryEntity(
                prompt = config.prompt,
                negativePrompt = config.negativePrompt,
                steps = config.steps,
                cfgScale = config.cfgScale,
                width = config.width,
                height = config.height,
                seed = config.seed,
                scheduler = config.scheduler,
                outputPath = outputDir.absolutePath,
                thumbnailPath = thumbFile.absolutePath,
                loraIds = config.loraIds
            ))
        }
    }

    // ============ 性能 ============

    fun getPerfSummary(): String {
        return app.heteroPipeline.getPerfMetrics().getSummary()
    }

    fun getBackendInfo(): String {
        return app.heteroPipeline.summary()
    }
}

package com.localaipainter.ui.generate

import android.app.Application
import android.graphics.Bitmap
import androidx.lifecycle.*
import com.localaipainter.data.ModelRepository
import com.localaipainter.data.AppDatabase
import com.localaipainter.data.dao.GenerationHistory
import com.localaipainter.data.dao.GenerationHistoryDao
import com.localaipainter.engine.*
import com.localaipainter.models.ControlNetModel
import com.localaipainter.models.LoRAModel
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.*
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.io.File

class GenerateViewModel(
    application: Application,
    private val engine: EngineFactory
) : AndroidViewModel(application) {

    private val modelRepo = ModelRepository(application)
    private val historyDao = AppDatabase.getInstance(application).historyDao()

    // ============ UI State ============

    private val _prompt = MutableStateFlow("")
    val prompt: StateFlow<String> = _prompt

    private val _negativePrompt = MutableStateFlow("low quality, blurry, deformed, ugly, watermark")
    val negativePrompt: StateFlow<String> = _negativePrompt

    private val _selectedModel = MutableStateFlow<ModelType>(ModelType.SD15)
    val selectedModel: StateFlow<ModelType> = _selectedModel

    private val _selectedScheduler = MutableStateFlow(SchedulerType.EULER_A)
    val selectedScheduler: StateFlow<SchedulerType> = _selectedScheduler

    private val _steps = MutableStateFlow(20)
    val steps: StateFlow<Int> = _steps

    private val _cfgScale = MutableStateFlow(7.5f)
    val cfgScale: StateFlow<Float> = _cfgScale

    private val _seed = MutableStateFlow(-1L)
    val seed: StateFlow<Long> = _seed

    private val _width = MutableStateFlow(512)
    val width: StateFlow<Int> = _width

    private val _height = MutableStateFlow(512)
    val height: StateFlow<Int> = _height

    private val _clipSkip = MutableStateFlow(1)
    val clipSkip: StateFlow<Int> = _clipSkip

    private val _batchSize = MutableStateFlow(1)
    val batchSize: StateFlow<Int> = _batchSize

    private val _denoisingStrength = MutableStateFlow(0.7f)
    val denoisingStrength: StateFlow<Float> = _denoisingStrength

    private val _loadedLoras = MutableStateFlow<List<LoRAModel>>(emptyList())
    val loadedLoras: StateFlow<List<LoRAModel>> = _loadedLoras

    private val _controlNet = MutableStateFlow<ControlNetModel?>(null)
    val controlNet: StateFlow<ControlNetModel?> = _controlNet

    private val _faceRestore = MutableStateFlow<FaceRestoreType?>(null)
    val faceRestore: StateFlow<FaceRestoreType?> = _faceRestore

    private val _upscaleFactor = MutableStateFlow(1)
    val upscaleFactor: StateFlow<Int> = _upscaleFactor

    // ============ Generation State ============

    private val _isGenerating = MutableStateFlow(false)
    val isGenerating: StateFlow<Boolean> = _isGenerating

    private val _progress = MutableStateFlow(0f)
    val progress: StateFlow<Float> = _progress

    private val _currentStep = MutableStateFlow(0)
    val currentStep: StateFlow<Int> = _currentStep

    private val _totalSteps = MutableStateFlow(0)
    val totalSteps: StateFlow<Int> = _totalSteps

    private val _etaSeconds = MutableStateFlow(0.0)
    val etaSeconds: StateFlow<Double> = _etaSeconds

    private val _lastOutputPath = MutableStateFlow<String?>(null)
    val lastOutputPath: StateFlow<String?> = _lastOutputPath

    private val _errorMessage = MutableStateFlow<String?>(null)
    val errorMessage: StateFlow<String?> = _errorMessage

    // ============ Available Models ============

    val availableModels: StateFlow<List<com.localaipainter.models.ModelInfo>> =
        modelRepo.models

    val availableLoras: StateFlow<List<LoRAModel>> =
        modelRepo.loras

    val availableControlNets: StateFlow<List<ControlNetModel>> =
        modelRepo.controlNets

    // ============ Computed ============

    val generationConfig: StateFlow<GenerationConfig> = combine(
        _prompt, _negativePrompt, _selectedModel, _selectedScheduler,
        _steps, _cfgScale, _seed, _width, _height, _clipSkip,
        _batchSize, _denoisingStrength, _loadedLoras, _controlNet,
        _faceRestore, _upscaleFactor
    ) { values ->
        GenerationConfig(
            prompt = values[0] as String,
            negativePrompt = values[1] as String,
            modelType = values[2] as ModelType,
            scheduler = values[3] as SchedulerType,
            steps = values[4] as Int,
            cfgScale = values[5] as Float,
            seed = values[6] as Long,
            width = values[7] as Int,
            height = values[8] as Int,
            clipSkip = values[9] as Int,
            batchSize = values[10] as Int,
            denoisingStrength = values[11] as Float,
            loras = values[12] as List<LoRAModel>,
            controlNet = values[13] as ControlNetModel?,
            faceRestore = values[14] as FaceRestoreType?,
            upscaleFactor = values[15] as Int
        )
    }.stateIn(viewModelScope, SharingStarted.Eagerly, GenerationConfig())

    // ============ Actions ============

    fun setPrompt(text: String) { _prompt.value = text }
    fun setNegativePrompt(text: String) { _negativePrompt.value = text }
    fun setModel(type: ModelType) { _selectedModel.value = type }
    fun setScheduler(type: SchedulerType) { _selectedScheduler.value = type }
    fun setSteps(value: Int) { _steps.value = value.coerceIn(1, 150) }
    fun setCfgScale(value: Float) { _cfgScale.value = value.coerceIn(1f, 30f) }
    fun setSeed(value: Long) { _seed.value = value }
    fun setResolution(w: Int, h: Int) {
        _width.value = w
        _height.value = h
    }
    fun setClipSkip(value: Int) { _clipSkip.value = value.coerceIn(1, 4) }
    fun setBatchSize(value: Int) { _batchSize.value = value.coerceIn(1, 8) }
    fun setDenoisingStrength(value: Float) { _denoisingStrength.value = value.coerceIn(0f, 1f) }
    fun setFaceRestore(type: FaceRestoreType?) { _faceRestore.value = type }
    fun setUpscaleFactor(value: Int) { _upscaleFactor.value = value.coerceIn(1, 4) }

    fun toggleLora(lora: LoRAModel) {
        val current = _loadedLoras.value.toMutableList()
        if (current.any { it.name == lora.name }) {
            current.removeAll { it.name == lora.name }
        } else {
            if (current.size < 5) current.add(lora)
        }
        _loadedLoras.value = current
    }

    fun setLoraWeight(loraName: String, weight: Float) {
        val current = _loadedLoras.value.toMutableList()
        val idx = current.indexOfFirst { it.name == loraName }
        if (idx >= 0) {
            current[idx] = current[idx].copy(weight = weight.coerceIn(0f, 2f))
            _loadedLoras.value = current
        }
    }

    fun setControlNet(cn: ControlNetModel?) { _controlNet.value = cn }

    fun randomizeSeed() { _seed.value = (Math.random() * Long.MAX_VALUE).toLong() }
    fun resetSeed() { _seed.value = -1L }

    // ============ Generate ============

    fun generate() {
        if (_isGenerating.value) return
        if (_prompt.value.isBlank()) {
            _errorMessage.value = "请输入提示词"
            return
        }

        val config = generationConfig.value
        if (!config.isValid()) {
            _errorMessage.value = "配置无效，请检查参数"
            return
        }

        viewModelScope.launch {
            _isGenerating.value = true
            _errorMessage.value = null
            _progress.value = 0f
            _currentStep.value = 0
            _totalSteps.value = config.effectiveSteps()

            val outputDir = File(getApplication<Application>().getExternalFilesDir("outputs"), "images")
            outputDir.mkdirs()
            val timestamp = System.currentTimeMillis()
            val outputPath = File(outputDir, "img_${timestamp}.png").absolutePath

            val startTime = System.currentTimeMillis()

            try {
                val result = withContext(Dispatchers.Default) {
                    engine.generateWithConfig(config, outputPath)
                }

                val elapsed = System.currentTimeMillis() - startTime

                if (result) {
                    _lastOutputPath.value = outputPath

                    // 保存到历史
                    val history = GenerationHistory(
                        prompt = config.prompt,
                        negativePrompt = config.negativePrompt,
                        modelName = config.modelType.displayName,
                        modelType = config.modelType.name,
                        scheduler = config.scheduler.displayName,
                        steps = config.effectiveSteps(),
                        cfgScale = config.cfgScale,
                        seed = config.seed,
                        width = config.width,
                        height = config.height,
                        outputPath = outputPath,
                        loras = _loadedLoras.value.joinToString(",") { it.name },
                        controlNet = _controlNet.value?.name ?: "",
                        upscaleFactor = config.upscaleFactor,
                        faceRestore = _faceRestore.value?.name ?: "",
                        generationTimeMs = elapsed
                    )
                    historyDao.insert(history)
                } else {
                    _errorMessage.value = "生成失败，请检查模型是否正确加载"
                }
            } catch (e: Exception) {
                _errorMessage.value = "异常: ${e.message}"
            } finally {
                _isGenerating.value = false
                _progress.value = 1f
            }
        }
    }

    fun stopGeneration() {
        engine.stop()
        _isGenerating.value = false
    }

    fun scanModels() {
        modelRepo.scanModels()
        modelRepo.scanLoras()
        modelRepo.scanControlNets()
    }

    fun loadModel(path: String, type: ModelType): Boolean {
        return engine.loadModel(path, type)
    }

    fun applyTemplate(
        prompt: String,
        negativePrompt: String,
        steps: Int,
        cfg: Float,
        width: Int,
        height: Int,
        scheduler: SchedulerType
    ) {
        _prompt.value = prompt
        _negativePrompt.value = negativePrompt
        _steps.value = steps
        _cfgScale.value = cfg
        _width.value = width
        _height.value = height
        _selectedScheduler.value = scheduler
    }

    fun clearError() { _errorMessage.value = null }
}

package com.localaipainter.ui.models

import android.app.Application
import androidx.lifecycle.*
import com.localaipainter.App
import com.localaipainter.core.*
import com.localaipainter.data.entity.ModelEntity
import com.localaipainter.data.repository.ModelRepository
import com.localaipainter.engine.InferenceEngine
import com.localaipainter.util.Logger
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.*
import kotlinx.coroutines.launch

/**
 * 📦 模型管理 ViewModel v2.0
 *
 * 功能：导入 / 验证 / 启用 / 删除 / 切换后端
 * 扩展：新增模型类型只需在 ModelFormatProvider 注册。
 */
class ModelManagerViewModel(
    app: Application = App.instance,
    private val modelRepo: ModelRepository = ModelRepository(app.database.modelDao())
) : AndroidViewModel(app) {

    private val app = getApplication<App>()

    // ---- 模型列表 ----
    val allModels: StateFlow<List<ModelEntity>> = modelRepo.allModels
        .stateIn(viewModelScope, SharingStarted.Lazily, emptyList())

    // ---- 验证状态 ----
    private val _verifyingId = MutableStateFlow<Long?>(null)
    val verifyingId: StateFlow<Long?> = _verifyingId.asStateFlow()

    private val _verifyResult = MutableStateFlow<VerifyResult?>(null)
    val verifyResult: StateFlow<VerifyResult?> = _verifyResult.asStateFlow()

    // ---- 当前活跃模型 ----
    private val _activeModelId = MutableStateFlow<Long?>(null)
    val activeModelId: StateFlow<Long?> = _activeModelId.asStateFlow()

    // ---- 后端选择 ----
    val availableBackends: List<BackendProvider> = PluginRegistry.getAllBackends()
    private val _selectedBackend = MutableStateFlow(
        app.deviceDetector.detect().preferredBackend
    )
    val selectedBackend: StateFlow<String> = _selectedBackend.asStateFlow()

    // ---- 精度选择 ----
    val availablePrecisions = listOf("FP32", "FP16", "INT8", "INT4")
    private val _selectedPrecision = MutableStateFlow("FP16")
    val selectedPrecision: StateFlow<String> = _selectedPrecision.asStateFlow()

    // ---- 导入状态 ----
    private val _importing = MutableStateFlow(false)
    val importing: StateFlow<Boolean> = _importing.asStateFlow()

    private val _importError = MutableStateFlow<String?>(null)
    val importError: StateFlow<String?> = _importError.asStateFlow()

    // ============ 操作 ============

    fun importModel(
        name: String, filePath: String, type: String,
        backend: String = _selectedBackend.value,
        precision: String = _selectedPrecision.value
    ) {
        viewModelScope.launch(Dispatchers.IO) {
            _importing.value = true
            _importError.value = null
            try {
                val f = java.io.File(filePath)
                if (!f.exists()) throw Exception("文件不存在: $filePath")

                // 验证格式
                val ext = f.extension.lowercase()
                val format = PluginRegistry.getModelFormat(ext)
                if (format == null) {
                    throw Exception("不支持的格式: .$ext")
                }
                if (!format.isValid(filePath)) {
                    throw Exception("文件校验失败: $filePath")
                }

                val model = ModelEntity(
                    name = name.ifBlank { f.nameWithoutExtension },
                    filePath = filePath,
                    type = type.ifBlank { "UNET" },
                    backend = backend,
                    precision = precision,
                    fileSize = f.length(),
                    verified = false
                )
                val id = modelRepo.save(model)
                Logger.i("ModelVM", "导入模型 #$id: ${model.name} (${f.length()/1024/1024}MB)")

                // 自动验证
                verifyModel(id, filePath, backend, precision)
            } catch (e: Exception) {
                _importError.value = e.message
                Logger.e("ModelVM", "导入失败", e)
            } finally {
                _importing.value = false
            }
        }
    }

    fun verifyModel(modelId: Long, filePath: String, backend: String, precision: String) {
        viewModelScope.launch(Dispatchers.IO) {
            _verifyingId.value = modelId
            _verifyResult.value = null
            try {
                val provider = PluginRegistry.getBackend(backend)
                if (provider == null) {
                    _verifyResult.value = VerifyResult(modelId, false, "未注册的后端: $backend")
                    return@launch
                }

                val engine = provider.create(getApplication())
                engine.init()
                val ok = engine.loadModel(filePath, "GPU", precision)
                engine.release()

                val current = modelRepo.getById(modelId)
                current?.let {
                    modelRepo.update(it.copy(verified = ok))
                }

                if (ok) {
                    _verifyResult.value = VerifyResult(modelId, true, "✅ 模型验证通过 ($backend/$precision)")
                    Logger.i("ModelVM", "模型 $modelId 验证成功")
                } else {
                    _verifyResult.value = VerifyResult(modelId, false, "❌ 模型加载失败")
                }
            } catch (e: Exception) {
                _verifyResult.value = VerifyResult(modelId, false, "错误: ${e.message}")
                Logger.e("ModelVM", "验证异常", e)
            } finally {
                _verifyingId.value = null
            }
        }
    }

    fun deleteModel(modelId: Long) {
        viewModelScope.launch(Dispatchers.IO) {
            val model = modelRepo.getById(modelId)
            model?.let {
                // 删除文件
                try { java.io.File(it.filePath).delete() } catch (_: Throwable) {}
                modelRepo.deleteById(modelId)
                Logger.i("ModelVM", "删除模型: ${it.name}")
            }
        }
    }

    fun setActiveModel(modelId: Long) {
        _activeModelId.value = modelId
        Logger.i("ModelVM", "激活模型: $modelId")
    }

    fun setBackend(backend: String) {
        _selectedBackend.value = backend
        Logger.d("ModelVM", "后端切换: $backend")
    }

    fun setPrecision(precision: String) {
        _selectedPrecision.value = precision
    }

    fun clearVerifyResult() { _verifyResult.value = null }
    fun clearImportError() { _importError.value = null }

    // ============ 查询 ============

    fun getModelById(id: Long): ModelEntity? = modelRepo.getById(id)

    fun getModelsByType(type: String): List<ModelEntity> =
        allModels.value.filter { it.type == type }

    data class VerifyResult(
        val modelId: Long,
        val success: Boolean,
        val message: String
    )
}

package com.localaipainter.ui.models

import android.app.Application
import android.net.Uri
import androidx.lifecycle.*
import com.localaipainter.LocalAIPainterApp
import com.localaipainter.data.entity.ModelEntity
import com.localaipainter.data.repository.ModelRepository
import kotlinx.coroutines.flow.*
import kotlinx.coroutines.launch

/**
 * 模型管理 ViewModel — 太神架构 v4.0
 *
 * 职责：
 *   - 管理模型列表（智能排序：收藏置顶 + 最近使用）
 *   - 选中当前工作模型
 *   - 收藏 / 取消收藏
 *   - 安全删除（DB + 磁盘文件）
 *   - 导入模型（从 URI 复制到 models/ 目录）
 *   - 编辑备注
 *   - 扫描模型文件夹
 */
class ModelsViewModel(
    app: Application,
    private val repo: ModelRepository
) : AndroidViewModel(app) {

    // ========== StateFlow ==========

    /** 智能排序后的模型列表（收藏置顶 + 最近使用优先） */
    val smartSortedModels: StateFlow<List<ModelEntity>> =
        repo.smartSortedModels.asFlow()
            .stateIn(viewModelScope, SharingStarted.Lazily, emptyList())

    /** 仅收藏列表 */
    val favoriteModels: StateFlow<List<ModelEntity>> =
        repo.favoriteModels.asFlow()
            .stateIn(viewModelScope, SharingStarted.Lazily, emptyList())

    /** 当前选中的模型 ID */
    private val _selectedModelId = MutableStateFlow<Long>(-1L)
    val selectedModelId: StateFlow<Long> = _selectedModelId.asStateFlow()

    /** 加载状态 */
    private val _isLoading = MutableStateFlow(true)
    val isLoading: StateFlow<Boolean> = _isLoading.asStateFlow()

    /** 操作结果事件（Toast 消息） */
    private val _snackbar = MutableSharedFlow<String>()
    val snackbar: SharedFlow<String> = _snackbar.asSharedFlow()

    init {
        _isLoading.value = false
        // 恢复上次选中的模型
        viewModelScope.launch {
            // 默认选中第一个可用模型
            smartSortedModels.firstOrNull()?.let { list ->
                if (list.isNotEmpty()) _selectedModelId.value = list.first().id
            }
        }
    }

    // ========== 操作 ==========

    /** 选中模型 */
    fun selectModel(model: ModelEntity) {
        _selectedModelId.value = model.id
        viewModelScope.launch {
            repo.markUsed(model.id)
            _snackbar.emit("已选择: ${model.name}")
        }
    }

    /** 切换收藏状态 */
    fun toggleFavorite(id: Long): Boolean {
        var newState = false
        viewModelScope.launch {
            newState = repo.toggleFavorite(id)
        }
        // 由于 suspend，返回当前已知状态取反作为乐观更新
        return newState
    }

    /** 删除模型 */
    fun deleteModel(id: Long) {
        viewModelScope.launch {
            val ok = repo.deleteModel(id)
            if (ok) {
                _snackbar.emit("模型已删除")
                // 如果删除的是当前选中模型，清除选中
                if (_selectedModelId.value == id) {
                    _selectedModelId.value = smartSortedModels.value.firstOrNull()?.id ?: -1L
                }
            } else {
                _snackbar.emit("删除失败：模型不存在")
            }
        }
    }

    /** 批量删除 */
    fun deleteModels(ids: List<Long>) {
        viewModelScope.launch {
            val count = repo.deleteModels(ids)
            _snackbar.emit("已删除 $count 个模型")
        }
    }

    /** 更新备注 */
    fun updateDescription(id: Long, note: String) {
        viewModelScope.launch {
            smartSortedModels.value.find { it.id == id }?.let { m ->
                val updated = m.copy(description = note)
                repo.update(updated)
                _snackbar.emit("备注已保存")
            } ?: run {
                _snackbar.emit("更新失败：模型不存在")
            }
        }
    }

    /** 从 URI 导入模型 */
    fun importModel(uri: Uri) {
        viewModelScope.launch {
            try {
                _isLoading.value = true
                val ctx = getApplication<Application>()
                val inputStream = ctx.contentResolver.openInputStream(uri)
                    ?: throw Exception("无法打开文件")

                // 从 URI 推断文件名
                val fileName = uri.lastPathSegment?.substringAfterLast("/")
                    ?: "imported_model_${System.currentTimeMillis()}.safetensors"

                // 复制到 models 目录
                val modelsDir = ctx.getExternalFilesDir("models")
                    ?: throw Exception("无法访问存储")
                if (!modelsDir.exists()) modelsDir.mkdirs()

                val destFile = java.io.File(modelsDir, fileName)
                inputStream.use { input ->
                    destFile.outputStream().use { out ->
                        input.copyTo(out)
                    }
                }

                // 推断模型类型
                val type = inferTypeFromName(fileName)
                val backend = "CPU"  // 默认 CPU，后续可自动检测
                val precision = inferPrecisionFromName(fileName)

                val entity = ModelEntity(
                    name = fileName.substringBeforeLast("."),
                    filePath = destFile.absolutePath,
                    type = type,
                    backend = backend,
                    precision = precision,
                    fileSize = destFile.length(),
                    verified = true,
                    isFavorite = false,
                    lastUsedAt = 0L,
                    description = "导入于 ${java.text.SimpleDateFormat("yyyy-MM-dd HH:mm", java.util.Locale.US).format(java.util.Date())}",
                    importedAt = System.currentTimeMillis()
                )
                repo.save(entity)
                _snackbar.emit("导入成功: ${entity.name}")
            } catch (e: Exception) {
                _snackbar.emit("导入失败: ${e.message}")
            } finally {
                _isLoading.value = false
            }
        }
    }

    /** 扫描 models 文件夹 */
    fun scanModelsFolder() {
        viewModelScope.launch {
            try {
                _isLoading.value = true
                val ctx = getApplication<Application>()
                val modelsDir = ctx.getExternalFilesDir("models") ?: return@launch
                if (!modelsDir.exists()) modelsDir.mkdirs()

                val extensions = setOf("safetensors", "ckpt", "pt", "pth", "onnx", "mnn", "ncnn")
                val existingPaths = smartSortedModels.value.map { it.filePath }.toSet()

                var count = 0
                modelsDir.walk().filter { it.isFile && it.extension.lowercase() in extensions }
                    .forEach { file ->
                        if (file.absolutePath !in existingPaths) {
                            val type = inferTypeFromName(file.name)
                            val entity = ModelEntity(
                                name = file.name.substringBeforeLast("."),
                                filePath = file.absolutePath,
                                type = type,
                                backend = "CPU",
                                precision = inferPrecisionFromName(file.name),
                                fileSize = file.length(),
                                verified = true,
                                description = "自动扫描发现",
                                importedAt = System.currentTimeMillis()
                            )
                            repo.save(entity)
                            count++
                        }
                    }
                _snackbar.emit(if (count > 0) "发现并导入 $count 个新模型" else "没有发现新模型")
            } catch (e: Exception) {
                _snackbar.emit("扫描失败: ${e.message}")
            } finally {
                _isLoading.value = false
            }
        }
    }

    // ========== 私有工具 ==========

    private fun inferTypeFromName(name: String): String {
        val n = name.lowercase()
        return when {
            "sdxl" in n || "xl" in n -> "SDXL"
            "lcm" in n -> "LCM"
            "pixart" in n -> "PixArt"
            "kolors" in n -> "Kolors"
            "2.1" in n || "v2" in n -> "SD2.1"
            else -> "SD1.5"
        }
    }

    private fun inferPrecisionFromName(name: String): String {
        val n = name.lowercase()
        return when {
            "int8" in n -> "INT8"
            "int4" in n -> "INT4"
            "fp16" in n || "16bit" in n -> "FP16"
            else -> "FP32"
        }
    }
}

// =====================================================================
//  ViewModel Factory
// =====================================================================

class ModelsViewModelFactory(
    private val app: LocalAIPainterApp,
    private val repo: com.localaipainter.data.repository.ModelRepository
) : ViewModelProvider.Factory {
    @Suppress("UNCHECKED_CAST")
    override fun <T : ViewModel> create(modelClass: Class<T>): T {
        return ModelsViewModel(app, repo) as T
    }
}

package com.localaipainter.ui.lora

import android.app.Application
import androidx.lifecycle.*
import com.localaipainter.App
import com.localaipainter.core.PluginRegistry
import com.localaipainter.data.entity.LoraEntity
import com.localaipainter.data.repository.LoraRepository
import com.localaipainter.util.Logger
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.*
import kotlinx.coroutines.launch

/**
 * 🎯 LoRA 管理 ViewModel v2.0
 *
 * 功能：导入 / 验证 / 启用 / 删除 / 权重调节
 * 扩展：新增 LoRA 类型只需在 ModelFormatProvider 注册。
 */
class LoraViewModel(
    app: Application = App.instance,
    private val loraRepo: LoraRepository = LoraRepository(app.database.loraDao())
) : AndroidViewModel(app) {

    // ---- 列表 ----
    val allLoras: StateFlow<List<LoraEntity>> =
        loraRepo.allLoras
            .stateIn(viewModelScope, SharingStarted.Lazily, emptyList())

    // ---- 选中的 LoRA ----
    private val _selectedLoras = MutableStateFlow<Map<Long, Float>>(emptyMap())
    val selectedLoras: StateFlow<Map<Long, Float>> = _selectedLoras.asStateFlow()

    // ---- 导入状态 ----
    private val _importing = MutableStateFlow(false)
    val importing: StateFlow<Boolean> = _importing.asStateFlow()

    private val _importResult = MutableStateFlow<ImportResult?>(null)
    val importResult: StateFlow<ImportResult?> = _importResult.asStateFlow()

    // ---- 验证状态 ----
    private val _verifyingId = MutableStateFlow<Long?>(null)
    val verifyingId: StateFlow<Long?> = _verifyingId.asStateFlow()

    // ---- 搜索 ----
    private val _searchQuery = MutableStateFlow("")
    val searchQuery: StateFlow<String> = _searchQuery.asStateFlow()

    val filteredLoras: StateFlow<List<LoraEntity>> =
        combine(allLoras, _searchQuery) { list, q ->
            if (q.isBlank()) list
            else list.filter { it.name.contains(q, ignoreCase = true) || it.triggerWords.contains(q, ignoreCase = true) }
        }.stateIn(viewModelScope, SharingStarted.Lazily, emptyList())

    // ============ 操作 ============

    fun importLora(
        name: String, filePath: String,
        triggerWords: String = "", scale: Float = 0.8f
    ) {
        viewModelScope.launch(Dispatchers.IO) {
            _importing.value = true
            _importResult.value = null
            try {
                val f = java.io.File(filePath)
                if (!f.exists()) throw Exception("文件不存在: $filePath")

                val ext = f.extension.lowercase()
                val format = PluginRegistry.getModelFormat(ext)
                if (format == null) throw Exception("不支持的 LoRA 格式: .$ext")

                val lora = LoraEntity(
                    name = name.ifBlank { f.nameWithoutExtension },
                    filePath = filePath,
                    triggerWords = triggerWords,
                    fileSize = f.length(),
                    verified = false
                )
                val id = loraRepo.save(lora)
                Logger.i("LoraVM", "导入 LoRA #$id: ${lora.name} (${f.length()/1024}KB)")

                // 自动验证
                verifyLora(id, filePath)
            } catch (e: Exception) {
                _importResult.value = ImportResult(false, e.message ?: "导入失败")
                Logger.e("LoraVM", "导入失败", e)
            } finally {
                _importing.value = false
            }
        }
    }

    fun verifyLora(loraId: Long, filePath: String) {
        viewModelScope.launch(Dispatchers.IO) {
            _verifyingId.value = loraId
            try {
                // 简化验证：检查文件有效性
                val f = java.io.File(filePath)
                val valid = f.exists() && f.length() > 1024
                val current = loraRepo.getById(loraId)
                current?.let {
                    loraRepo.update(it.copy(verified = valid))
                }
                _importResult.value = if (valid) {
                    ImportResult(true, "✅ LoRA 验证通过")
                } else {
                    ImportResult(false, "❌ LoRA 文件无效")
                }
                Logger.i("LoraVM", "LoRA $loraId 验证: $valid")
            } catch (e: Exception) {
                _importResult.value = ImportResult(false, e.message ?: "验证失败")
            } finally {
                _verifyingId.value = null
            }
        }
    }

    fun deleteLora(loraId: Long) {
        viewModelScope.launch(Dispatchers.IO) {
            val lora = loraRepo.getById(loraId)
            lora?.let {
                try { java.io.File(it.filePath).delete() } catch (_: Throwable) {}
                loraRepo.deleteById(loraId)
                // 从选中移除
                val current = _selectedLoras.value.toMutableMap()
                current.remove(loraId)
                _selectedLoras.value = current
                Logger.i("LoraVM", "删除 LoRA: ${it.name}")
            }
        }
    }

    fun toggleLora(id: Long, defaultScale: Float = 0.8f) {
        val current = _selectedLoras.value.toMutableMap()
        if (current.containsKey(id)) current.remove(id)
        else current[id] = defaultScale.coerceIn(-2f, 2f)
        _selectedLoras.value = current
        Logger.d("LoraVM", "切换 LoRA $id: ${current.containsKey(id)}")
    }

    fun setLoraScale(id: Long, scale: Float) {
        val current = _selectedLoras.value.toMutableMap()
        if (current.containsKey(id)) {
            current[id] = scale.coerceIn(-2f, 2f)
            _selectedLoras.value = current
        }
    }

    fun selectAll() {
        val all = allLoras.value.associate { it.id to 0.8f }
        _selectedLoras.value = all
    }

    fun clearSelection() { _selectedLoras.value = emptyMap() }

    fun setSearchQuery(q: String) { _searchQuery.value = q }
    fun clearImportResult() { _importResult.value = null }

    // ============ 查询 ============

    fun getById(id: Long): LoraEntity? = loraRepo.getById(id)

    fun getSelectedAsMap(): Map<Long, Float> = _selectedLoras.value

    data class ImportResult(val success: Boolean, val message: String)
}

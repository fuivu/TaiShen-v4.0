package com.localaipainter.ui.gallery

import android.app.Application
import androidx.lifecycle.*
import com.localaipainter.data.AppDatabase
import com.localaipainter.data.dao.GenerationHistoryDao
import kotlinx.coroutines.flow.*
import kotlinx.coroutines.launch

class GalleryViewModel(application: Application) : AndroidViewModel(application) {

    private val historyDao = AppDatabase.getInstance(application).historyDao()

    enum class FilterMode { ALL, FAVORITES, RECENT }

    private val _filter = MutableStateFlow(FilterMode.ALL)
    val filter: StateFlow<FilterMode> = _filter

    private val _searchQuery = MutableStateFlow("")
    val searchQuery: StateFlow<String> = _searchQuery

    private val _selectedIds = MutableStateFlow<Set<Long>>(emptySet())
    val selectedIds: StateFlow<Set<Long>> = _selectedIds

    val allHistory: StateFlow<List<GenerationHistory>> =
        historyDao.getAll()
            .asFlow()
            .stateIn(viewModelScope, SharingStarted.Eagerly, emptyList())

    val favoriteHistory: StateFlow<List<GenerationHistory>> =
        historyDao.getFavorites()
            .asFlow()
            .stateIn(viewModelScope, SharingStarted.Eagerly, emptyList())

    val displayedHistory: StateFlow<List<GenerationHistory>> =
        combine(allHistory, favoriteHistory, _filter, _searchQuery) { all, fav, filter, query ->
            val base = when (filter) {
                FilterMode.ALL -> all
                FilterMode.FAVORITES -> fav
                FilterMode.RECENT -> all.take(20)
            }
            if (query.isBlank()) base else base.filter {
                it.prompt.contains(query, ignoreCase = true) ||
                it.modelName.contains(query, ignoreCase = true)
            }
        }.stateIn(viewModelScope, SharingStarted.Eagerly, emptyList())

    // ============ Actions ============

    fun setFilter(mode: FilterMode) { _filter.value = mode }
    fun setSearchQuery(query: String) { _searchQuery.value = query }

    fun toggleFavorite(id: Long) {
        viewModelScope.launch {
            val item = historyDao.getById(id) ?: return@launch
            historyDao.setFavorite(id, !item.isFavorite)
        }
    }

    fun toggleSelection(id: Long) {
        val current = _selectedIds.value.toMutableSet()
        if (current.contains(id)) current.remove(id) else current.add(id)
        _selectedIds.value = current
    }

    fun clearSelection() { _selectedIds.value = emptySet() }

    fun deleteSelected() {
        viewModelScope.launch {
            val ids = _selectedIds.value.toList()
            if (ids.isNotEmpty()) {
                historyDao.deleteBatch(ids)
                // 同时删除文件
                val all = allHistory.value
                ids.forEach { id ->
                    all.find { it.id == id }?.let { item ->
                        java.io.File(item.outputPath).delete()
                    }
                }
            }
            _selectedIds.value = emptySet()
        }
    }

    fun deleteItem(id: Long) {
        viewModelScope.launch {
            historyDao.getById(id)?.let { item ->
                java.io.File(item.outputPath).delete()
                historyDao.delete(item)
            }
        }
    }
}

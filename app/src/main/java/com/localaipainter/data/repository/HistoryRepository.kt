package com.localaipainter.data.repository

import com.localaipainter.data.dao.HistoryDao
import com.localaipainter.data.entity.HistoryEntity

class HistoryRepository(private val dao: HistoryDao) {
    val allHistory = dao.getAllLive()
    suspend fun save(h: HistoryEntity): Long = dao.insert(h)
    suspend fun delete(id: Long) = dao.deleteById(id)
    suspend fun clear() = dao.clearAll()
}

package com.localaipainter.data.repository

import com.localaipainter.data.dao.GenerationConfigDao
import com.localaipainter.data.entity.GenerationConfig

class ConfigRepository(private val dao: GenerationConfigDao) {
    val allConfigs = dao.getAllLive()
    suspend fun save(c: GenerationConfig): Long = dao.insert(c)
    suspend fun update(c: GenerationConfig) = dao.update(c)
    suspend fun delete(c: GenerationConfig) = dao.delete(c)
    suspend fun getById(id: Long): GenerationConfig? = dao.getById(id)
}

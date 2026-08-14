package com.localaipainter.data.repository

import com.localaipainter.data.dao.LoraDao
import com.localaipainter.data.entity.LoraEntity

class LoraRepository(private val dao: LoraDao) {
    val allLoras = dao.getAllLive()
    suspend fun save(l: LoraEntity): Long = dao.insert(l)
    suspend fun delete(id: Long) = dao.deleteById(id)
    suspend fun getById(id: Long): LoraEntity? = dao.getById(id)
}

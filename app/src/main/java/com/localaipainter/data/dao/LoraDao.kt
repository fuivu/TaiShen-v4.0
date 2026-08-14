package com.localaipainter.data.dao

import androidx.lifecycle.LiveData
import androidx.room.*
import com.localaipainter.data.entity.LoraEntity

@Dao
interface LoraDao {
    @Insert suspend fun insert(l: LoraEntity): Long
    @Query("SELECT * FROM loras ORDER BY importedAt DESC") fun getAllLive(): LiveData<List<LoraEntity>>
    @Query("SELECT * FROM loras WHERE id=:id") suspend fun getById(id: Long): LoraEntity?
    @Delete suspend fun delete(l: LoraEntity)
    @Query("DELETE FROM loras WHERE id=:id") suspend fun deleteById(id: Long)
}

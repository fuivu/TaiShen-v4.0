package com.localaipainter.data.dao

import androidx.lifecycle.LiveData
import androidx.room.*
import com.localaipainter.data.entity.ModelEntity

@Dao
interface ModelDao {

    // ========== 基础 CRUD ==========

    @Insert suspend fun insert(m: ModelEntity): Long
    @Update suspend fun update(m: ModelEntity)
    @Delete suspend fun delete(m: ModelEntity)
    @Query("DELETE FROM models WHERE id=:id") suspend fun deleteById(id: Long)
    @Query("SELECT * FROM models WHERE id=:id") suspend fun getById(id: Long): ModelEntity?

    // ========== 查询 ==========

    /** 全部模型，按导入时间倒序 */
    @Query("SELECT * FROM models ORDER BY importedAt DESC")
    fun getAllLive(): LiveData<List<ModelEntity>>

    /** 按类型筛选 */
    @Query("SELECT * FROM models WHERE type=:t ORDER BY importedAt DESC")
    fun getByTypeLive(t: String): LiveData<List<ModelEntity>>

    /**
     * ★ 收藏模型列表（置顶展示）
     * 收藏模型按最后使用时间倒序，未使用的排在前面
     */
    @Query("SELECT * FROM models WHERE isFavorite=1 ORDER BY lastUsedAt DESC, importedAt DESC")
    fun getFavoritesLive(): LiveData<List<ModelEntity>>

    /**
     * ★ 智能排序：收藏置顶 + 最近使用优先
     * 用于主列表展示
     */
    @Query("""
        SELECT * FROM models
        ORDER BY isFavorite DESC, lastUsedAt DESC, importedAt DESC
    """)
    fun getAllSmartSortedLive(): LiveData<List<ModelEntity>>

    /** 按类型和后端查找第一个匹配 */
    @Query("SELECT * FROM models WHERE type=:t AND backend=:b LIMIT 1")
    suspend fun findOne(t: String, b: String): ModelEntity?

    // ========== 收藏操作 ==========

    /** 切换收藏状态，返回切换后的状态 */
    @Query("UPDATE models SET isFavorite = NOT isFavorite WHERE id=:id")
    suspend fun toggleFavorite(id: Long)

    /** 设为收藏 */
    @Query("UPDATE models SET isFavorite=1 WHERE id=:id")
    suspend fun setFavorite(id: Long)

    /** 取消收藏 */
    @Query("UPDATE models SET isFavorite=0 WHERE id=:id")
    suspend fun unsetFavorite(id: Long)

    // ========== 使用记录 ==========

    /** 更新最后使用时间 */
    @Query("UPDATE models SET lastUsedAt=:ts WHERE id=:id")
    suspend fun updateLastUsed(id: Long, ts: Long = System.currentTimeMillis())

    // ========== 批量操作 ==========

    @Query("DELETE FROM models WHERE type=:t")
    suspend fun deleteByType(t: String)

    @Query("DELETE FROM models")
    suspend fun clearAll()
}

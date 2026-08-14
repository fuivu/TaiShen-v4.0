package com.localaipainter.data.repository

import com.localaipainter.data.dao.ModelDao
import com.localaipainter.data.entity.ModelEntity
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.map

/**
 * 模型仓库 —— 封装所有模型相关的数据操作
 *
 * 太神架构 v4.0 功能：
 *   - 收藏管理（toggleFavorite / getFavorites）
 *   - 智能排序（收藏置顶 + 最近使用）
 *   - 安全删除（先校验再删除文件）
 *   - 最近使用记录
 */
class ModelRepository(private val dao: ModelDao) {

    // ========== 查询 ==========

    /** 全部模型（兼容旧调用） */
    val allModels = dao.getAllLive()

    /** ★ 智能排序：收藏置顶 + 最近使用优先 + 导入时间 */
    val smartSortedModels = dao.getAllSmartSortedLive()

    /** 仅收藏列表 */
    val favoriteModels = dao.getFavoritesLive()

    /** 按类型筛选 */
    fun getByType(t: String) = dao.getByTypeLive(t)

    // ========== CRUD ==========

    suspend fun save(m: ModelEntity): Long = dao.insert(m)

    suspend fun update(m: ModelEntity) = dao.update(m)

    suspend fun findOne(t: String, b: String): ModelEntity? = dao.findOne(t, b)

    // ========== ★ 收藏操作 ==========

    /**
     * 切换收藏状态
     * @return 切换后的收藏状态（true=已收藏）
     */
    suspend fun toggleFavorite(id: Long): Boolean {
        val model = dao.getById(id) ?: return false
        if (model.isFavorite) {
            dao.unsetFavorite(id)
        } else {
            dao.setFavorite(id)
        }
        return !model.isFavorite
    }

    suspend fun setFavorite(id: Long) = dao.setFavorite(id)

    suspend fun unsetFavorite(id: Long) = dao.unsetFavorite(id)

    // ========== ★ 安全删除 ==========

    /**
     * 删除模型：从数据库移除 + 尝试删除磁盘文件
     * @return true=数据库已删除（文件可能残留）
     */
    suspend fun deleteModel(id: Long): Boolean {
        val model = dao.getById(id) ?: return false
        // 先删数据库记录
        dao.deleteById(id)
        // 再尝试删除磁盘文件
        try {
            val f = java.io.File(model.filePath)
            if (f.exists()) f.delete()
        } catch (e: Exception) {
            // 文件删除失败不影响数据库记录已删除的事实
        }
        return true
    }

    /**
     * 批量删除
     */
    suspend fun deleteModels(ids: List<Long>): Int {
        var count = 0
        ids.forEach { id ->
            if (deleteModel(id)) count++
        }
        return count
    }

    // ========== 最近使用 ==========

    suspend fun markUsed(id: Long) = dao.updateLastUsed(id, System.currentTimeMillis())

    // ========== 统计 ==========

    /** 收藏数量 Flow */
    val favoriteCount: Flow<Int> = favoriteModels.map { it.size }
}

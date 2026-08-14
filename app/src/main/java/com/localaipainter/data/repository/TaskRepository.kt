package com.localaipainter.data.repository

import com.localaipainter.data.dao.TaskDao
import com.localaipainter.data.entity.TaskEntity

class TaskRepository(private val dao: TaskDao) {
    val allTasks = dao.getAllLive()
    suspend fun enqueue(t: TaskEntity): Long = dao.insert(t)
    suspend fun update(t: TaskEntity) = dao.update(t)
    suspend fun getPending(): List<TaskEntity> = dao.getPending()
    suspend fun getRunning(): TaskEntity? = dao.getRunning()
    suspend fun cancel(id: Long) = dao.cancelById(id)
    suspend fun clearFinished() = dao.clearFinished()
}

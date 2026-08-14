package com.localaipainter.data.dao

import androidx.lifecycle.LiveData
import androidx.room.*
import com.localaipainter.data.entity.TaskEntity

@Dao
interface TaskDao {
    @Insert suspend fun insert(t: TaskEntity): Long
    @Update suspend fun update(t: TaskEntity)
    @Query("SELECT * FROM tasks WHERE status='PENDING' ORDER BY createdAt ASC") suspend fun getPending(): List<TaskEntity>
    @Query("SELECT * FROM tasks WHERE status='RUNNING' LIMIT 1") suspend fun getRunning(): TaskEntity?
    @Query("SELECT * FROM tasks ORDER BY createdAt DESC") fun getAllLive(): LiveData<List<TaskEntity>>
    @Query("UPDATE tasks SET status='CANCELLED' WHERE id=:id") suspend fun cancelById(id: Long)
    @Query("DELETE FROM tasks WHERE status IN ('DONE','FAILED','CANCELLED')") suspend fun clearFinished()
}

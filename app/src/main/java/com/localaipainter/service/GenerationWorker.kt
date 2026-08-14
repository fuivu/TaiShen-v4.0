package com.localaipainter.service

import android.content.Context
import android.content.Intent
import androidx.work.*
import com.localaipainter.App
import com.localaipainter.data.entity.TaskEntity
import kotlinx.coroutines.flow.first

/**
 * WorkManager Worker - 系统重启后自动恢复未完成任务
 */
class GenerationWorker(
    context: Context,
    params: WorkerParameters
) : CoroutineWorker(context, params) {

    override suspend fun doWork(): Result {
        val app = applicationContext as App
        val taskId = inputData.getString("task_id") ?: return Result.failure()

        // 启动前台服务处理
        val intent = Intent(applicationContext, GenerationForegroundService::class.java)
        intent.putExtra("task_id", taskId)
        applicationContext.startForegroundService(intent)

        return Result.success()
    }

    companion object {
        fun enqueue(context: Context, task: TaskEntity) {
            val request = OneTimeWorkRequestBuilder<GenerationWorker>()
                .setInputData(workDataOf("task_id" to task.id))
                .setConstraints(
                    Constraints.Builder()
                        .setRequiredNetworkType(NetworkType.NOT_REQUIRED)
                        .setRequiresBatteryNotLow(false)
                        .build()
                )
                .build()

            WorkManager.getInstance(context).enqueueUniqueWork(
                "gen_${task.id}",
                ExistingWorkPolicy.REPLACE,
                request
            )
        }
    }
}

package com.localaipainter.service

import android.app.*
import android.content.Intent
import android.os.*
import androidx.core.app.NotificationCompat
import com.localaipainter.R
import com.localaipainter.App
import com.localaipainter.data.entity.TaskEntity
import kotlinx.coroutines.*
import kotlinx.coroutines.flow.first

/**
 * 前台服务 - 保证生成过程不被系统杀死
 */
class GenerationForegroundService : Service() {

    private val serviceScope = CoroutineScope(SupervisorJob() + Dispatchers.Default)
    private lateinit var app: App

    override fun onCreate() {
        super.onCreate()
        app = application as App
        startForeground(NOTIFICATION_ID, createNotification("准备生成...", 0, 100))
    }

    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        val taskId = intent?.getStringExtra("task_id") ?: return START_NOT_STICKY

        serviceScope.launch {
            val task = app.database.taskDao().getPendingTasks()
                .firstOrNull { it.id == taskId } ?: return@launch

            try {
                app.database.taskDao().updateProgress(taskId, "running", 0)

                // 获取引擎
                val engine = app.engineFactory.getCurrentEngine()
                    ?: error("No engine available")

                // 获取模型
                val model = app.database.modelDao().getById(task.modelId)
                    ?: error("Model not found: ${task.modelId}")

                // 初始化引擎
                engine.initialize(model).onFailure { throw it }

                // 创建调度器
                val scheduler = app.schedulerFactory.create(task.scheduler)

                // 构建管线
                val pipeline = com.localaipainter.pipeline.SDPipeline(engine, app.memoryManager)
                pipeline.load(model).onFailure { throw it }

                // 解析 LoRA
                val loras = if (task.loras.isNotEmpty()) {
                    task.loras.split(",").map { loraName ->
                        com.localaipainter.lora.LoraWeight(name = loraName, path = "")
                    }
                } else emptyList()

                // 执行生成
                val result = pipeline.generate(
                    prompt = task.prompt,
                    negativePrompt = task.negativePrompt,
                    scheduler = scheduler,
                    steps = task.steps,
                    cfgScale = task.cfgScale,
                    seed = task.seed,
                    width = task.width,
                    height = task.height,
                    loras = loras
                ) { step, total ->
                    updateNotification("生成中... $step/$total", step, total)
                    serviceScope.launch {
                        app.database.taskDao().updateProgress(taskId, "running", (step * 100 / total))
                    }
                }

                result.onSuccess { pixels ->
                    // 保存图片
                    val path = saveImage(pixels, task.width, task.height)
                    // 记录历史
                    val history = com.localaipainter.data.entity.HistoryEntity(
                        id = java.util.UUID.randomUUID().toString(),
                        prompt = task.prompt,
                        negativePrompt = task.negativePrompt,
                        modelId = task.modelId,
                        scheduler = task.scheduler,
                        steps = task.steps,
                        cfgScale = task.cfgScale,
                        seed = task.seed,
                        width = task.width,
                        height = task.height,
                        imagePath = path,
                        engine = engine.name
                    )
                    app.database.historyDao().insert(history)
                    app.database.taskDao().updateProgress(taskId, "done", 100)
                    updateNotification("生成完成!", 100, 100)
                }.onFailure { e ->
                    app.database.taskDao().updateProgress(taskId, "failed", 0)
                    updateNotification("生成失败: ${e.message}", 0, 100)
                }

                pipeline.release()
                engine.release()

            } catch (e: Exception) {
                app.database.taskDao().updateProgress(taskId, "failed", 0)
                updateNotification("错误: ${e.message}", 0, 100)
            }

            // 检查是否有更多任务
            val nextTask = app.database.taskDao().getPendingTasks().firstOrNull()
            if (nextTask == null) {
                stopSelf()
            } else {
                // 处理下一个
                val nextIntent = Intent(this@GenerationForegroundService, GenerationForegroundService::class.java)
                nextIntent.putExtra("task_id", nextTask.id)
                startService(nextIntent)
                stopSelf()
            }
        }

        return START_STICKY
    }

    override fun onBind(intent: Intent?): IBinder? = null

    override fun onDestroy() {
        super.onDestroy()
        serviceScope.cancel()
    }

    private fun createNotification(text: String, progress: Int, max: Int): Notification {
        val channelId = "generation_service"
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
            val channel = NotificationChannel(
                channelId, "AI 生成服务", NotificationManager.IMPORTANCE_LOW
            )
            getSystemService(NotificationManager::class.java).createNotificationChannel(channel)
        }

        return NotificationCompat.Builder(this, channelId)
            .setContentTitle("Local AI Painter")
            .setContentText(text)
            .setSmallIcon(R.drawable.ic_notification)
            .setOngoing(true)
            .setProgress(max, progress, false)
            .build()
    }

    private fun updateNotification(text: String, progress: Int, max: Int) {
        val nm = getSystemService(NotificationManager::class.java)
        nm.notify(NOTIFICATION_ID, createNotification(text, progress, max))
    }

    private fun saveImage(pixels: FloatArray, w: Int, h: Int): String {
        // 实际应使用 Bitmap 保存
        val path = "${getExternalFilesDir(null)}/images/${System.currentTimeMillis()}.png"
        java.io.File(path).parentFile?.mkdirs()
        // 简化：实际要做 RGB float -> PNG 编码
        return path
    }

    companion object {
        const val NOTIFICATION_ID = 1001
    }
}

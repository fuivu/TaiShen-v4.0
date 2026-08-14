package com.localaipainter.service

import android.app.*
import android.content.Intent
import android.os.*
import androidx.core.app.NotificationCompat
import com.localaipainter.App
import com.localaipainter.data.entity.ModelEntity
import kotlinx.coroutines.*
import java.io.File
import java.net.URL

/**
 * 模型下载服务 - 从 HuggingFace 下载 ONNX 模型
 */
class ModelDownloadService : Service() {

    private val scope = CoroutineScope(SupervisorJob() + Dispatchers.IO)

    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        val modelUrl = intent?.getStringExtra("url") ?: return START_NOT_STICKY
        val modelName = intent.getStringExtra("name") ?: "model"
        val modelId = intent.getStringExtra("model_id") ?: return START_NOT_STICKY

        startForeground(NOTIFICATION_ID, createNotification("准备下载 $modelName...", 0, 100))

        scope.launch {
            try {
                val destDir = File(getExternalFilesDir("models"), modelId)
                destDir.mkdirs()

                // 下载主模型文件
                val destFile = File(destDir, "model.onnx")
                downloadFile(modelUrl, destFile) { progress ->
                    updateNotification("下载中... ${progress}%", progress, 100)
                }

                // 获取文件大小
                val sizeMB = destFile.length() / (1024 * 1024)

                // 保存到数据库
                val app = application as App
                val model = ModelEntity(
                    id = modelId,
                    name = modelName,
                    path = destFile.absolutePath,
                    format = "ONNX",
                    sizeMB = sizeMB,
                    isBaseModel = true
                )
                app.database.modelDao().insert(model)

                updateNotification("下载完成: $modelName", 100, 100)
                stopSelf()
            } catch (e: Exception) {
                updateNotification("下载失败: ${e.message}", 0, 100)
                stopSelf()
            }
        }

        return START_STICKY
    }

    private suspend fun downloadFile(url: String, dest: File, onProgress: (Int) -> Unit) {
        withContext(Dispatchers.IO) {
            val connection = URL(url).openConnection()
            val total = connection.contentLengthLong
            connection.getInputStream().use { input ->
                dest.outputStream().use { output ->
                    val buffer = ByteArray(8192)
                    var downloaded = 0L
                    var read: Int
                    while (input.read(buffer).also { read = it } != -1) {
                        output.write(buffer, 0, read)
                        downloaded += read
                        if (total > 0) {
                            onProgress((downloaded * 100 / total).toInt())
                        }
                    }
                }
            }
        }
    }

    override fun onBind(intent: Intent?): IBinder? = null

    private fun createNotification(text: String, progress: Int, max: Int): Notification {
        val channelId = "model_download"
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
            val channel = NotificationChannel(channelId, "模型下载", NotificationManager.IMPORTANCE_LOW)
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

    companion object {
        const val NOTIFICATION_ID = 2001
    }
}

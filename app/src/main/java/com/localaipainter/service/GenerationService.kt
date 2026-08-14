package com.localaipainter.service

import android.app.*
import android.content.Intent
import android.os.*
import androidx.core.app.NotificationCompat
import com.localaipainter.LocalAIPainterApp
import com.localaipainter.R
import com.localaipainter.engine.EngineFactory
import com.localaipainter.engine.GenerationConfig
import kotlinx.coroutines.*
import java.io.File

/**
 * 前台生成服务 - 保证后台生成不被系统杀死
 */
class GenerationService : Service() {

    private val serviceScope = CoroutineScope(SupervisorJob() + Dispatchers.Default)
    private lateinit var engine: EngineFactory
    private lateinit var notificationManager: NotificationManager

    companion object {
        const val CHANNEL_ID = "generation_channel"
        const val NOTIFICATION_ID = 1001
        const val EXTRA_PROMPT = "prompt"
        const val EXTRA_NEGATIVE = "negative"
        const val EXTRA_CONFIG = "config"
        const val EXTRA_OUTPUT = "output"
    }

    override fun onCreate() {
        super.onCreate()
        engine = EngineFactory(this)
        notificationManager = getSystemService(NOTIFICATION_SERVICE) as NotificationManager
        createNotificationChannel()
        startForeground(NOTIFICATION_ID, buildNotification("准备中...", 0, 100))
    }

    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        if (intent == null) return START_NOT_STICKY

        val prompt = intent.getStringExtra(EXTRA_PROMPT) ?: ""
        val negative = intent.getStringExtra(EXTRA_NEGATIVE) ?: ""
        val outputPath = intent.getStringExtra(EXTRA_OUTPUT) ?: return START_NOT_STICKY

        // 从 intent 重建 config (简化版)
        val config = GenerationConfig(prompt = prompt, negativePrompt = negative)

        serviceScope.launch {
            updateNotification("正在生成...", 0, config.steps)
            val result = engine.generateWithConfig(config, outputPath)
            if (result) {
                updateNotification("✅ 生成完成!", config.steps, config.steps)
                // 发送广播通知 UI 刷新
                val broadcast = Intent("com.localaipainter.GENERATION_DONE").apply {
                    putExtra("output_path", outputPath)
                }
                sendBroadcast(broadcast)
            } else {
                updateNotification("❌ 生成失败", 0, 1)
            }
            delay(3000)
            stopSelf()
        }

        return START_NOT_STICKY
    }

    override fun onDestroy() {
        super.onDestroy()
        serviceScope.cancel()
        engine.shutdown()
    }

    override fun onBind(intent: Intent?): IBinder? = null

    private fun createNotificationChannel() {
        val channel = NotificationChannel(
            CHANNEL_ID,
            "AI 生成服务",
            NotificationManager.IMPORTANCE_LOW
        ).apply {
            description = "后台执行图片生成任务"
            setShowBadge(false)
        }
        notificationManager.createNotificationChannel(channel)
    }

    private fun buildNotification(text: String, progress: Int, max: Int): Notification {
        return NotificationCompat.Builder(this, CHANNEL_ID)
            .setContentTitle("Local AI Painter")
            .setContentText(text)
            .setSmallIcon(R.drawable.ic_notification)
            .setOngoing(true)
            .setProgress(max, progress, max == 0)
            .setPriority(NotificationCompat.PRIORITY_LOW)
            .build()
    }

    private fun updateNotification(text: String, progress: Int, max: Int) {
        notificationManager.notify(
            NOTIFICATION_ID,
            buildNotification(text, progress, max)
        )
    }
}

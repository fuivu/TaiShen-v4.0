package com.localaipainter.service

import android.content.BroadcastReceiver
import android.content.Context
import android.content.Intent
import android.util.Log
import com.localaipainter.App
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch

/**
 * 开机自启 - 恢复未完成的生成任务
 */
class BootCompletedReceiver : BroadcastReceiver() {

    override fun onReceive(context: Context, intent: Intent) {
        if (intent.action != Intent.ACTION_BOOT_COMPLETED &&
            intent.action != "android.intent.action.QUICKBOOT_POWERON") return

        Log.i(TAG, "Boot completed - checking pending tasks")

        val app = context.applicationContext as App
        CoroutineScope(Dispatchers.IO).launch {
            val pending = app.database.taskDao().getPendingTasks()
            pending.forEach { task ->
                GenerationWorker.enqueue(context, task)
            }
        }
    }

    companion object {
        private const val TAG = "BootReceiver"
    }
}

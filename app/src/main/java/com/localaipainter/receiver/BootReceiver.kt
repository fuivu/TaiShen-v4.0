package com.localaipainter.receiver

import android.content.BroadcastReceiver
import android.content.Context
import android.content.Intent
import com.localaipainter.service.GenerationService

/**
 * 开机自启 Receiver - 预热引擎
 */
class BootReceiver : BroadcastReceiver() {

    override fun onReceive(context: Context, intent: Intent) {
        if (intent.action == Intent.ACTION_BOOT_COMPLETED) {
            // 启动前台服务进行预热
            val serviceIntent = Intent(context, GenerationService::class.java).apply {
                putExtra(GenerationService.EXTRA_PROMPT, "")
                putExtra(GenerationService.EXTRA_NEGATIVE, "")
                putExtra(GenerationService.EXTRA_OUTPUT, "")
            }
            context.startForegroundService(serviceIntent)
        }
    }
}

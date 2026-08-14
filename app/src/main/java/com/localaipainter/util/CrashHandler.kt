package com.localaipainter.util

import android.content.Context
import android.os.Environment
import java.io.File
import java.io.PrintWriter
import java.io.StringWriter
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale

/**
 * 全局崩溃捕获 —— 记录堆栈到文件，下次启动可上报
 */
object CrashHandler : Thread.UncaughtExceptionHandler {
    private var context: Context? = null
    private val defaultHandler = Thread.getDefaultUncaughtExceptionHandler()
    private val sdf = SimpleDateFormat("yyyy-MM-dd_HH-mm-ss", Locale.US)

    fun init(ctx: Context) {
        context = ctx.applicationContext
        Thread.setDefaultUncaughtExceptionHandler(this)
    }

    override fun uncaughtException(t: Thread, e: Throwable) {
        try {
            val dir = File(context?.getExternalFilesDir(null), "crash")
            if (!dir.exists()) dir.mkdirs()
            val file = File(dir, "crash_${sdf.format(Date())}.log")
            PrintWriter(file).use { pw ->
                pw.println("Thread: ${t.name}")
                pw.println("Time: ${sdf.format(Date())}")
                e.printStackTrace(pw)
                val sw = StringWriter()
                e.printStackTrace(PrintWriter(sw))
                pw.println(sw.toString())
            }
            Logger.e("Uncaught exception", e)
        } catch (ignored: Exception) {
        }
        defaultHandler?.uncaughtException(t, e)
    }
}

package com.localaipainter.util

import android.content.ClipData
import android.content.ClipboardManager
import android.content.Context
import android.content.Intent
import android.net.Uri
import androidx.core.content.FileProvider
import java.io.File
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale
import java.util.zip.ZipEntry
import java.util.zip.ZipOutputStream

/**
 * 📤 日志导出工具 v2.0
 *
 * 支持：
 *   - 复制到剪贴板（短日志）
 *   - 分享单个文件（FileProvider）
 *   - 打包全部日志为 ZIP（含崩溃报告）
 *   - 一键上传（预留接口）
 */
object LogExportHelper {

    enum class ExportFormat { TEXT, ZIP }

    // ============ 剪贴板 ============

    fun copyToClipboard(context: Context, text: String): Boolean {
        return try {
            val cm = context.getSystemService(Context.CLIPBOARD_SERVICE) as ClipboardManager
            cm.setPrimaryClip(ClipData.newPlainText("Local AI Painter Logs", text))
            Logger.i("LogExport", "日志已复制到剪贴板 (${text.length} 字符)")
            true
        } catch (e: Exception) {
            Logger.e("LogExport", "复制失败", e)
            false
        }
    }

    fun copyRecentLogs(context: Context, maxLines: Int = 500): Boolean =
        copyToClipboard(context, Logger.getRecentLogs(maxLines))

    // ============ 分享 ============

    fun shareLatest(context: Context): Boolean {
        return try {
            val logDir = File(Logger.getLogDirPath())
            val latest = logDir.listFiles { f -> f.name.endsWith(".log") }
                ?.maxByOrNull { it.lastModified() } ?: return false
            shareFile(context, latest, "text/plain")
            true
        } catch (e: Exception) {
            Logger.e("LogExport", "分享失败", e)
            false
        }
    }

    fun shareCrashReport(context: Context): Boolean {
        return try {
            val logDir = File(Logger.getLogDirPath())
            val crash = logDir.listFiles { f -> f.name.startsWith("crash_") }
                ?.maxByOrNull { it.lastModified() } ?: return false
            shareFile(context, crash, "text/plain")
            true
        } catch (e: Exception) {
            Logger.e("LogExport", "分享崩溃报告失败", e)
            false
        }
    }

    private fun shareFile(context: Context, file: File, mime: String) {
        val uri = FileProvider.getUriForFile(
            context, "${context.packageName}.fileprovider", file
        )
        val intent = Intent(Intent.ACTION_SEND).apply {
            type = mime
            putExtra(Intent.EXTRA_STREAM, uri)
            putExtra(Intent.EXTRA_SUBJECT, "Local AI Painter - ${file.name}")
            addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION)
        }
        context.startActivity(Intent.createChooser(intent, "分享日志"))
    }

    // ============ 打包 ZIP ============

    fun packAllLogs(context: Context): File? {
        return try {
            val logDir = File(Logger.getLogDirPath())
            if (!logDir.exists()) return null

            val ts = SimpleDateFormat("yyyyMMdd_HHmmss", Locale.getDefault()).format(Date())
            val zipFile = File(logDir, "LocalAIPainter_Logs_$ts.zip")

            ZipOutputStream(zipFile.outputStream()).use { zos ->
                logDir.listFiles { f -> f.name.endsWith(".log") }?.forEach { f ->
                    zos.putNextEntry(ZipEntry(f.name))
                    f.inputStream().use { it.copyTo(zos) }
                    zos.closeEntry()
                }
                // 附带设备信息
                zos.putNextEntry(ZipEntry("device_info.txt"))
                zos.write(Logger.getAllLogs().toByteArray())
                zos.closeEntry()
            }

            Logger.i("LogExport", "日志已打包: ${zipFile.absolutePath} (${zipFile.length()/1024}KB)")
            zipFile
        } catch (e: Exception) {
            Logger.e("LogExport", "打包失败", e)
            null
        }
    }

    fun shareAllLogs(context: Context): Boolean {
        val zip = packAllLogs(context) ?: return false
        return try {
            shareFile(context, zip, "application/zip")
            true
        } catch (e: Exception) {
            Logger.e("LogExport", "分享 ZIP 失败", e)
            false
        }
    }

    // ============ 上传（预留接口） ============

    var uploadCallback: ((File) -> Boolean)? = null

    fun uploadLogs(context: Context): Boolean {
        val zip = packAllLogs(context) ?: return false
        return uploadCallback?.invoke(zip) ?: run {
            Logger.w("LogExport", "未设置上传回调，回退到分享")
            shareAllLogs(context)
        }
    }

    // ============ 清理 ============

    fun clearAllLogs(): Boolean {
        return try {
            val logDir = File(Logger.getLogDirPath())
            logDir.listFiles { f -> f.name.endsWith(".log") }?.forEach { it.delete() }
            Logger.clearLogs()
            true
        } catch (e: Exception) {
            Logger.e("LogExport", "清理失败", e)
            false
        }
    }
}

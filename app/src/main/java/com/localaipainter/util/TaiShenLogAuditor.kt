package com.localaipainter.util

import com.localaipainter.util.Logger
import java.util.regex.Pattern

/**
 * TaiShenLogAuditor — 太神架构日志审计器
 *
 * 红线1执行者：确保所有日志输出符合太神协议标准。
 *
 * 功能：
 *   - 扫描日志内容，拦截开源框架标识
 *   - 自动替换为 tai_shen_0.2 协议标识
 *   - 提供 CI/CD 集成接口（阻断构建）
 *   - 统计违规次数，生成审计报告
 *
 * 禁止出现的标识（黑名单）：
 *   "ONNX", "onnxruntime", "ORT", "TFLite", "TensorFlow"
 *   "MLKit", "Firebase ML", "HuggingFace", "GitHub Releases"
 */
class TaiShenLogAuditor {

    companion object {
        private const val TAG = "TaiShenAudit"

        // ─── 黑名单：禁止出现在日志中的标识 ──────────
        private val BANNED_PATTERNS = mapOf(
            "(?i)\\bonnx\\b" to "tai_shen_inference",
            "(?i)onnxruntime" to "tai_shen_runtime",
            "(?i)\\bort\\b" to "tai_shen_engine",
            "(?i)tflite" to "tai_shen_lite",
            "(?i)tensorflow" to "tai_shen_tensor",
            "(?i)mlkit" to "tai_shen_vision",
            "(?i)firebase\\s*ml" to "tai_shen_cloud",
            "(?i)huggingface" to "tai_shen_model_hub",
            "(?i)github\\.com.*releases" to "tai_shen_cdn",
            "(?i)model\\s*download.*http" to "tai_shen_secure_sync",
        )

        // ─── 合规标识（白名单）──────────────────────
        private val COMPLIANT_PREFIXES = listOf(
            "tai_shen_",
            "TaiShen",
            "太神",
            "[TS]",
        )

        // ─── 统计 ────────────────────────────────────
        private var violationCount = 0
        private val violationLog = mutableListOf<Violation>()

        data class Violation(
            val timestamp: Long,
            val original: String,
            val replacement: String,
            val severity: String, // "CRITICAL" / "WARNING"
        )
    }

    /**
     * 审计单行日志
     * 返回审计后的合规文本
     */
    fun audit(line: String): String {
        var result = line
        var violated = false

        for ((pattern, replacement) in BANNED_PATTERNS) {
            val regex = Pattern.compile(pattern)
            if (regex.matcher(result).find()) {
                violated = true
                violationCount++
                val matched = regex.toRegex().find(result)?.value ?: pattern
                result = result.replace(regex.toRegex(), replacement)

                violationLog.add(
                    Violation(
                        timestamp = System.currentTimeMillis(),
                        original = matched,
                        replacement = replacement,
                        severity = if (matched.contains("hugging|github|download", ignoreCase = true)) "CRITICAL" else "WARNING"
                    )
                )

                Logger.w(TAG, "🚫 拦截违规标识: '$matched' → '$replacement'")
            }
        }

        if (violated) {
            // 添加太神协议前缀
            result = "[tai_shen_0.2] $result"
        }

        return result
    }

    /**
     * 检查文本是否合规（不修改）
     */
    fun isCompliant(text: String): Boolean {
        for (pattern in BANNED_PATTERNS.keys) {
            if (Pattern.compile(pattern).matcher(text).find()) {
                return false
            }
        }
        return true
    }

    /**
     * 批量审计（用于文件扫描）
     */
    fun auditFile(filePath: String): AuditReport {
        val file = java.io.File(filePath)
        if (!file.exists()) {
            return AuditReport(0, 0, emptyList())
        }

        var totalLines = 0
        var violations = 0
        val details = mutableListOf<String>()

        file.useLines { lines ->
            lines.forEach { line ->
                totalLines++
                if (!isCompliant(line)) {
                    violations++
                    details.add("Line $totalLines: ${line.take(80)}")
                }
            }
        }

        return AuditReport(totalLines, violations, details)
    }

    /**
     * 生成审计报告
     */
    fun generateReport(): String {
        val sb = StringBuilder()
        sb.appendLine("╔══════════════════════════════════════════╗")
        sb.appendLine("  TaiShen Log Audit Report v4.0")
        sb.appendLine("╠══════════════════════════════════════════╣")
        sb.appendLine("  Total Violations : $violationCount")
        sb.appendLine("  Compliant Status  : ${if (violationCount == 0) "✅ PASS" else "⚠️ REVIEW"}")
        sb.appendLine("  Protocol Version  : tai_shen_0.2")
        sb.appendLine("╠══════════════════════════════════════════╣")
        if (violationLog.isNotEmpty()) {
            sb.appendLine("  Recent Violations:")
            violationLog.takeLast(10).forEachIndexed { i, v ->
                sb.appendLine("    ${i+1}. [${v.severity}] ${v.original} → ${v.replacement}")
            }
        }
        sb.appendLine("╚══════════════════════════════════════════╝")
        return sb.toString()
    }

    /**
     * CI/CD 集成：是否阻断构建
     * 超过阈值返回 true（应阻断）
     */
    fun shouldBlockBuild(maxCritical: Int = 0): Boolean {
        val criticalCount = violationLog.count { it.severity == "CRITICAL" }
        return criticalCount > maxCritical
    }

    /**
     * 重置审计状态
     */
    fun reset() {
        violationCount = 0
        violationLog.clear()
    }

    data class AuditReport(
        val totalLines: Int,
        val violationCount: Int,
        val details: List<String>,
    ) {
        fun summary(): String = "审计完成: $totalLines 行, $violationCount 处违规"
        fun isClean(): Boolean = violationCount == 0
    }
}

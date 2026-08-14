package com.localaipainter.ui.theme

import android.content.Context
import android.content.SharedPreferences
import androidx.compose.ui.graphics.Color

/**
 * 🎨 主题偏好持久化 —— 记住用户的选择
 *
 * 保存：当前预设 / 自定义色 / 暗色模式 / 动态取色开关
 * 扩展：新增主题相关设置只需加一个 key + getter/setter。
 */
class ThemePreferences(context: Context) {

    private val prefs: SharedPreferences =
        context.getSharedPreferences("theme_prefs", Context.MODE_PRIVATE)

    companion object {
        private const val KEY_PRESET = "preset"
        private const val KEY_CUSTOM_SEED = "custom_seed"
        private const val KEY_DARK_MODE = "dark_mode"
        private const val KEY_DYNAMIC_COLOR = "dynamic_color"
        private const val KEY_BG_PATH = "bg_image_path"
        private const val KEY_BG_BLUR = "bg_blur_radius"
        private const val KEY_BG_DARKEN = "bg_darken"
        private const val KEY_ACCENT_OVERRIDE = "accent_override"
    }

    // ---- 预设 ----
    var preset: String
        get() = prefs.getString(KEY_PRESET, "OBSIDIAN") ?: "OBSIDIAN"
        set(value) = prefs.edit().putString(KEY_PRESET, value).apply()

    // ---- 自定义种子色 ----
    var customSeedColor: Int
        get() = prefs.getInt(KEY_CUSTOM_SEED, 0xFF7C4DFF.toInt())
        set(value) = prefs.edit().putInt(KEY_CUSTOM_SEED, value).apply()

    // ---- 暗色模式 ----
    var darkMode: Boolean
        get() = prefs.getBoolean(KEY_DARK_MODE, true)
        set(value) = prefs.edit().putBoolean(KEY_DARK_MODE, value).apply()

    // ---- 动态取色 ----
    var dynamicColorEnabled: Boolean
        get() = prefs.getBoolean(KEY_DYNAMIC_COLOR, true)
        set(value) = prefs.edit().putBoolean(KEY_DYNAMIC_COLOR, value).apply()

    // ---- 背景图 ----
    var bgImagePath: String?
        get() = prefs.getString(KEY_BG_PATH, null)
        set(value) = prefs.edit().putString(KEY_BG_PATH, value).apply()

    var bgBlurRadius: Float
        get() = prefs.getFloat(KEY_BG_BLUR, 20f)
        set(value) = prefs.edit().putFloat(KEY_BG_BLUR, value.coerceIn(0f, 50f)).apply()

    var bgDarkenAlpha: Float
        get() = prefs.getFloat(KEY_BG_DARKEN, 0.6f)
        set(value) = prefs.edit().putFloat(KEY_BG_DARKEN, value.coerceIn(0f, 0.95f)).apply()

    // ---- 强调色覆盖 ----
    var accentOverride: Int?
        get() {
            val v = prefs.getInt(KEY_ACCENT_OVERRIDE, -1)
            return if (v == -1) null else v
        }
        set(value) {
            if (value == null) prefs.edit().remove(KEY_ACCENT_OVERRIDE).apply()
            else prefs.edit().putInt(KEY_ACCENT_OVERRIDE, value).apply()
        }

    // ---- 批量操作 ----
    fun reset() {
        prefs.edit().clear().apply()
    }

    fun exportToString(): String = buildString {
        append("preset=$preset\n")
        append("custom_seed=#${"%06X".format(customSeedColor and 0xFFFFFF)}\n")
        append("dark_mode=$darkMode\n")
        append("dynamic_color=$dynamicColorEnabled\n")
        append("bg_path=$bgImagePath\n")
        append("bg_blur=$bgBlurRadius\n")
        append("bg_darken=$bgDarkenAlpha\n")
        append("accent=${accentOverride?.let { "#${"%06X".format(it and 0xFFFFFF)}" } ?: "auto"}\n")
    }

    fun importFromString(text: String) {
        text.lines().forEach { line ->
            val parts = line.split("=", limit = 2)
            if (parts.size != 2) return@forEach
            when (parts[0].trim()) {
                "preset" -> preset = parts[1].trim()
                "dark_mode" -> darkMode = parts[1].trim().toBooleanStrictOrNull() ?: true
                "dynamic_color" -> dynamicColorEnabled = parts[1].trim().toBooleanStrictOrNull() ?: true
                "bg_blur" -> bgBlurRadius = parts[1].trim().toFloatOrNull() ?: 20f
                "bg_darken" -> bgDarkenAlpha = parts[1].trim().toFloatOrNull() ?: 0.6f
            }
        }
    }

    // ---- 转换为 Compose Color ----
    fun getCustomColor(): Color = Color(customSeedColor)
    fun getAccentColor(): Color? = accentOverride?.let { Color(it) }
}

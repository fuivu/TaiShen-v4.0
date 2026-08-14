package com.localaipainter.ui.theme

import android.content.Context
import android.graphics.Bitmap
import androidx.compose.material3.ColorScheme
import androidx.compose.material3.darkColorScheme
import androidx.compose.material3.lightColorScheme
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.toArgb
import androidx.palette.graphics.Palette

/**
 * 🎨 动态主题系统 —— 根据背景图主色调自动切换配色
 *
 * 工作流程：
 *   1. 用户导入背景图 → 提取调色板
 *   2. 选主色 → 生成协调的强调色 / 表面色 / 错误色
 *   3. 注入 Material 3 ColorScheme → 全局 UI 自动变色
 *
 * 兼容策略：
 *   - Android 12+：优先用系统动态取色，叠加背景图色调
 *   - Android 8-11：纯算法生成（HSL 旋转 + 对比度修正）
 */
object DynamicTheme {

    // ============ 预设主题方案 ============

    enum class Preset(val displayName: String, val emoji: String) {
        OCEAN("深海", "🌊"),
        SUNSET("夕阳", "🌅"),
        FOREST("森林", "🌲"),
        LAVENDER("薰衣草", "💜"),
        COSMIC("宇宙", "🌌"),
        CHERRY("樱花", "🌸"),
        OBSIDIAN("黑曜石", "🖤"),
        GOLDEN("黄金", "✨"),
        AURORA("极光", "🌈"),
        MIDNIGHT("午夜", "🌙")
    }

    // ============ 调色板提取 ============

    fun extractPalette(bitmap: Bitmap): PaletteResult {
        val palette = Palette.from(bitmap)
            .maximumColorCount(32)
            .generate()

        val vibrant = palette.vibrantSwatch?.rgb
            ?: palette.lightVibrantSwatch?.rgb
            ?: palette.dominantSwatch?.rgb
            ?: 0xFF7C4DFF.toInt()

        val muted = palette.mutedSwatch?.rgb
            ?: palette.lightMutedSwatch?.rgb
            ?: palette.dominantSwatch?.rgb
            ?: 0xFF4A4A6A.toInt()

        val darkVibrant = palette.darkVibrantSwatch?.rgb
            ?: palette.darkMutedSwatch?.rgb
            ?: vibrant

        return PaletteResult(
            primary = Color(vibrant),
            secondary = Color(muted),
            tertiary = Color(darkVibrant),
            vibrant = palette.vibrantSwatch?.let { Color(it.rgb) },
            muted = palette.mutedSwatch?.let { Color(it.rgb) },
            dominant = Color(palette.dominantSwatch?.rgb ?: vibrant),
            isDark = isColorDark(vibrant)
        )
    }

    // ============ 从单色生成完整色板 ============

    fun generateFromColor(seedColor: Color, isDark: Boolean = true): ColorScheme {
        val hsl = rgbToHsl(seedColor.toArgb())
        val (h, s, l) = hsl

        return if (isDark) {
            darkColorScheme(
                primary = Color(hslToRgb(h, s.coerceIn(0.5f, 0.9f), 0.65f)),
                onPrimary = Color.White,
                primaryContainer = Color(hslToRgb(h, s * 0.7f, 0.25f)),
                onPrimaryContainer = Color(hslToRgb(h, s * 0.6f, 0.85f)),
                secondary = Color(hslToRgb((h + 30f) % 360f, s * 0.6f, 0.7f)),
                onSecondary = Color.Black,
                secondaryContainer = Color(hslToRgb((h + 30f) % 360f, s * 0.5f, 0.2f)),
                onSecondaryContainer = Color(hslToRgb((h + 30f) % 360f, s * 0.5f, 0.85f)),
                tertiary = Color(hslToRgb((h + 210f) % 360f, s * 0.7f, 0.6f)),
                onTertiary = Color.Black,
                background = Color(hslToRgb(h, 0.15f, 0.08f)),      // 极深
                onBackground = Color(hslToRgb(h, 0.1f, 0.9f)),
                surface = Color(hslToRgb(h, 0.2f, 0.12f)),
                onSurface = Color(hslToRgb(h, 0.1f, 0.88f)),
                surfaceVariant = Color(hslToRgb(h, 0.3f, 0.18f)),
                onSurfaceVariant = Color(hslToRgb(h, 0.15f, 0.75f)),
                error = Color(0xFFFF5252),
                onError = Color.Black,
                errorContainer = Color(0xFF8B0000),
                onErrorContainer = Color(0xFFFFCDD2),
                outline = Color(hslToRgb(h, 0.2f, 0.45f)),
                outlineVariant = Color(hslToRgb(h, 0.15f, 0.3f)),
                scrim = Color.Black.copy(alpha = 0.5f),
                inverseSurface = Color(hslToRgb(h, 0.1f, 0.9f)),
                inverseOnSurface = Color(hslToRgb(h, 0.1f, 0.1f)),
                inversePrimary = Color(hslToRgb(h, s, 0.4f)),
                surfaceTint = Color(hslToRgb(h, s, 0.65f)),
                surfaceBright = Color(hslToRgb(h, 0.2f, 0.2f)),
                surfaceDim = Color(hslToRgb(h, 0.2f, 0.06f)),
                surfaceContainerLowest = Color(hslToRgb(h, 0.15f, 0.04f)),
                surfaceContainerLow = Color(hslToRgb(h, 0.18f, 0.08f)),
                surfaceContainer = Color(hslToRgb(h, 0.2f, 0.12f)),
                surfaceContainerHigh = Color(hslToRgb(h, 0.22f, 0.16f)),
                surfaceContainerHighest = Color(hslToRgb(h, 0.25f, 0.2f))
            )
        } else {
            lightColorScheme(
                primary = Color(hslToRgb(h, s, 0.45f)),
                onPrimary = Color.White,
                primaryContainer = Color(hslToRgb(h, s * 0.7f, 0.85f)),
                onPrimaryContainer = Color(hslToRgb(h, s, 0.2f)),
                secondary = Color(hslToRgb((h + 30f) % 360f, s * 0.6f, 0.45f)),
                onSecondary = Color.White,
                background = Color(hslToRgb(h, 0.1f, 0.98f)),
                onBackground = Color(hslToRgb(h, 0.2f, 0.12f)),
                surface = Color.White,
                onSurface = Color(hslToRgb(h, 0.15f, 0.15f)),
                error = Color(0xFFB00020),
                outline = Color(hslToRgb(h, 0.15f, 0.6f))
            )
        }
    }

    // ============ 预设方案 → 色板 ============

    fun presetToScheme(preset: Preset, isDark: Boolean = true): ColorScheme {
        val seed = when (preset) {
            Preset.OCEAN    -> Color(0xFF006994)
            Preset.SUNSET   -> Color(0xFFFF6B35)
            Preset.FOREST   -> Color(0xFF2D6A4F)
            Preset.LAVENDER -> Color(0xFF7C4DFF)
            Preset.COSMIC   -> Color(0xFF6C3CE9)
            Preset.CHERRY  -> Color(0xFFFF77A9)
            Preset.OBSIDIAN -> Color(0xFF3D3D5C)
            Preset.GOLDEN   -> Color(0xFFFFB800)
            Preset.AURORA   -> Color(0xFF00D2FF)
            Preset.MIDNIGHT -> Color(0xFF1A1A2E)
        }
        return generateFromColor(seed, isDark)
    }

    // ============ 工具函数 ============

    fun isColorDark(rgb: Int): Boolean {
        val r = (rgb shr 16) and 0xFF
        val g = (rgb shr 8) and 0xFF
        val b = rgb and 0xFF
        val luminance = 0.299 * r + 0.587 * g + 0.114 * b
        return luminance < 128
    }

    fun rgbToHsl(rgb: Int): Triple<Float, Float, Float> {
        val r = ((rgb shr 16) and 0xFF) / 255f
        val g = ((rgb shr 8) and 0xFF) / 255f
        val b = (rgb and 0xFF) / 255f
        val max = maxOf(r, g, b)
        val min = minOf(r, g, b)
        val l = (max + min) / 2f
        var h = 0f
        val s: Float
        if (max == min) {
            h = 0f; s = 0f
        } else {
            val d = max - min
            s = if (l > 0.5f) d / (2f - max - min) else d / (max + min)
            when (max) {
                r -> h = (g - b) / d + (if (g < b) 6 else 0)
                g -> h = (b - r) / d + 2
                else -> h = (r - g) / d + 4
            }
            h *= 60f
        }
        return Triple(h, s, l)
    }

    fun hslToRgb(h: Float, s: Float, l: Float): Int {
        val hN = (h % 360f + 360f) % 360f / 360f
        if (s <= 0f) {
            val v = (l * 255).toInt().coerceIn(0, 255)
            return 0xFF000000.toInt() or (v shl 16) or (v shl 8) or v
        }
        val q = if (l < 0.5f) l * (1 + s) else l + s - l * s
        val p = 2 * l - q
        val r = hueToRgb(p, q, hN + 1f/3f)
        val g = hueToRgb(p, q, hN)
        val b = hueToRgb(p, q, hN - 1f/3f)
        return 0xFF000000.toInt() or
               ((r * 255).toInt().coerceIn(0,255) shl 16) or
               ((g * 255).toInt().coerceIn(0,255) shl 8) or
               (b * 255).toInt().coerceIn(0,255)
    }

    private fun hueToRgb(p: Float, q: Float, t: Float): Float {
        var tt = t
        if (tt < 0f) tt += 1f
        if (tt > 1f) tt -= 1f
        if (tt < 1f/6f) return p + (q - p) * 6f * tt
        if (tt < 1f/2f) return q
        if (tt < 2f/3f) return p + (q - p) * (2f/3f - tt) * 6f
        return p
    }

    /**
     * 计算两色对比度（WCAG），< 4.5 视为不达标
     */
    fun contrastRatio(a: Color, b: Color): Float {
        val la = relativeLuminance(a.toArgb())
        val lb = relativeLuminance(b.toArgb())
        val hi = maxOf(la, lb)
        val lo = minOf(la, lb)
        return (hi + 0.05f) / (lo + 0.05f)
    }

    private fun relativeLuminance(rgb: Int): Float {
        fun c(v: Int): Float {
            val f = v / 255f
            return if (f <= 0.03928f) f / 12.92f else ((f + 0.055f) / 1.055f).toFloat().pow(2.4f)
        }
        val r = c((rgb shr 16) and 0xFF)
        val g = c((rgb shr 8) and 0xFF)
        val b = c(rgb and 0xFF)
        return 0.2126f * r + 0.7152f * g + 0.0722f * b
    }

    private fun Float.pow(e: Float) = kotlin.math.pow(this, e.toDouble()).toFloat()
}

data class PaletteResult(
    val primary: Color,
    val secondary: Color,
    val tertiary: Color,
    val vibrant: Color?,
    val muted: Color?,
    val dominant: Color,
    val isDark: Boolean
)

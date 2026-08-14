package com.localaipainter.ui.theme

import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.darkColorScheme
import androidx.compose.material3.lightColorScheme
import androidx.compose.runtime.Composable
import androidx.compose.ui.graphics.Color

// ============ 深色主题（默认） ============

private val DarkColorScheme = darkColorScheme(
    primary       = Color(0xFF7C4DFF),  // 紫色
    onPrimary     = Color.White,
    secondary     = Color(0xFF00BCD4),  // 青色
    onSecondary   = Color.Black,
    tertiary      = Color(0xFFFF6E40),  // 橙红
    onTertiary    = Color.Black,
    background    = Color(0xFF0F0F1A),
    onBackground  = Color(0xFFE0E0E0),
    surface       = Color(0xFF1A1A2E),
    onSurface     = Color(0xFFE0E0E0),
    surfaceVariant= Color(0xFF252540),
    onSurfaceVariant = Color(0xFFB0B0C0),
    error         = Color(0xFFFF5252),
    onError       = Color.White,
    outline       = Color(0xFF3A3A55),
    outlineVariant= Color(0xFF2A2A45),
)

// ============ 浅色主题 ============

private val LightColorScheme = lightColorScheme(
    primary       = Color(0xFF6200EE),
    onPrimary     = Color.White,
    secondary     = Color(0xFF03DAC6),
    onSecondary   = Color.Black,
    tertiary      = Color(0xFFE65100),
    onTertiary    = Color.White,
    background    = Color(0xFFFAFAFA),
    onBackground  = Color(0xFF1A1A1A),
    surface       = Color.White,
    onSurface     = Color(0xFF1A1A1A),
    surfaceVariant= Color(0xFFF0F0F0),
    onSurfaceVariant = Color(0xFF4A4A4A),
    error         = Color(0xFFB00020),
    onError       = Color.White,
    outline       = Color(0xFFD0D0D0),
    outlineVariant= Color(0xFFE0E0E0),
)

// ============ AMOLED 纯黑 ============

private val AmoledColorScheme = darkColorScheme(
    primary       = Color(0xFFBB86FC),
    onPrimary     = Color.Black,
    secondary     = Color(0xFF03DAC6),
    onSecondary   = Color.Black,
    background    = Color.Black,
    onBackground  = Color.White,
    surface       = Color(0xFF0A0A0A),
    onSurface     = Color.White,
    surfaceVariant= Color(0xFF151515),
    onSurfaceVariant = Color(0xFFAAAAAA),
    outline       = Color(0xFF333333),
)

// ============ 日落 ============

private val SunsetColorScheme = darkColorScheme(
    primary       = Color(0xFFFF7043),
    onPrimary     = Color.White,
    secondary     = Color(0xFFFFAB40),
    onSecondary   = Color.Black,
    tertiary      = Color(0xFFE91E63),
    onTertiary    = Color.White,
    background    = Color(0xFF1A0E0A),
    onBackground  = Color(0xFFFFE0D0),
    surface       = Color(0xFF2D1B14),
    onSurface     = Color(0xFFFFE0D0),
    surfaceVariant= Color(0xFF3D2518),
    onSurfaceVariant = Color(0xFFFFB088),
    outline       = Color(0xFF5D3A28),
)

// ============ 深海 ============

private val DeepSeaColorScheme = darkColorScheme(
    primary       = Color(0xFF00ACC1),
    onPrimary     = Color.Black,
    secondary     = Color(0xFF009688),
    onSecondary   = Color.White,
    tertiary      = Color(0xFF3F51B5),
    onTertiary    = Color.White,
    background    = Color(0xFF001A1A),
    onBackground  = Color(0xFFB2EBF2),
    surface       = Color(0xFF002B2B),
    onSurface     = Color(0xFFB2EBF2),
    surfaceVariant= Color(0xFF003D3D),
    onSurfaceVariant = Color(0xFF80CBC4),
    outline       = Color(0xFF00695C),
)

// ============ 森林 ============

private val ForestColorScheme = darkColorScheme(
    primary       = Color(0xFF4CAF50),
    onPrimary     = Color.Black,
    secondary     = Color(0xFF8BC34A),
    onSecondary   = Color.Black,
    tertiary      = Color(0xFF2E7D32),
    onTertiary    = Color.White,
    background    = Color(0xFF0A1A0A),
    onBackground  = Color(0xFFC8E6C9),
    surface       = Color(0xFF1A2A1A),
    onSurface     = Color(0xFFC8E6C9),
    surfaceVariant= Color(0xFF2A3A2A),
    onSurfaceVariant = Color(0xFFA5D6A7),
    outline       = Color(0xFF4A7A4A),
)

// ============ 主题枚举 ============

enum class AppTheme(val displayName: String) {
    DARK("深色"),
    LIGHT("浅色"),
    AMOLED("AMOLED"),
    SUNSET("日落"),
    DEEP_SEA("深海"),
    FOREST("森林"),
}

@Composable
fun LocalAIPainterTheme(
    theme: AppTheme = AppTheme.DARK,
    content: @Composable () -> Unit,
) {
    val colorScheme = when (theme) {
        AppTheme.DARK    -> DarkColorScheme
        AppTheme.LIGHT   -> LightColorScheme
        AppTheme.AMOLED  -> AmoledColorScheme
        AppTheme.SUNSET  -> SunsetColorScheme
        AppTheme.DEEP_SEA-> DeepSeaColorScheme
        AppTheme.FOREST  -> ForestColorScheme
    }

    MaterialTheme(
        colorScheme = colorScheme,
        content = content,
    )
}

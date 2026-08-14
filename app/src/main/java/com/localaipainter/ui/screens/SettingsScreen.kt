package com.localaipainter.ui.screens

import androidx.compose.foundation.*
import androidx.compose.foundation.layout.*
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.localaipainter.ui.theme.AppTheme

/**
 * 设置页 — 主题 / 性能 / 高级
 */
@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun SettingsScreen(
    currentTheme: AppTheme,
    onThemeChange: (AppTheme) -> Unit,
    deviceInfo: String = "Detecting...",
    gpuCacheEnabled: Boolean,
    onGpuCacheChange: (Boolean) -> Unit,
    gpuQuantEnabled: Boolean,
    onGpuQuantChange: (Boolean) -> Unit,
    threadCount: Int,
    onThreadsChange: (Int) -> Unit,
    memoryLimitMB: Int,
    onMemoryLimitChange: (Int) -> Unit,
    autoStartEnabled: Boolean,
    onAutoStartChange: (Boolean) -> Unit,
    keepScreenOnEnabled: Boolean,
    onKeepScreenOnChange: (Boolean) -> Unit,
    onExportLog: () -> Unit,
    onClearCache: () -> Unit,
) {
    Column(
        modifier = Modifier
            .fillMaxSize()
            .verticalScroll(rememberScrollState())
            .padding(16.dp),
    ) {
        Text("⚙️ 设置", fontSize = 24.sp, fontWeight = FontWeight.Bold)
        Spacer(Modifier.height(4.dp))
        Text(deviceInfo, fontSize = 12.sp, color = MaterialTheme.colorScheme.onSurfaceVariant)
        Spacer(Modifier.height(20.dp))

        // ===== 主题 =====
        SettingsSection("外观")
        ThemeSelector(currentTheme, onThemeChange)
        Spacer(Modifier.height(20.dp))

        // ===== 性能 =====
        SettingsSection("性能")
        SwitchRow("GPU 缓存优化", gpuCacheEnabled, onGpuCacheChange, "加速推理，多消耗约 200MB 显存")
        SwitchRow("GPU 量化推理", gpuQuantEnabled, onGpuQuantChange, "INT8 量化，速度提升 2-3 倍")
        SliderRow("线程数", threadCount.toFloat(), 1f..8f, 7, "${threadCount}", onThreadsChange)
        SliderRow("内存上限 (MB)", memoryLimitMB.toFloat(), 512f..8192f, 15, "${memoryLimitMB}", onMemoryLimitChange)
        Spacer(Modifier.height(20.dp))

        // ===== 系统 =====
        SettingsSection("系统")
        SwitchRow("开机自启", autoStartEnabled, onAutoStartChange, "后台预热引擎，缩短首次生成等待")
        SwitchRow("保持屏幕常亮", keepScreenOnEnabled, onKeepScreenOnChange, "生成时防止锁屏")
        Spacer(Modifier.height(20.dp))

        // ===== 数据 =====
        SettingsSection("数据与日志")
        Button(
            onClick = onExportLog,
            modifier = Modifier.fillMaxWidth(),
            colors = ButtonDefaults.outlinedButtonColors(),
        ) { Text("📤 导出日志") }
        Spacer(Modifier.height(8.dp))
        Button(
            onClick = onClearCache,
            modifier = Modifier.fillMaxWidth(),
            colors = ButtonDefaults.outlinedButtonColors(),
        ) { Text("🗑️ 清除缓存") }
        Spacer(Modifier.height(40.dp))
    }
}

@Composable
private fun SettingsSection(title: String) {
    Text(title, fontSize = 13.sp, color = MaterialTheme.colorScheme.primary, fontWeight = FontWeight.SemiBold)
    Spacer(Modifier.height(8.dp))
}

@Composable
private fun ThemeSelector(current: AppTheme, onSelect: (AppTheme) -> Unit) {
    val themes = AppTheme.values()
    Row(
        horizontalArrangement = Arrangement.spacedBy(8.dp),
        modifier = Modifier.fillMaxWidth(),
    ) {
        themes.forEach { theme ->
            FilterChip(
                selected = current == theme,
                onClick = { onSelect(theme) },
                label = { Text(theme.displayName, fontSize = 12.sp) },
                modifier = Modifier.weight(1f),
            )
        }
    }
}

@Composable
private fun SwitchRow(
    title: String, checked: Boolean,
    onCheckedChange: (Boolean) -> Unit,
    subtitle: String = "",
) {
    Row(
        verticalAlignment = Alignment.CenterVertically,
        modifier = Modifier.fillMaxWidth().padding(vertical = 4.dp),
    ) {
        Column(Modifier.weight(1f)) {
            Text(title, fontSize = 14.sp)
            if (subtitle.isNotEmpty()) {
                Text(subtitle, fontSize = 11.sp, color = MaterialTheme.colorScheme.onSurfaceVariant)
            }
        }
        Switch(checked = checked, onCheckedChange = onCheckedChange)
    }
}

@Composable
private fun SliderRow(
    title: String, value: Float,
    range: ClosedFloatingPointRange<Float>, steps: Int, valueText: String,
    onValueChange: (Int) -> Unit,
) {
    Row(verticalAlignment = Alignment.CenterVertically) {
        Text(title, Modifier.width(100.dp), fontSize = 14.sp)
        Slider(
            value = value, onValueChange = { onValueChange(it.toInt()) },
            valueRange = range, steps = steps,
            modifier = Modifier.weight(1f),
        )
        Text(valueText, Modifier.width(50.dp), fontSize = 13.sp, color = MaterialTheme.colorScheme.primary)
    }
}

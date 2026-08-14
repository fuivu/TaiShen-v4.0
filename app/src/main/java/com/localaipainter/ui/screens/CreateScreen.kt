package com.localaipainter.ui.screens

import androidx.compose.foundation.*
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.lazy.*
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.localaipainter.engine.*
import com.localaipainter.data.PromptTemplates

/**
 * 创作页 — 太神架构 v4.0
 * 支持混动模式 + 后端选择 + 并行策略 + DevicePowerScore 自适应
 */
 */
@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun CreateScreen(
    config: GenerationConfig,
    onConfigChange: (GenerationConfig) -> Unit,
    onGenerate: () -> Unit,
    onStop: () -> Unit,
    isGenerating: Boolean,
    progress: Float,
    currentStep: Int,
    totalSteps: Int,
    previewPath: String?,
    recentTemplates: List<PromptTemplate> = PromptTemplates.ALL.take(6),
    // 新增：后端 & 混动
    availableBackends: List<BackendInfo> = emptyList(),
    selectedBackend: String = "HYBRID",
    onBackendChange: (String) -> Unit = {},
    hybridMode: Int = HybridPipeline.MODE_ADAPTIVE,
    onHybridModeChange: (Int) -> Unit = {},
    perfStats: String = "",
) {
    Column(
        modifier = Modifier
            .fillMaxSize()
            .verticalScroll(rememberScrollState())
            .padding(16.dp)
    ) {
        // ===== 标题 =====
        Text("✨ 创作", fontSize = 24.sp, fontWeight = FontWeight.Bold)
        Spacer(Modifier.height(4.dp))
        Text(
            "输入提示词，AI 将为你生成图像",
            fontSize = 13.sp, color = MaterialTheme.colorScheme.onSurfaceVariant,
        )
        Spacer(Modifier.height(16.dp))

        // ===== 后端选择卡片（太神架构 v4.0）=====
        Card(
            modifier = Modifier.fillMaxWidth(),
            shape = RoundedCornerShape(16.dp),
            colors = CardDefaults.cardColors(
                containerColor = MaterialTheme.colorScheme.primaryContainer
            ),
        ) {
            Column(Modifier.padding(16.dp)) {
                Text("🚀 推理引擎", fontSize = 16.sp, fontWeight = FontWeight.SemiBold)
                Spacer(Modifier.height(4.dp))
                Text(
                    "选择 AI 推理后端，混动模式可同时利用 CPU+GPU 加速",
                    fontSize = 11.sp, color = MaterialTheme.colorScheme.onSurfaceVariant,
                )
                Spacer(Modifier.height(12.dp))

                // 后端选择 Row
                LazyRow(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                    items(availableBackends.size) { i ->
                        val backend = availableBackends[i]
                        val isSelected = selectedBackend == backend.name
                        FilterChip(
                            selected = isSelected,
                            onClick = { onBackendChange(backend.name) },
                            label = {
                                Text(
                                    "${backend.icon} ${backend.displayName}",
                                    fontSize = 12.sp,
                                    fontWeight = if (isSelected) FontWeight.Bold else FontWeight.Normal,
                                )
                            },
                            colors = FilterChipDefaults.filterChipColors(
                                selectedContainerColor = MaterialTheme.colorScheme.primary,
                                selectedLabelColor = MaterialTheme.colorScheme.onPrimary,
                            ),
                            enabled = backend.available,
                        )
                    }
                }

                // 混动模式子选项
                if (selectedBackend == "HYBRID") {
                    Spacer(Modifier.height(12.dp))
                    Text("并行策略", fontSize = 13.sp, fontWeight = FontWeight.Medium)
                    Spacer(Modifier.height(6.dp))
                    Row(horizontalArrangement = Arrangement.spacedBy(6.dp)) {
                        HybridModeChip(
                            label = "🎯 自适应",
                            desc = "自动选择最优策略",
                            selected = hybridMode == HybridPipeline.MODE_ADAPTIVE,
                            onClick = { onHybridModeChange(HybridPipeline.MODE_ADAPTIVE) },
                        )
                        HybridModeChip(
                            label = "⏭ 串行",
                            desc = "CPU→GPU 交替",
                            selected = hybridMode == HybridPipeline.MODE_SEQUENTIAL,
                            onClick = { onHybridModeChange(HybridPipeline.MODE_SEQUENTIAL) },
                        )
                        HybridModeChip(
                            label = "🔀 流水线",
                            desc = "CPU/GPU 同时跑不同层",
                            selected = hybridMode == HybridPipeline.MODE_PIPELINED,
                            onClick = { onHybridModeChange(HybridPipeline.MODE_PIPELINED) },
                        )
                        HybridModeChip(
                            label = "💥 数据并行",
                            desc = "数据分两半同时算",
                            selected = hybridMode == HybridPipeline.MODE_DATA_PARALLEL,
                            onClick = { onHybridModeChange(HybridPipeline.MODE_DATA_PARALLEL) },
                        )
                    }
                }

                // 性能统计
                if (perfStats.isNotEmpty()) {
                    Spacer(Modifier.height(10.dp))
                    Divider(color = MaterialTheme.colorScheme.outlineVariant)
                    Spacer(Modifier.height(8.dp))
                    Text(
                        perfStats,
                        fontSize = 11.sp,
                        color = MaterialTheme.colorScheme.onSurfaceVariant,
                        fontFamily = androidx.compose.ui.text.font.FontFamily.Monospace,
                    )
                }
            }
        }
        Spacer(Modifier.height(16.dp))

        // ===== 快速模板 =====
        Text("快捷模板", fontSize = 15.sp, fontWeight = FontWeight.SemiBold)
        Spacer(Modifier.height(8.dp))
        LazyRow(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
            items(recentTemplates.size) { i ->
                val tpl = recentTemplates[i]
                AssistChip(
                    onClick = { onConfigChange(config.copy(prompt = tpl.prompt, negativePrompt = tpl.negativePrompt)) },
                    label = { Text("${tpl.thumbnailEmoji} ${tpl.name}") },
                    colors = AssistChipDefaults.assistChipColors(
                        containerColor = MaterialTheme.colorScheme.surfaceVariant,
                    ),
                )
            }
        }
        Spacer(Modifier.height(16.dp))

        // ===== 提示词输入 =====
        OutlinedTextField(
            value = config.prompt,
            onValueChange = { onConfigChange(config.copy(prompt = it)) },
            label = { Text("提示词 (Prompt)") },
            placeholder = { Text("描述你想画的画面...") },
            minLines = 3, maxLines = 6,
            modifier = Modifier.fillMaxWidth(),
            shape = RoundedCornerShape(12.dp),
        )
        Spacer(Modifier.height(12.dp))

        // ===== 负向提示词 =====
        OutlinedTextField(
            value = config.negativePrompt,
            onValueChange = { onConfigChange(config.copy(negativePrompt = it)) },
            label = { Text("负向提示词 (Negative)") },
            placeholder = { Text("不想出现的内容...") },
            minLines = 2, maxLines = 4,
            modifier = Modifier.fillMaxWidth(),
            shape = RoundedCornerShape(12.dp),
        )
        Spacer(Modifier.height(16.dp))

        // ===== 参数面板 =====
        Card(
            modifier = Modifier.fillMaxWidth(),
            shape = RoundedCornerShape(16.dp),
            colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surfaceVariant),
        ) {
            Column(Modifier.padding(16.dp)) {
                Text("⚙️ 参数设置", fontSize = 16.sp, fontWeight = FontWeight.SemiBold)
                Spacer(Modifier.height(12.dp))

                // 分辨率
                Row(verticalAlignment = Alignment.CenterVertically) {
                    Text("分辨率", Modifier.weight(1f), fontSize = 14.sp)
                    ResolutionSelector(
                        width = config.width, height = config.height,
                        onSelect = { w, h -> onConfigChange(config.copy(width = w, height = h)) },
                    )
                }
                Spacer(Modifier.height(8.dp))

                // 采样器
                Row(verticalAlignment = Alignment.CenterVertically) {
                    Text("采样器", Modifier.weight(1f), fontSize = 14.sp)
                    SchedulerDropdown(
                        selected = config.scheduler,
                        onSelect = { onConfigChange(config.copy(scheduler = it)) },
                    )
                }
                Spacer(Modifier.height(8.dp))

                // 步数
                SliderWithLabel(
                    label = "步数", value = config.steps.toFloat(),
                    onValueChange = { onConfigChange(config.copy(steps = it.toInt())) },
                    range = 1f..100f, steps = 99, valueText = "${config.steps}",
                )

                // CFG
                SliderWithLabel(
                    label = "CFG 引导", value = config.cfgScale,
                    onValueChange = { onConfigChange(config.copy(cfgScale = it)) },
                    range = 1f..20f, steps = 38, valueText = "%.1f".format(config.cfgScale),
                )

                // 种子
                Row(verticalAlignment = Alignment.CenterVertically) {
                    Text("种子", Modifier.weight(1f), fontSize = 14.sp)
                    OutlinedTextField(
                        value = if (config.seed < 0) "" else config.seed.toString(),
                        onValueChange = { s ->
                            val v = s.toLongOrNull() ?: -1
                            onConfigChange(config.copy(seed = v))
                        },
                        placeholder = { Text("随机") },
                        modifier = Modifier.width(120.dp), singleLine = true,
                    )
                }
                Spacer(Modifier.height(8.dp))

                // 批量
                SliderWithLabel(
                    label = "批量", value = config.batchSize.toFloat(),
                    onValueChange = { onConfigChange(config.copy(batchSize = it.toInt())) },
                    range = 1f..9f, steps = 8, valueText = "${config.batchSize}",
                )

                // CLIP Skip
                SliderWithLabel(
                    label = "CLIP跳", value = config.clipSkip.toFloat(),
                    onValueChange = { onConfigChange(config.copy(clipSkip = it.toInt())) },
                    range = 1f..4f, steps = 3, valueText = "${config.clipSkip}",
                )

                // 重绘幅度
                SliderWithLabel(
                    label = "重绘", value = config.denoisingStrength,
                    onValueChange = { onConfigChange(config.copy(denoisingStrength = it)) },
                    range = 0f..1f, steps = 20, valueText = "%.2f".format(config.denoisingStrength),
                )

                // 线程数（混动时控制 CPU 侧）
                if (selectedBackend == "HYBRID") {
                    SliderWithLabel(
                        label = "CPU线程", value = config.threads.toFloat(),
                        onValueChange = { onConfigChange(config.copy(threads = it.toInt())) },
                        range = 1f..8f, steps = 7, valueText = "${config.threads}",
                    )
                }
            }
        }
        Spacer(Modifier.height(16.dp))

        // ===== 进度条 =====
        if (isGenerating) {
            LinearProgressIndicator(
                progress = { progress.coerceIn(0f, 1f) },
                modifier = Modifier.fillMaxWidth().height(8.dp),
                color = MaterialTheme.colorScheme.primary,
                trackColor = MaterialTheme.colorScheme.surfaceVariant,
            )
            Spacer(Modifier.height(4.dp))
            Text(
                "Step $currentStep / $totalSteps  (${(progress*100).toInt()}%)",
                fontSize = 12.sp, color = MaterialTheme.colorScheme.onSurfaceVariant,
            )
            Spacer(Modifier.height(12.dp))
        }

        // ===== 生成按钮 =====
        Button(
            onClick = if (isGenerating) onStop else onGenerate,
            modifier = Modifier.fillMaxWidth().height(56.dp),
            shape = RoundedCornerShape(28.dp),
            colors = ButtonDefaults.buttonColors(
                containerColor = if (isGenerating) MaterialTheme.colorScheme.error
                                else MaterialTheme.colorScheme.primary,
            ),
        ) {
            Text(
                if (isGenerating) "⏹ 停止生成" else "✨ 开始生成",
                fontSize = 18.sp, fontWeight = FontWeight.Bold,
            )
        }
        Spacer(Modifier.height(24.dp))
    }
}

// ============ 子组件 ============

@Composable
private fun HybridModeChip(
    label: String, desc: String,
    selected: Boolean, onClick: () -> Unit,
) {
    AssistChip(
        selected = selected,
        onClick = onClick,
        label = {
            Column {
                Text(label, fontSize = 11.sp, fontWeight = FontWeight.SemiBold)
                Text(desc, fontSize = 9.sp, color = MaterialTheme.colorScheme.onSurfaceVariant)
            }
        },
        colors = AssistChipDefaults.assistChipColors(
            selectedContainerColor = MaterialTheme.colorScheme.tertiaryContainer,
            selectedLabelColor = MaterialTheme.colorScheme.onTertiaryContainer,
        ),
    )
}

@Composable
private fun SliderWithLabel(
    label: String, value: Float,
    onValueChange: (Float) -> Unit,
    range: ClosedFloatingPointRange<Float>,
    steps: Int, valueText: String,
) {
    Row(verticalAlignment = Alignment.CenterVertically) {
        Text(label, Modifier.width(80.dp), fontSize = 13.sp)
        Slider(
            value = value, onValueChange = onValueChange,
            valueRange = range, steps = steps,
            modifier = Modifier.weight(1f),
        )
        Text(valueText, Modifier.width(55.dp), fontSize = 13.sp,
             color = MaterialTheme.colorScheme.primary)
    }
}

@Composable
private fun ResolutionSelector(
    width: Int, height: Int,
    onSelect: (Int, Int) -> Unit,
) {
    var expanded by remember { mutableStateOf(false) }
    val options = listOf("512×512", "512×768", "768×512", "768×768", "1024×1024")
    val current = "${width}×${height}"
    Box {
        OutlinedButton(onClick = { expanded = true }) { Text(current) }
        DropdownMenu(expanded = expanded, onDismissRequest = { expanded = false }) {
            options.forEach { opt ->
                val parts = opt.split("×")
                DropdownMenuItem(
                    text = { Text(opt) },
                    onClick = { onSelect(parts[0].toInt(), parts[1].toInt()); expanded = false },
                )
            }
        }
    }
}

@OptIn(ExperimentalMaterial3Api::class)
@Composable
private fun SchedulerDropdown(
    selected: SchedulerType,
    onSelect: (SchedulerType) -> Unit,
) {
    var expanded by remember { mutableStateOf(false) }
    val items = SchedulerType.values().toList()
    Box {
        OutlinedButton(onClick = { expanded = true }) { Text(selected.displayName) }
        DropdownMenu(expanded = expanded, onDismissRequest = { expanded = false }) {
            items.forEach { s ->
                DropdownMenuItem(
                    text = { Text(s.displayName) },
                    onClick = { onSelect(s); expanded = false },
                )
            }
        }
    }
}

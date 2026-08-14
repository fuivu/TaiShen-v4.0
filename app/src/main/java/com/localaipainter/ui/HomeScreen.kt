package com.localaipainter.ui.home

import androidx.compose.foundation.*
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.localaipainter.App
import com.localaipainter.core.*
import com.localaipainter.scheduler.SchedulerFactory
import kotlinx.coroutines.launch

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun HomeScreen(
    app: App,
    onImageGenerated: () -> Unit
) {
    var prompt by remember { mutableStateOf("") }
    var negativePrompt by remember { mutableStateOf("") }
    var steps by remember { mutableIntStateOf(20) }
    var cfgScale by remember { mutableFloatStateOf(7.5f) }
    var seed by remember { mutableLongStateOf(-1L) }
    var width by remember { mutableIntStateOf(512) }
    var height by remember { mutableIntStateOf(512) }
    var isGenerating by remember { mutableStateOf(false) }
    var progress by remember { mutableFloatStateOf(0f) }
    var statusText by remember { mutableStateOf("就绪") }
    var selectedScheduler by remember { mutableStateOf("LCM (最快)") }
    var selectedEngine by remember { mutableStateOf("自动选择") }

    val scope = rememberCoroutineScope()
    val schedulerFactory = remember { SchedulerFactory() }
    val memoryManager = remember { app.memoryManager }

    Column(
        modifier = Modifier
            .fillMaxSize()
            .background(Color(0xFF0F0F1A))
            .verticalScroll(rememberScrollState())
            .padding(16.dp)
    ) {
        Text(
            "AI 绘画",
            fontSize = 24.sp,
            fontWeight = FontWeight.Bold,
            color = Color.White
        )
        Spacer(Modifier.height(16.dp))

        // Prompt 输入
        OutlinedTextField(
            value = prompt,
            onValueChange = { prompt = it },
            label = { Text("提示词（英文效果更佳）", color = Color.Gray) },
            placeholder = { Text("A beautiful sunset over mountains...", color = Color.DarkGray) },
            modifier = Modifier.fillMaxWidth(),
            minLines = 3,
            colors = OutlinedTextFieldDefaults.colors(
                focusedTextColor = Color.White,
                unfocusedTextColor = Color.White,
                focusedBorderColor = Color(0xFF7C4DFF),
                unfocusedBorderColor = Color(0xFF303048)
            )
        )
        Spacer(Modifier.height(8.dp))

        // Negative prompt
        OutlinedTextField(
            value = negativePrompt,
            onValueChange = { negativePrompt = it },
            label = { Text("不想出现的元素（可选）", color = Color.Gray) },
            modifier = Modifier.fillMaxWidth(),
            minLines = 2,
            colors = OutlinedTextFieldDefaults.colors(
                focusedTextColor = Color.White,
                unfocusedTextColor = Color.White,
                focusedBorderColor = Color(0xFF7C4DFF),
                unfocusedBorderColor = Color(0xFF303048)
            )
        )
        Spacer(Modifier.height(16.dp))

        // 参数卡片
        Card(
            modifier = Modifier.fillMaxWidth(),
            colors = CardDefaults.cardColors(containerColor = Color(0xFF1A1A2E)),
            shape = RoundedCornerShape(12.dp)
        ) {
            Column(Modifier.padding(16.dp)) {
                Text("参数设置", color = Color.White, fontWeight = FontWeight.Bold)

                Spacer(Modifier.height(8.dp))
                Text("步数: $steps", color = Color.LightGray)
                Slider(
                    value = steps.toFloat(),
                    onValueChange = { steps = it.toInt() },
                    valueRange = 4f..50f,
                    steps = 46,
                    colors = SliderDefaults.colors(
                        thumbColor = Color(0xFF7C4DFF),
                        activeTrackColor = Color(0xFF00E5FF)
                    )
                )

                Text("CFG Scale: ${"%.1f".format(cfgScale)}", color = Color.LightGray)
                Slider(
                    value = cfgScale,
                    onValueChange = { cfgScale = it },
                    valueRange = 1f..20f,
                    steps = 38,
                    colors = SliderDefaults.colors(
                        thumbColor = Color(0xFF7C4DFF),
                        activeTrackColor = Color(0xFF00E5FF)
                    )
                )

                Text("随机种子: $seed (-1=随机)", color = Color.LightGray)
                Slider(
                    value = if (seed < 0) -1f else seed.toFloat(),
                    onValueChange = { seed = it.toLong() },
                    valueRange = -1f..99999f,
                    colors = SliderDefaults.colors(
                        thumbColor = Color(0xFF7C4DFF),
                        activeTrackColor = Color(0xFF00E5FF)
                    )
                )

                // 分辨率选择
                Row(Modifier.fillMaxWidth()) {
                    Text("分辨率:", color = Color.LightGray, Modifier.weight(1f))
                    listOf(256, 384, 512, 768).forEach { size ->
                        val selected = width == size
                        FilterChip(
                            selected = selected,
                            onClick = { width = size; height = size },
                            label = { Text("${size}px", fontSize = 10.sp) },
                            colors = FilterChipDefaults.filterChipColors(
                                selectedContainerColor = Color(0xFF7C4DFF)
                            )
                        )
                        Spacer(Modifier.width(4.dp))
                    }
                }

                Spacer(Modifier.height(8.dp))

                // 调度器选择
                var schedulerExpanded by remember { mutableStateOf(false) }
                ExposedDropdownMenuBox(
                    expanded = schedulerExpanded,
                    onExpandedChange = { schedulerExpanded = !schedulerExpanded }
                ) {
                    OutlinedTextField(
                        value = selectedScheduler,
                        onValueChange = {},
                        readOnly = true,
                        label = { Text("采样器", color = Color.Gray) },
                        trailingIcon = { ExposedDropdownMenuDefaults.TrailingIcon(expanded = schedulerExpanded) },
                        modifier = Modifier.fillMaxWidth().menuAnchor(),
                        colors = OutlinedTextFieldDefaults.colors(
                            focusedTextColor = Color.White,
                            unfocusedTextColor = Color.White
                        )
                    )
                    ExposedDropdownMenu(
                        expanded = schedulerExpanded,
                        onDismissRequest = { schedulerExpanded = false }
                    ) {
                        schedulerFactory.getAllNames().forEach { name ->
                            DropdownMenuItem(
                                text = { Text(name, color = Color.White) },
                                onClick = {
                                    selectedScheduler = name
                                    schedulerExpanded = false
                                }
                            )
                        }
                    }
                }
            }
        }

        Spacer(Modifier.height(16.dp))

        // 状态显示
        if (isGenerating) {
            LinearProgressIndicator(
                progress = { progress },
                modifier = Modifier.fillMaxWidth(),
                color = Color(0xFF00E5FF),
                trackColor = Color(0xFF252540)
            )
            Spacer(Modifier.height(8.dp))
            Text(statusText, color = Color(0xFF00E5FF))
            Spacer(Modifier.height(8.dp))
        }

        // 生成按钮
        Button(
            onClick = {
                if (prompt.isBlank()) return@Button
                scope.launch {
                    isGenerating = true
                    progress = 0f
                    statusText = "初始化引擎..."
                    // 实际生成逻辑在此调用 app.engineFactory + SDPipeline
                    // 简化：模拟进度
                    repeat(steps) { i ->
                        progress = (i + 1) / steps.toFloat()
                        statusText = "生成中... ${i+1}/$steps"
                        kotlinx.coroutines.delay(100)
                    }
                    isGenerating = false
                    statusText = "生成完成"
                    onImageGenerated()
                }
            },
            modifier = Modifier.fillMaxWidth().height(56.dp),
            colors = ButtonDefaults.buttonColors(
                containerColor = Color(0xFF7C4DFF)
            ),
            enabled = !isGenerating && prompt.isNotBlank()
        ) {
            Text(if (isGenerating) "生成中..." else "开始生成", fontSize = 18.sp)
        }

        Spacer(Modifier.height(8.dp))

        // 内存信息
        Text(
            app.memoryManager.getMemoryReport(),
            color = Color.DarkGray,
            fontSize = 10.sp
        )
    }
}

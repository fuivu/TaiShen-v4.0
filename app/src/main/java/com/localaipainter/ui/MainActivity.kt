package com.localaipainter.ui

import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.compose.foundation.layout.*
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.unit.dp
import com.localaipainter.App
import com.localaipainter.engine.HeterogeneousPipeline
import com.localaipainter.scheduler.SchedulerFactory
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext

class MainActivity : ComponentActivity() {

    private lateinit var pipeline: HeterogeneousPipeline

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        pipeline = App.instance.heteroPipeline

        setContent {
            MaterialTheme(
                colorScheme = darkColorScheme(
                    primary = Color(0xFF7C4DFF),
                    secondary = Color(0xFF00BCD4),
                    background = Color(0xFF0F0F1A),
                    surface = Color(0xFF1A1A2E),
                    onPrimary = Color.White,
                    onBackground = Color(0xFFE0E0E0),
                    onSurface = Color(0xFFE0E0E0),
                )
            ) {
                Surface(
                    modifier = Modifier.fillMaxSize(),
                    color = MaterialTheme.colorScheme.background
                ) {
                    MainScreen(pipeline)
                }
            }
        }
    }
}

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun MainScreen(pipeline: HeterogeneousPipeline) {
    var prompt by remember { mutableStateOf("") }
    var negativePrompt by remember { mutableStateOf("") }
    var steps by remember { mutableStateOf(20) }
    var cfgScale by remember { mutableStateOf(7.5f) }
    var isGenerating by remember { mutableStateOf(false) }
    var progress by remember { mutableFloatStateOf(0f) }
    var selectedScheduler by remember { mutableStateOf("Euler A") }
    val schedulers = remember { SchedulerFactory.names() }

    val scope = rememberCoroutineScope()

    Scaffold(
        topBar = {
            TopAppBar(
                title = { Text("Local AI Painter v4.0 TaiShen") },
                colors = TopAppBarDefaults.topAppBarColors(
                    containerColor = Color(0xFF0F0F1A),
                    titleContentColor = Color.White
                )
            )
        },
        containerColor = Color(0xFF0F0F1A)
    ) { padding ->
        Column(
            modifier = Modifier
                .fillMaxSize()
                .padding(padding)
                .padding(16.dp),
            verticalArrangement = Arrangement.spacedBy(12.dp)
        ) {
            // Prompt 输入
            OutlinedTextField(
                value = prompt,
                onValueChange = { prompt = it },
                label = { Text("描述你想画的画面...") },
                modifier = Modifier.fillMaxWidth(),
                minLines = 3,
                colors = OutlinedTextFieldDefaults.colors(
                    focusedBorderColor = Color(0xFF7C4DFF),
                    unfocusedBorderColor = Color(0xFF33242424),
                    cursorColor = Color(0xFF7C4DFF),
                    focusedLabelColor = Color(0xFF7C4DFF),
                )
            )

            // Negative Prompt
            OutlinedTextField(
                value = negativePrompt,
                onValueChange = { negativePrompt = it },
                label = { Text("不想出现的内容...") },
                modifier = Modifier.fillMaxWidth(),
                minLines = 2,
                colors = OutlinedTextFieldDefaults.colors(
                    focusedBorderColor = Color(0xFF7C4DFF),
                    unfocusedBorderColor = Color(0xFF33242424),
                    cursorColor = Color(0xFF7C4DFF),
                )
            )

            // 调度器选择
            Text("采样器: $selectedScheduler", color = Color(0xFFE0E0E0))
            Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                schedulers.forEach { name ->
                    FilterChip(
                        selected = selectedScheduler == name,
                        onClick = { selectedScheduler = name },
                        label = { Text(name) },
                        colors = FilterChipDefaults.filterChipColors(
                            selectedContainerColor = Color(0xFF7C4DFF),
                            selectedLabelColor = Color.White,
                        )
                    )
                }
            }

            // 步数滑块
            Text("步数: $steps", color = Color(0xFFE0E0E0))
            Slider(
                value = steps.toFloat(),
                onValueChange = { steps = it.toInt() },
                valueRange = 5f..50f,
                steps = 9,
                colors = SliderDefaults.colors(
                    thumbColor = Color(0xFF7C4DFF),
                    activeTrackColor = Color(0xFF7C4DFF),
                )
            )

            // CFG 滑块
            Text("CFG 引导: ${"%.1f".format(cfgScale)}", color = Color(0xFFE0E0E0))
            Slider(
                value = cfgScale,
                onValueChange = { cfgScale = it },
                valueRange = 1f..15f,
                steps = 14,
                colors = SliderDefaults.colors(
                    thumbColor = Color(0xFF00BCD4),
                    activeTrackColor = Color(0xFF00BCD4),
                )
            )

            // 进度条
            if (isGenerating) {
                LinearProgressIndicator(
                    progress = progress,
                    modifier = Modifier.fillMaxWidth(),
                    color = Color(0xFF7C4DFF),
                    trackColor = Color(0xFF33242424),
                )
            }

            Spacer(Modifier.weight(1f))

            // 生成按钮
            Button(
                onClick = {
                    isGenerating = true
                    progress = 0f
                    pipeline.setScheduler(selectedScheduler)
                    scope.launch {
                        pipeline.generate(
                            prompt = prompt,
                            negativePrompt = negativePrompt,
                            steps = steps,
                            cfgScale = cfgScale,
                            onProgress = { step, total ->
                                // onProgress 在 Dispatchers.Default 调用，切回 Main 更新 UI
                                launch(Dispatchers.Main) {
                                    progress = step.toFloat() / total
                                }
                            }
                        )
                        withContext(Dispatchers.Main) {
                            isGenerating = false
                        }
                    }
                },
                enabled = !isGenerating && prompt.isNotBlank(),
                modifier = Modifier
                    .fillMaxWidth()
                    .height(56.dp),
                colors = ButtonDefaults.buttonColors(
                    containerColor = Color(0xFF7C4DFF),
                    disabledContainerColor = Color(0xFF33242424),
                )
            ) {
                Text(
                    if (isGenerating) "生成中..." else "✨ 开始生成",
                    color = Color.White
                )
            }
        }
    }
}

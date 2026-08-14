package com.localaipainter.ui.screens

import androidx.compose.foundation.*
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.lazy.*
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.compose.ui.graphics.Color

/**
 * 模型管理页 — Checkpoint / LoRA / ControlNet / VAE / Embedding
 */
@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun ModelsScreen(
    checkpointPaths: List<String>,
    loraPaths: List<String>,
    controlNetPaths: List<String>,
    vaePaths: List<String>,
    embeddingPaths: List<String>,
    selectedCheckpoint: String,
    onSelectCheckpoint: (String) -> Unit,
    onImportModel: (ModelCategory) -> Unit,
    onDeleteModel: (ModelCategory, String) -> Unit,
    loraWeights: Map<String, Float> = emptyMap(),
    onLoraWeightChange: (String, Float) -> Unit = { _, _ -> },
) {
    var selectedTab by remember { mutableStateOf(ModelCategory.CHECKPOINT) }

    Column(Modifier.fillMaxSize()) {
        // ===== 标题 =====
        Text(
            "📦 模型管理",
            fontSize = 24.sp, fontWeight = FontWeight.Bold,
            modifier = Modifier.padding(16.dp),
        )

        // ===== Tab 栏 =====
        ScrollableTabRow(
            selectedTabIndex = selectedTab.ordinal,
            modifier = Modifier.padding(horizontal = 8.dp),
            edgePadding = 8.dp,
        ) {
            ModelCategory.values().forEach { cat ->
                Tab(
                    selected = selectedTab == cat,
                    onClick = { selectedTab = cat },
                    text = { Text(cat.displayName) },
                )
            }
        }

        Spacer(Modifier.height(12.dp))

        // ===== 导入按钮 =====
        Button(
            onClick = { onImportModel(selectedTab) },
            modifier = Modifier.padding(horizontal = 16.dp).fillMaxWidth(),
            shape = MaterialTheme.shapes.medium,
        ) {
            Text("📥 导入${selectedTab.displayName}")
        }

        Spacer(Modifier.height(12.dp))

        // ===== 列表 =====
        when (selectedTab) {
            ModelCategory.CHECKPOINT -> ModelList(
                items = checkpointPaths,
                selected = selectedCheckpoint,
                onSelect = onSelectCheckpoint,
                onDelete = { onDeleteModel(ModelCategory.CHECKPOINT, it) },
                showSelected = true,
            )
            ModelCategory.LORA -> LoraList(
                items = loraPaths,
                weights = loraWeights,
                onWeightChange = onLoraWeightChange,
                onDelete = { onDeleteModel(ModelCategory.LORA, it) },
            )
            ModelCategory.CONTROLNET -> ModelList(
                items = controlNetPaths,
                onDelete = { onDeleteModel(ModelCategory.CONTROLNET, it) },
            )
            ModelCategory.VAE -> ModelList(
                items = vaePaths,
                onDelete = { onDeleteModel(ModelCategory.VAE, it) },
            )
            ModelCategory.EMBEDDING -> ModelList(
                items = embeddingPaths,
                onDelete = { onDeleteModel(ModelCategory.EMBEDDING, it) },
            )
        }
    }
}

@Composable
private fun ModelList(
    items: List<String>,
    selected: String = "",
    onSelect: (String) -> Unit = {},
    onDelete: (String) -> Unit = {},
    showSelected: Boolean = false,
) {
    if (items.isEmpty()) {
        Box(Modifier.fillMaxWidth().padding(40.dp), contentAlignment = Alignment.Center) {
            Text("暂无模型文件", fontSize = 14.sp, color = MaterialTheme.colorScheme.onSurfaceVariant)
        }
    } else {
        LazyColumn(
            modifier = Modifier.fillMaxSize().padding(horizontal = 12.dp),
            verticalArrangement = Arrangement.spacedBy(6.dp),
        ) {
            items(items.size) { i ->
                val path = items[i]
                val name = path.substringAfterLast("/")
                Card(
                    modifier = Modifier.fillMaxWidth(),
                    shape = MaterialTheme.shapes.small,
                    colors = CardDefaults.cardColors(
                        containerColor = if (showSelected && path == selected)
                            MaterialTheme.colorScheme.primary.copy(alpha = 0.15f)
                        else MaterialTheme.colorScheme.surfaceVariant,
                    ),
                    onClick = { onSelect(path) },
                ) {
                    Row(
                        modifier = Modifier.padding(12.dp).fillMaxWidth(),
                        verticalAlignment = Alignment.CenterVertically,
                    ) {
                        Text(
                            if (showSelected && path == selected) "✅ " else "📄 ",
                            fontSize = 16.sp,
                        )
                        Text(
                            name, Modifier.weight(1f), fontSize = 13.sp,
                            color = MaterialTheme.colorScheme.onSurface,
                        )
                        IconButton(onClick = { onDelete(path) }, modifier = Modifier.size(32.dp)) {
                            Text("🗑️", fontSize = 14.sp)
                        }
                    }
                }
            }
        }
    }
}

@Composable
private fun LoraList(
    items: List<String>,
    weights: Map<String, Float>,
    onWeightChange: (String, Float) -> Unit,
    onDelete: (String) -> Unit,
) {
    if (items.isEmpty()) {
        Box(Modifier.fillMaxWidth().padding(40.dp), contentAlignment = Alignment.Center) {
            Text("暂无 LoRA 模型", fontSize = 14.sp, color = MaterialTheme.colorScheme.onSurfaceVariant)
        }
    } else {
        LazyColumn(
            modifier = Modifier.fillMaxSize().padding(horizontal = 12.dp),
            verticalArrangement = Arrangement.spacedBy(6.dp),
        ) {
            items(items.size) { i ->
                val path = items[i]
                val name = path.substringAfterLast("/")
                val weight = weights[path] ?: 1.0f
                Card(
                    modifier = Modifier.fillMaxWidth(),
                    shape = MaterialTheme.shapes.small,
                    colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surfaceVariant),
                ) {
                    Column(Modifier.padding(12.dp)) {
                        Row(verticalAlignment = Alignment.CenterVertically) {
                            Text("🎯 $name", Modifier.weight(1f), fontSize = 13.sp)
                            Text("权重: %.2f".format(weight), fontSize = 11.sp, color = MaterialTheme.colorScheme.onSurfaceVariant)
                        }
                        Spacer(Modifier.height(4.dp))
                        Slider(
                            value = weight,
                            onValueChange = { onWeightChange(path, it) },
                            valueRange = 0f..2f,
                            steps = 20,
                        )
                        Row {
                            Spacer(Modifier.weight(1f))
                            TextButton(onClick = { onDelete(path) }) { Text("删除") }
                        }
                    }
                }
            }
        }
    }
}

enum class ModelCategory(val displayName: String) {
    CHECKPOINT("Checkpoint"),
    LORA("LoRA"),
    CONTROLNET("ControlNet"),
    VAE("VAE"),
    EMBEDDING("Embedding"),
}

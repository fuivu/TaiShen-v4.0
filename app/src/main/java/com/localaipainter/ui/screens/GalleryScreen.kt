package com.localaipainter.ui.screens

import androidx.compose.foundation.*
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.lazy.grid.*
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.compose.foundation.selection.selectable
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.graphics.asImageBitmap
import android.graphics.BitmapFactory
import java.io.File

/**
 * 画廊页 — 网格展示 + 筛选 + 多选操作
 */
@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun GalleryScreen(
    imageDir: String,
    onImageClick: (String) -> Unit,
    onDeleteSelected: (List<String>) -> Unit,
    onToggleFavorite: (String) -> Unit,
    favoritePaths: Set<String> = emptySet(),
) {
    var filter by remember { mutableStateOf(GalleryFilter.ALL) }
    var selectedPaths by remember { mutableStateOf<Set<String>>(emptySet()) }
    val isMultiSelect = selectedPaths.isNotEmpty()

    // 扫描目录
    val allImages = remember(imageDir) {
        val dir = File(imageDir)
        if (dir.exists()) {
            dir.listFiles { f -> f.extension.lowercase() in setOf("png","jpg","jpeg","webp") }
                ?.sortedByDescending { it.lastModified() }
                ?.map { it.absolutePath }
                ?: emptyList()
        } else emptyList()
    }

    val filtered = when (filter) {
        GalleryFilter.ALL     -> allImages
        GalleryFilter.FAVORITE-> allImages.filter { favoritePaths.contains(it) }
        GalleryFilter.RECENT  -> allImages.take(20)
    }

    Column(Modifier.fillMaxSize()) {
        // ===== 标题 + 筛选 =====
        Row(
            modifier = Modifier.padding(16.dp),
            verticalAlignment = Alignment.CenterVertically,
        ) {
            Text("🖼️ 画廊", fontSize = 24.sp, fontWeight = FontWeight.Bold, modifier = Modifier.weight(1f))
            FilterChip(
                selected = filter == GalleryFilter.ALL, onClick = { filter = GalleryFilter.ALL },
                label = { Text("全部") },
            )
            Spacer(Modifier.width(8.dp))
            FilterChip(
                selected = filter == GalleryFilter.FAVORITE, onClick = { filter = GalleryFilter.FAVORITE },
                label = { Text("⭐ 收藏") },
            )
            Spacer(Modifier.width(8.dp))
            FilterChip(
                selected = filter == GalleryFilter.RECENT, onClick = { filter = GalleryFilter.RECENT },
                label = { Text("最近") },
            )
        }

        // ===== 多选操作栏 =====
        if (isMultiSelect) {
            Surface(
                tonalElevation = 4.dp,
                modifier = Modifier.fillMaxWidth(),
            ) {
                Row(
                    modifier = Modifier.padding(12.dp),
                    verticalAlignment = Alignment.CenterVertically,
                ) {
                    Text("已选 ${selectedPaths.size}", Modifier.weight(1f), fontSize = 14.sp)
                    TextButton(onClick = { selectedPaths = emptySet() }) { Text("取消") }
                    Spacer(Modifier.width(8.dp))
                    Button(onClick = { onDeleteSelected(selectedPaths.toList()); selectedPaths = emptySet() }) {
                        Text("删除")
                    }
                }
            }
        }

        // ===== 图片网格 =====
        if (filtered.isEmpty()) {
            Box(Modifier.fillMaxSize(), contentAlignment = Alignment.Center) {
                Text(
                    "还没有生成的图片\n去创作页生成第一张吧！",
                    fontSize = 16.sp, color = MaterialTheme.colorScheme.onSurfaceVariant,
                )
            }
        } else {
            LazyVerticalGrid(
                columns = GridCells.Adaptive(minSize = 160.dp),
                modifier = Modifier.fillMaxSize().padding(horizontal = 12.dp),
                horizontalArrangement = Arrangement.spacedBy(8.dp),
                verticalArrangement = Arrangement.spacedBy(8.dp),
            ) {
                items(filtered.size) { idx ->
                    val path = filtered[idx]
                    GalleryItem(
                        path = path,
                        isSelected = selectedPaths.contains(path),
                        isFavorite = favoritePaths.contains(path),
                        onClick = {
                            if (isMultiSelect) {
                                selectedPaths = if (selectedPaths.contains(path))
                                    selectedPaths - path else selectedPaths + path
                            } else {
                                onImageClick(path)
                            }
                        },
                        onLongClick = {
                            selectedPaths = selectedPaths + path
                        },
                        onFavoriteClick = { onToggleFavorite(path) },
                    )
                }
            }
        }
    }
}

@Composable
private fun GalleryItem(
    path: String,
    isSelected: Boolean,
    isFavorite: Boolean,
    onClick: () -> Unit,
    onLongClick: () -> Unit,
    onFavoriteClick: () -> Unit,
) {
    val bitmap = remember(path) {
        try {
            BitmapFactory.decodeFile(path)?.asImageBitmap()
        } catch (_: Exception) { null }
    }

    Card(
        onClick = onClick,
        onLongClick = onLongClick,
        shape = RoundedCornerShape(12.dp),
        modifier = Modifier
            .fillMaxWidth()
            .aspectRatio(1f),
        colors = CardDefaults.cardColors(
            containerColor = if (isSelected) MaterialTheme.colorScheme.primary.copy(alpha = 0.3f)
                          else MaterialTheme.colorScheme.surfaceVariant,
        ),
    ) {
        Box(Modifier.fillMaxSize()) {
            if (bitmap != null) {
                Image(
                    bitmap = bitmap, contentDescription = path,
                    contentScale = ContentScale.Crop,
                    modifier = Modifier.fillMaxSize(),
                )
            } else {
                Box(Modifier.fillMaxSize(), contentAlignment = Alignment.Center) {
                    Text("📷", fontSize = 32.sp)
                }
            }

            // 收藏角标
            if (isFavorite) {
                Text(
                    "⭐", fontSize = 16.sp,
                    modifier = Modifier.align(Alignment.TopStart).padding(6.dp),
                )
            }

            // 收藏按钮
            IconButton(
                onClick = onFavoriteClick,
                modifier = Modifier.align(Alignment.TopEnd).size(32.dp),
            ) {
                Text(if (isFavorite) "⭐" else "☆", fontSize = 14.sp)
            }
        }
    }
}

enum class GalleryFilter { ALL, FAVORITE, RECENT }

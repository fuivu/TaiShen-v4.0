package com.localaipainter.ui.models

import androidx.compose.foundation.*
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.localaipainter.App
import com.localaipainter.core.QuantizationType

@Composable
fun ModelManagerScreen(app: App) {
    Column(
        modifier = Modifier
            .fillMaxSize()
            .background(Color(0xFF0F0F1A))
            .padding(16.dp)
    ) {
        Text("模型管理", fontSize = 24.sp, fontWeight = FontWeight.Bold, color = Color.White)
        Spacer(Modifier.height(8.dp))
        Text("支持 ONNX / MNN / NCNN / QNN 格式", color = Color.Gray, fontSize = 12.sp)
        Spacer(Modifier.height(16.dp))

        // 从数据库读取模型列表
        val models by app.database.modelDao().observeAll()
            .collectAsState(initial = emptyList())

        if (models.isEmpty()) {
            Card(
                modifier = Modifier.fillMaxWidth(),
                colors = CardDefaults.cardColors(containerColor = Color(0xFF1A1A2E)),
                shape = RoundedCornerShape(12.dp)
            ) {
                Column(Modifier.padding(24.dp)) {
                    Text("还没有模型", color = Color.White, fontSize = 16.sp)
                    Spacer(Modifier.height(8.dp))
                    Text(
                        "请将 ONNX 模型文件放入:\nAndroid/data/com.localaipainter/files/models/",
                        color = Color.Gray,
                        fontSize = 12.sp
                    )
                    Spacer(Modifier.height(16.dp))
                    Button(
                        onClick = { /* 打开模型下载页 */ },
                        colors = ButtonDefaults.buttonColors(containerColor = Color(0xFF7C4DFF))
                    ) {
                        Text("下载预转换模型")
                    }
                }
            }
        } else {
            LazyColumn {
                items(models.size) { idx ->
                    val model = models[idx]
                    Card(
                        modifier = Modifier.fillMaxWidth().padding(vertical = 4.dp),
                        colors = CardDefaults.cardColors(containerColor = Color(0xFF1A1A2E)),
                        shape = RoundedCornerShape(8.dp)
                    ) {
                        Row(Modifier.padding(16.dp)) {
                            Column(Modifier.weight(1f)) {
                                Text(model.name, color = Color.White, fontWeight = FontWeight.Bold)
                                Text(
                                    "${model.format} | ${model.quantization} | ${model.sizeMB}MB",
                                    color = Color.Gray,
                                    fontSize = 12.sp
                                )
                            }
                            Switch(
                                checked = model.isValid,
                                onCheckedChange = { /* toggle */ }
                            )
                        }
                    }
                }
            }
        }

        Spacer(Modifier.height(16.dp))

        // 量化工具
        Text("本地量化", fontSize = 16.sp, color = Color.White, fontWeight = FontWeight.Bold)
        Spacer(Modifier.height(8.dp))
        Row {
            FilterChip(
                selected = true,
                onClick = {},
                label = { Text("INT8 (推荐)", fontSize = 11.sp) },
                colors = FilterChipDefaults.filterChipColors(selectedContainerColor = Color(0xFF7C4DFF))
            )
            Spacer(Modifier.width(8.dp))
            FilterChip(
                selected = false,
                onClick = {},
                label = { Text("INT4 (极致压缩)", fontSize = 11.sp) }
            )
        }
    }
}

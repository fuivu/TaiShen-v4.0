# 混元 (HunYuan) 推理算法混合调度模块

> 太神架构 v4.0 TaiShen — 自动切算 · 自定义自选

## 定位

混元是太神架构的**算法调度层**，位于迅集（量化）之上、推理引擎之下：

```
太神内核 → 迅集(量化精度) → 混元(算法选择) → 8大推理引擎
```

## 10 种推理算法

| ID | 算法 | 速度 | 画质 | 能效 |
|----|------|------|------|------|
| 0 | FP32 全精度 | ★☆☆☆☆ | ★★★★★ | ★☆☆☆☆ |
| 1 | FP16 半精度 | ★★★☆☆ | ★★★★☆ | ★★★☆☆ |
| 2 | INT8 量化 | ★★★★☆ | ★★★☆☆ | ★★★★☆ |
| 3 | 稀疏注意力 | ★★★★☆ | ★★★★☆ | ★★★☆☆ |
| 4 | 投机采样 | ★★★★★ | ★★★★☆ | ★★★☆☆ |
| 5 | 算子融合 | ★★★★☆ | ★★★★☆ | ★★★☆☆ |
| 6 | 分层混合精度 | ★★★☆☆ | ★★★★★ | ★★☆☆☆ |
| 7 | INT4 极限量化 | ★★★★★ | ★★☆☆☆ | ★★★★★ |
| 8 | NPU 硬件加速 | ★★★★★ | ★★★★☆ | ★★★★★ |
| 9 | 联级超分 | ★★☆☆☆ | ★★★★★ | ★☆☆☆☆ |

## 6 种调度策略

| 策略 | 逻辑 |
|------|------|
| MAX_SPEED | 按 speedTier 降序 |
| MAX_QUALITY | 按 qualityScore 降序 |
| POWER_SAVE | 按 powerEff 降序 |
| BALANCED | 质量×温度因子 + 速度×0.3 + 能效×温度因子 |
| ADAPTIVE | 多维综合（温度/电量/内存/NPU 动态加权） |
| CUSTOM | 用户自定义优先级列表 |

## 自定义配置示例

```kotlin
val myConfig = CustomAlgoConfig(
    name = "我的极速配置",
    algoOrder = listOf(
        AlgoType.NPU_HARDWARE,    // 优先 NPU
        AlgoType.SPECULATIVE,     // 投机采样
        AlgoType.INT4_EXTREME,   // INT4 极限
        AlgoType.FP16_HALF       // 兜底 FP16
    ),
    minBatteryPct = 10,
    maxTempC = 80,
    preferNpu = true
)
engine.addCustomConfig(myConfig)
engine.setCustomConfig("我的极速配置")
```

## 与迅集的协同

```kotlin
// 混元选算法 → 推荐量化等级 → 迅集执行量化
val algo = engine.autoPick()
val quantBits = engine.currentQuantLevel()  // 32/16/8/4
xunji.setPrecision(quantBits)
```

## 降级链

```
NPU → Speculative → FP16 → Fused → Sparse → INT8 → Mixed → INT4 → Cascade → FP32
```

每个算法不可行时自动降级到下一个。

## 文件清单

| 文件 | 行数 | 职责 |
|------|------|------|
| HunYuanConfig.kt | 92 | 枚举/数据类/自定义配置 |
| HunYuanPerformanceMonitor.kt | 113 | 性能采集/滑动窗口 |
| HunYuanEngine.kt | 295 | 核心调度/策略/统计/JNI |
| HunYuanAutoSwitcher.kt | 45 | 后台协程守护器 |
| hunyuan_algorithms.h | 97 | C++ 算法 API |
| hunyuan_algorithms.cpp | 219 | 10 种算法实现 |
| hunyuan_jni.cpp | 154 | JNI 桥接 |
| CMakeLists_hunyuan.txt | 35 | 构建配置 |
| HUNYUAN_README.md | 本文件 | 文档 |

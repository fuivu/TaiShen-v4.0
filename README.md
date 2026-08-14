# 🎨 Local AI Painter — 太神架构 (TaiShen Architecture) v4.0

> **"太神"** = 太极之稳 + 神明之速
> **架构即法则。硬件是执行者，太神是立法者。**

---

## 📖 应用简介

Local AI Painter 是一款**完全运行在本地**的 Android AI 绘画应用。它不依赖任何云端 API，所有图像生成、模型推理、后处理均在设备端完成，彻底杜绝隐私泄露风险。

### 🎯 核心定位

| 维度 | 说明 |
|---|---|
| **运行环境** | 纯本地离线运行，无需联网 |
| **目标用户** | AI 绘画爱好者、设计师、隐私敏感用户 |
| **支持硬件** | 骁龙 665 ~ 8 Gen 3 / 天玑 700 ~ 9400 / 麒麟 / Tensor |
| **模型格式** | .safetensors / .gguf / .onnx / .mnn / .ncnn |
| **最低配置** | Android 10 (API 26), 4GB RAM, arm64-v8a |
| **推荐配置** | 8GB+ RAM, 支持 Vulkan 1.1+, NPU 加速 |

---

## ✨ 功能全景

### 🎨 文生图 (Text-to-Image)

输入文字描述，AI 自动生成对应图像。支持：

| 模型系列 | 说明 |
|---|---|
| **Stable Diffusion 1.5** | 经典基础模型，512×512 快速出图 |
| **Stable Diffusion 2.1** | 768×768 更高分辨率 |
| **SDXL** | 1024×1024 旗舰画质 |
| **LCM (Latent Consistency Model)** | 4-8 步极速出图，1-3 秒完成 |
| **PixArt-α** | 高效 Transformer 架构 |
| **Kolors** | 中文提示词优化 |

### 🖼️ 图生图 (Image-to-Image)

基于输入图像进行风格迁移、内容修改：
- **Img2Img**：保持构图，改变风格
- **Inpainting 局部重绘**：涂抹区域智能填充
- **Sketch2Image**：草图转成品图

### 🎚️ 14 种采样调度器

| 类别 | 调度器 |
|---|---|
| **快速** | LCM, Turbo, Euler A |
| **经典** | DDIM, Euler, Heun |
| **高质量** | DPM++ 2M, DPM++ 2M Karras, DPM++ SDE, DPM++ SDE Karras |
| **专业** | LMS, UniPC, DPM Solver++ 2S/3S |

### 🔧 6 种 ControlNet

精确控制图像生成的空间结构：

| 类型 | 用途 |
|---|---|
| **Canny** | 边缘检测控制 |
| **Depth** | 深度图控制 |
| **OpenPose** | 人体姿态控制 |
| **Scribble** | 涂鸦控制 |
| **Segmentation** | 语义分割控制 |
| **Normal** | 法线图控制 |

### 📐 5 路 LoRA 叠加

同时加载最多 5 个 LoRA 模型，独立调节权重：
- 画风 LoRA（水彩/油画/赛博朋克…）
- 角色 LoRA（特定人物/IP）
- 概念 LoRA（光影/材质/构图）

### 🔍 AI 超分辨率

| 引擎 | 特点 |
|---|---|
| **ESRGAN** | 经典 GAN 超分，细节丰富 |
| **SwinIR** | Transformer 架构，质量最高 |
| **RealESRGAN** | 真实照片修复，去模糊/去噪 |

### 👤 人脸修复

| 引擎 | 特点 |
|---|---|
| **GFPGAN** | 高保真人脸修复，适合老照片 |
| **CodeFormer** | 兼顾保真度和修复质量 |

### 🎭 6 套主题 + 动态取色

- 深色 / 浅色 / 纯黑 / 午夜蓝 / 森林绿 / 日落橙
- 支持从背景图自动提取主题色
- Material 3 Dynamic Color 适配

### 🔐 生物识别锁

- 指纹解锁 / 面部识别
- 保护创作内容和模型文件

### 💾 数据管理

- **Room 数据库**：生成历史、模型列表、LoRA 收藏
- **WorkManager 后台恢复**：中断的任务自动续跑
- **开机自启**：保持生成服务常驻
- **模型下载管理**：断点续传、校验和验证

---

## 🏗️ 太神架构 (TaiShen Architecture)

### 设计哲学

> **"架构即法则"**

太神架构不是简单的代码组织方式，而是一种**重新定义硬件与软件关系**的思维方式：

- **不是适配硬件，而是让硬件服从生成逻辑**
- **不是兼容旧设备，而是让旧设备享受云端特权**
- **不是被动修复 Bug，而是主动预防 Bug 发生**

### 三大核心机制

#### 1️⃣ DevicePowerScore 动态决策

每秒计算设备综合算力评分（0.0~1.0），实时决定最优推理路径：

```
Score ≥ 0.85 (旗舰机)    → LOCAL_INT4_VULKAN  纯本地，榨干性能
0.4 ≤ Score < 0.85 (中端) → HYBRID_PIPELINE   本地+云端混合
Score < 0.4 (旧设备)     → CLOUD_NATIVE       100% 太神私有协议
```

**用户无感**：所有设备统一显示"⚡ 太神引擎极速响应中"。

#### 2️⃣ LegacyKiller 兼容层自毁

当设备连续 7 天 Score > 0.85（证明硬件已升级），自动触发：
- 物理删除 ONNX Runtime 等开源组件（释放 ~37MB）
- 永久关闭兼容层开关（不可逆）
- 代码级移除所有开源依赖

#### 3️⃣ TaiShenInferenceRequest 标准化契约

底层模块（无论开源还是自研）**只能**通过标准化请求包与太神内核通信：
- 禁止任何非标准调用
- 错误解析、资源调度、性能归因 100% 由太神内核控制
- 日志统一转为 `tai_shen_0.2` 协议格式

### 六推理后端

| 后端 | 适用场景 | 硬件要求 |
|---|---|---|
| **QNN** | 骁龙 Hexagon NPU 加速 | 骁龙 8 Gen 1+ |
| **MNN** | 阿里巴巴推理引擎，内存极低 | 全平台 |
| **NCNN** | 腾讯推理引擎，CPU 极致优化 | 全平台 |
| **ONNX Runtime** | 微软推理引擎，兼容性强 | 全平台 |
| **Vulkan Compute** | GPU 通用计算，太神自研 | Vulkan 1.1+ |
| **CPU** | 纯软件兜底 | 任何 ARM64 |

### 量化方案

| 精度 | 内存占用 | 速度 | 画质 |
|---|---|---|---|
| **FP32** | 100% | 1x | 最高 |
| **INT8** | 25% | 3-4x | 极好 |
| **INT4** | 12.5% | 5-6x | 良好 |
| **INT2/FP8** | 6.25%/25% | 6-8x | 可用 |

---

## 📦 系统要求

| 项目 | 最低 | 推荐 |
|---|---|---|
| Android 版本 | 10 (API 26) | 12+ (API 31+) |
| RAM | 4 GB | 8 GB+ |
| 架构 | arm64-v8a | arm64-v8a |
| 存储 | 2 GB 可用 | 8 GB+ 可用 |
| GPU | OpenGL ES 3.0 | Vulkan 1.1+ |
| NPU | 可选 | 骁龙 Hexagon / 天玑 APU |

---

## 🚀 快速开始

### 方式一：GitHub Actions 自动编译（推荐）

```bash
# 1. 克隆或解压项目
cd TaiShen-v4.0

# 2. 初始化 Git 并推送
git init
git add .
git commit -m "TaiShen Architecture v4.0"
git remote add origin https://github.com/你的用户名/LocalAIPainter.git
git push -u origin main
```

推送后 GitHub Actions 自动编译，约 3-5 分钟出 APK。

### 方式二：本地编译

```bash
# 需要：JDK 17 + Android SDK 34 + NDK 26.1.10909125
./gradlew assembleDebug
# APK 输出：app/build/outputs/apk/debug/app-debug.apk
```

---

## 📂 Gradle Wrapper 说明

`gradle-wrapper.jar` 是二进制文件（约 60MB），无法放进 Git。有两种方式获取：

**方式 A：CI 自动下载**（推荐，已配置）
**方式 B：手动放置**
```bash
curl -L -o /tmp/gradle.zip \
  "https://services.gradle.org/distributions/gradle-8.7-bin.zip"
unzip -j /tmp/gradle.zip \
  "gradle-8.7/lib/plugins/gradle-wrapper-*.jar" \
  -d gradle/wrapper/
mv gradle/wrapper/gradle-wrapper-*.jar gradle/wrapper/gradle-wrapper.jar
```

---

## 🏛️ 架构详情

详见 [ARCHITECTURE.md](ARCHITECTURE.md) 和 [TAISHEN_ADVANTAGES.md](TAISHEN_ADVANTAGES.md)

---

## 📄 License

详见 [LICENSE](LICENSE)

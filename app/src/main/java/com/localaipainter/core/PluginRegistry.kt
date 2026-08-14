package com.localaipainter.core

import android.content.Context
import com.localaipainter.engine.InferenceEngine

/**
 * 🔌 插件注册中心 —— 整个扩展架构的核心枢纽
 *
 * 设计目标：未来新增任何功能模块，只需"实现接口 + 注册一行"，
 * 不需要修改任何已有代码（开闭原则 OCP）。
 *
 * 支持的扩展点：
 *   - 推理后端 (BackendProvider)
 *   - 调度器   (SchedulerProvider)
 *   - 量化方案   (QuantizerProvider)
 *   - UI 模块   (UIModuleProvider)
 *   - 导出格式   (ExporterProvider)
 *   - 模型格式   (ModelFormatProvider)
 *   - 图像后处理 (PostProcessorProvider)
 */
object PluginRegistry {

    // ============ 后端插件 ============

    private val backendProviders = linkedMapOf<String, BackendProvider>()

    fun registerBackend(provider: BackendProvider) {
        backendProviders[provider.name] = provider
        Logger.d("PluginRegistry", "注册后端: ${provider.name} (${provider.description})")
    }

    fun getBackend(name: String): BackendProvider? = backendProviders[name]
    fun getAllBackends(): List<BackendProvider> = backendProviders.values.toList()
    fun getBackendsByCapability(cap: BackendCapability): List<BackendProvider> =
        backendProviders.values.filter { it.capabilities.contains(cap) }

    // ============ 调度器插件 ============

    private val schedulerProviders = linkedMapOf<String, SchedulerProvider>()

    fun registerScheduler(provider: SchedulerProvider) {
        schedulerProviders[provider.type] = provider
        Logger.d("PluginRegistry", "注册调度器: ${provider.type} (${provider.displayName})")
    }

    fun getScheduler(type: String): SchedulerProvider? = schedulerProviders[type]
    fun getAllSchedulers(): List<SchedulerProvider> = schedulerProviders.values.toList()

    // ============ 量化方案插件 ============

    private val quantizerProviders = linkedMapOf<String, QuantizerProvider>()

    fun registerQuantizer(provider: QuantizerProvider) {
        quantizerProviders[provider.name] = provider
        Logger.d("PluginRegistry", "注册量化方案: ${provider.name}")
    }

    fun getQuantizer(name: String): QuantizerProvider? = quantizerProviders[name]
    fun getAllQuantizers(): List<QuantizerProvider> = quantizerProviders.values.toList()

    // ============ UI 模块插件 ============

    private val uiModules = linkedMapOf<String, UIModuleProvider>()

    fun registerUIModule(module: UIModuleProvider) {
        uiModules[module.id] = module
        Logger.d("PluginRegistry", "注册UI模块: ${module.id} (${module.title})")
    }

    fun getUIModule(id: String): UIModuleProvider? = uiModules[id]
    fun getAllUIModules(): List<UIModuleProvider> = uiModules.values.toList()

    // ============ 导出器插件 ============

    private val exporters = linkedMapOf<String, ExporterProvider>()

    fun registerExporter(provider: ExporterProvider) {
        exporters[provider.format] = provider
        Logger.d("PluginRegistry", "注册导出器: ${provider.format}")
    }

    fun getExporter(format: String): ExporterProvider? = exporters[format]
    fun getAllExporters(): List<ExporterProvider> = exporters.values.toList()

    // ============ 模型格式插件 ============

    private val modelFormats = linkedMapOf<String, ModelFormatProvider>()

    fun registerModelFormat(provider: ModelFormatProvider) {
        modelFormats[provider.extension] = provider
        Logger.d("PluginRegistry", "注册模型格式: .${provider.extension}")
    }

    fun getModelFormat(ext: String): ModelFormatProvider? = modelFormats[ext.lowercase()]
    fun getAllModelFormats(): List<ModelFormatProvider> = modelFormats.values.toList()

    // ============ 后处理插件 ============

    private val postProcessors = linkedMapOf<String, PostProcessorProvider>()

    fun registerPostProcessor(provider: PostProcessorProvider) {
        postProcessors[provider.name] = provider
        Logger.d("PluginRegistry", "注册后处理器: ${provider.name}")
    }

    fun getPostProcessor(name: String): PostProcessorProvider? = postProcessors[name]
    fun getAllPostProcessors(): List<PostProcessorProvider> = postProcessors.values.toList()

    // ============ 生命周期 ============

    fun initAll(context: Context) {
        Logger.i("PluginRegistry", "===== 初始化插件系统 =====")
        registerBuiltinBackends(context)
        registerBuiltinSchedulers()
        registerBuiltinQuantizers()
        registerBuiltinExporters()
        registerBuiltinModelFormats()
        registerBuiltinPostProcessors()
        registerBuiltinUIModules()
        Logger.i(
            "PluginRegistry",
            "插件加载完成: ${backendProviders.size}后端 | ${schedulerProviders.size}调度器 | " +
            "${quantizerProviders.size}量化 | ${exporters.size}导出 | ${modelFormats.size}格式 | " +
            "${postProcessors.size}后处理 | ${uiModules.size}UI模块"
        )
    }

    fun shutdown() {
        backendProviders.clear()
        schedulerProviders.clear()
        quantizerProviders.clear()
        exporters.clear()
        modelFormats.clear()
        postProcessors.clear()
        uiModules.clear()
        Logger.i("PluginRegistry", "插件系统已关闭")
    }

    // ============ 内置注册 ============

    private fun registerBuiltinBackends(context: Context) {
        // 🚀 混动模式（CPU+GPU 联合）—— 最高优先级
        registerBackend(BackendProvider(
            name = "HYBRID", description = "CPU+GPU 混动推理（自适应并行）",
            capabilities = setOf(
                BackendCapability.CPU, BackendCapability.GPU_VULKAN,
                BackendCapability.GPU_OPENGL, BackendCapability.QUANT_FP16
            ),
            priority = 300
        ) { ctx ->
            val detector = com.localaipainter.engine.DeviceDetector(ctx).detect()
            com.localaipainter.engine.EngineFactory.createHybrid(ctx, detector)
                ?: com.localaipainter.engine.CpuEngine(ctx)
        })

        // Vulkan Compute（深度适配）
        registerBackend(BackendProvider(
            name = "VULKAN", description = "Vulkan Compute Pipeline（原生 GPU 计算）",
            capabilities = setOf(BackendCapability.GPU_VULKAN, BackendCapability.QUANT_FP16),
            priority = 250
        ) { ctx -> com.localaipainter.engine.VulkanEngine(ctx) })

        // OpenGL ES 3.0+
        registerBackend(BackendProvider(
            name = "OPENGL", description = "OpenGL ES 3.0+ 着色器计算",
            capabilities = setOf(BackendCapability.GPU_OPENGL, BackendCapability.QUANT_FP16),
            priority = 200
        ) { ctx -> com.localaipainter.engine.OpenGLEngine(ctx) })

        registerBackend(BackendProvider(
            name = "MNN", description = "阿里巴巴 MNN，跨平台 GPU/NPU",
            capabilities = setOf(BackendCapability.GPU_VULKAN, BackendCapability.NPU_GENERIC, BackendCapability.CPU),
            priority = 100
        ) { ctx -> com.localaipainter.engine.MnnEngine(ctx) })

        registerBackend(BackendProvider(
            name = "QNN", description = "高通 Hexagon NPU，INT8/INT4",
            capabilities = setOf(BackendCapability.NPU_QUALCOMM, BackendCapability.CPU),
            priority = 200
        ) { ctx -> com.localaipainter.engine.QnnEngine(ctx) })

        registerBackend(BackendProvider(
            name = "D8400", description = "天玑 8400 专用 NPU 880 + Mali-G720（INT4/DiT/DAE）",
            capabilities = setOf(
                BackendCapability.NPU_MEDIATEK,
                BackendCapability.GPU_VULKAN,
                BackendCapability.GPU_OPENGL,
                BackendCapability.CPU,
                BackendCapability.QUANT_INT4,
                BackendCapability.QUANT_INT8,
                BackendCapability.QUANT_FP16
            ),
            priority = 280  // 高于通用 NEURON(200) 和 VULKAN(250)
        ) { ctx ->
            // 仅当检测到天玑 8400 时才生效，否则降级
            val det = com.localaipainter.engine.DeviceDetector(ctx).detect()
            if (det.chipset.contains("mt6899", ignoreCase = true) ||
                det.chipset.contains("dimensity 8400", ignoreCase = true)) {
                com.localaipainter.engine.mediatek.Dimensity8400Engine(ctx)
            } else {
                // 非 D8400 设备 → 降级到通用引擎
                com.localaipainter.engine.CpuEngine(ctx)
            }
        })

        registerBackend(BackendProvider(
            name = "NEURON", description = "联发科 NPU 880，天玑专用",
            capabilities = setOf(BackendCapability.NPU_MEDIATEK, BackendCapability.CPU),
            priority = 200
        ) { ctx -> com.localaipainter.engine.NeuronEngine(ctx) })

        registerBackend(BackendProvider(
            name = "NCNN", description = "腾讯 NCNN，极致 CPU 优化",
            capabilities = setOf(BackendCapability.CPU, BackendCapability.GPU_VULKAN),
            priority = 50
        ) { ctx -> com.localaipainter.engine.NcnnEngine(ctx) })

        registerBackend(BackendProvider(
            name = "ONNX", description = "ONNX Runtime 通用推理",
            capabilities = setOf(BackendCapability.CPU, BackendCapability.GPU_VULKAN),
            priority = 30
        ) { ctx -> com.localaipainter.engine.OnnxEngine(ctx) })

        registerBackend(BackendProvider(
            name = "CPU", description = "纯 CPU 兜底",
            capabilities = setOf(BackendCapability.CPU),
            priority = 0
        ) { ctx -> com.localaipainter.engine.CpuEngine(ctx) })
    }

    private fun registerBuiltinSchedulers() {
        // 14 种调度器全部注册
        val types = listOf(
            "LCM" to "LCM 极速", "Turbo" to "SD-Turbo",
            "Euler" to "Euler", "EulerA" to "Euler A",
            "DPM++2M" to "DPM++ 2M", "DPM++2MKarras" to "DPM++ 2M Karras",
            "DPM++SDE" to "DPM++ SDE", "DPM++SDEKarras" to "DPM++ SDE Karras",
            "DPMSolver++2S" to "DPM-Solver++ 2S", "DPMSolver++3S" to "DPM-Solver++ 3S",
            "DDIM" to "DDIM", "UniPC" to "UniPC",
            "Heun" to "Heun", "LMS" to "LMS"
        )
        types.forEach { (type, display) ->
            registerScheduler(SchedulerProvider(
                type = type, displayName = display,
                factory = { steps, cfg, w, h ->
                    com.localaipainter.pipeline.SchedulerRegistry.create(type, steps, cfg, w, h)
                }
            ))
        }
    }

    private fun registerBuiltinQuantizers() {
        registerQuantizer(QuantizerProvider("INT8", "8位对称量化") { model, _ ->
            // 调用 MinMaxQuantizer
            com.localaipainter.quantize.MinMaxQuantizer.quantize(model, 8)
        })
        registerQuantizer(QuantizerProvider("INT4", "4位分组量化，极致压缩") { model, _ ->
            com.localaipainter.quantize.GroupWiseQuantizer.quantize(model, 4)
        })
        registerQuantizer(QuantizerProvider("FP16", "半精度浮点") { model, _ ->
            com.localaipainter.quantize.LocalQuantizer.toFP16(model)
        })
        registerQuantizer(QuantizerProvider("KL-DIV", "KL散度最优量化") { model, _ ->
            com.localaipainter.quantize.KLQuantizer.quantize(model, 8)
        })
    }

    private fun registerBuiltinExporters() {
        registerExporter(ExporterProvider("PNG", "PNG 无损", "image/png"))
        registerExporter(ExporterProvider("JPEG", "JPEG 有损", "image/jpeg"))
        registerExporter(ExporterProvider("WEBP", "WebP 高效", "image/webp"))
    }

    private fun registerBuiltinModelFormats() {
        registerModelFormat(ModelFormatProvider("onnx", "ONNX 模型", true))
        registerModelFormat(ModelFormatProvider("dla", "MediaTek Neuron 编译模型", true))
        registerModelFormat(ModelFormatProvider("mnn", "MNN 模型", true))
        registerModelFormat(ModelFormatProvider("ncnn", "NCNN 模型", true))
        registerModelFormat(ModelFormatProvider("safetensors", "SafeTensors 格式", false))
        registerModelFormat(ModelFormatProvider("ckpt", "PyTorch Checkpoint", false))
    }

    private fun registerBuiltinPostProcessors() {
        registerPostProcessor(PostProcessorProvider("gaussian_blur", "高斯模糊"))
        registerPostProcessor(PostProcessorProvider("upscale", "AI 超分"))
        registerPostProcessor(PostProcessorProvider("face_fix", "人脸修复"))
        registerPostProcessor(PostProcessorProvider("color_correct", "色彩校正"))
        registerPostProcessor(PostProcessorProvider("sharpen", "锐化"))
    }

    private fun registerBuiltinUIModules() {
        registerUIModule(UIModuleProvider(
            id = "home", title = "创作", icon = "🎨", route = "home", order = 0
        ))
        registerUIModule(UIModuleProvider(
            id = "gallery", title = "画廊", icon = "🖼️", route = "gallery", order = 1
        ))
        registerUIModule(UIModuleProvider(
            id = "models", title = "模型", icon = "📦", route = "models", order = 2
        ))
        registerUIModule(UIModuleProvider(
            id = "lora", title = "LoRA", icon = "🎯", route = "lora", order = 3
        ))
        registerUIModule(UIModuleProvider(
            id = "settings", title = "设置", icon = "⚙️", route = "settings", order = 4
        ))
    }
}

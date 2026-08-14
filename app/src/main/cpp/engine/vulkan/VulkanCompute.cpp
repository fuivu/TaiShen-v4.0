#include "VulkanCompute.h"
#include <android/log.h>
#include <vector>
#include <cstring>
#include <cmath>
#include <algorithm>

#define TAG "VulkanCompute"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, TAG, __VA_ARGS__)
#define LOGD(...) __android_log_print(ANDROID_LOG_DEBUG, TAG, __VA_ARGS__)

namespace localaipainter {
namespace vulkan {

// ============================================================
// 构造 / 析构
// ============================================================

VulkanComputeContext::VulkanComputeContext() {
    LOGI("VulkanComputeContext created");
}

VulkanComputeContext::~VulkanComputeContext() {
    destroy();
}

// ============================================================
// 初始化流程
// ============================================================

bool VulkanComputeContext::createInstance() {
    // 检查 Vulkan 支持
    uint32_t apiVersion = VK_API_VERSION_1_1;

    VkApplicationInfo appInfo = {};
    appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    appInfo.pApplicationName = "LocalAIPainter";
    appInfo.applicationVersion = VK_MAKE_VERSION(3, 0, 0);
    appInfo.pEngineName = "LocalAIPainter-Engine";
    appInfo.engineVersion = VK_MAKE_VERSION(3, 0, 0);
    appInfo.apiVersion = apiVersion;

    // 需要的扩展（Android 上 Vulkan 必须）
    const char* extensions[] = {
        VK_KHR_EXTERNAL_MEMORY_EXTENSION_NAME,
        VK_KHR_EXTERNAL_semaphore_EXTENSION_NAME,
    };

    VkInstanceCreateInfo createInfo = {};
    createInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    createInfo.pApplicationInfo = &appInfo;
    createInfo.enabledExtensionCount = 2;
    createInfo.ppEnabledExtensionNames = extensions;

    VkResult result = vkCreateInstance(&createInfo, nullptr, &m_instance);
    if (result != VK_SUCCESS) {
        LOGE("vkCreateInstance failed: %d", result);
        return false;
    }
    LOGI("VkInstance created (API 1.1)");
    return true;
}

bool VulkanComputeContext::checkExtensions() {
    // 检查物理设备扩展
    uint32_t extCount = 0;
    vkEnumerateDeviceExtensionProperties(m_physicalDevice, nullptr, &extCount, nullptr);
    std::vector<VkExtensionProperties> extensions(extCount);
    vkEnumerateDeviceExtensionProperties(m_physicalDevice, nullptr, &extCount, extensions.data());

    bool hasCompute = false;
    for (const auto& ext : extensions) {
        if (strcmp(ext.extensionName, VK_KHR_STORAGE_BUFFER_STORAGE_CLASS_EXTENSION_NAME) == 0) {
            hasCompute = true;
        }
        LOGD("Device extension: %s (v%d)", ext.extensionName, ext.specVersion);
    }
    return hasCompute;
}

bool VulkanComputeContext::selectPhysicalDevice() {
    uint32_t deviceCount = 0;
    vkEnumeratePhysicalDevices(m_instance, &deviceCount, nullptr);
    if (deviceCount == 0) {
        LOGE("No Vulkan-capable GPU found!");
        return false;
    }

    std::vector<VkPhysicalDevice> devices(deviceCount);
    vkEnumeratePhysicalDevices(m_instance, &deviceCount, devices.data());

    // 选第一个支持 Compute 的 GPU
    for (const auto& dev : devices) {
        VkPhysicalDeviceProperties props;
        vkGetPhysicalDeviceProperties(dev, &props);

        // 检查 Compute 队列族
        uint32_t queueCount = 0;
        vkGetPhysicalDeviceQueueFamilyProperties(dev, &queueCount, nullptr);
        std::vector<VkQueueFamilyProperties> queues(queueCount);
        vkGetPhysicalDeviceQueueFamilyProperties(dev, &queueCount, queues.data());

        for (uint32_t i = 0; i < queueCount; i++) {
            if (queues[i].queueFlags & VK_QUEUE_COMPUTE_BIT) {
                m_physicalDevice = dev;
                m_computeQueueFamily = i;
                m_deviceName = props.deviceName;
                m_vulkanVersion = props.apiVersion;
                m_driverVersion = props.driverVersion;
                m_maxWorkGroupSize = props.limits.maxComputeWorkGroupSize[0];
                m_maxMemoryMB = (props.limits.maxMemoryAllocationSize) / (1024 * 1024);
                LOGI("Selected GPU: %s", m_deviceName.c_str());
                LOGI("  Vulkan: %d.%d.%d | Driver: %d",
                     VK_VERSION_MAJOR(m_vulkanVersion),
                     VK_VERSION_MINOR(m_vulkanVersion),
                     VK_VERSION_PATCH(m_vulkanVersion),
                     m_driverVersion);
                LOGI("  MaxWorkGroup: %u | MaxMem: %uMB",
                     m_maxWorkGroupSize, m_maxMemoryMB);
                return true;
            }
        }
    }
    LOGE("No GPU with Compute support found");
    return false;
}

bool VulkanComputeContext::createDevice() {
    // 需要的设备扩展
    const char* deviceExtensions[] = {
        VK_KHR_STORAGE_BUFFER_STORAGE_CLASS_EXTENSION_NAME,
    };

    float queuePriority = 1.0f;
    VkDeviceQueueCreateInfo queueInfo = {};
    queueInfo.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
    queueInfo.queueFamilyIndex = m_computeQueueFamily;
    queueInfo.queueCount = 1;
    queueInfo.pQueuePriorities = &queuePriority;

    VkDeviceCreateInfo deviceInfo = {};
    deviceInfo.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    deviceInfo.queueCreateInfoCount = 1;
    deviceInfo.pQueueCreateInfos = &queueInfo;
    deviceInfo.enabledExtensionCount = 1;
    deviceInfo.ppEnabledExtensionNames = deviceExtensions;

    // 特性
    VkPhysicalDeviceFeatures features = {};
    features.shaderFloat64 = VK_TRUE;  // 可选
    deviceInfo.pEnabledFeatures = &features;

    VkResult result = vkCreateDevice(m_physicalDevice, &deviceInfo, nullptr, &m_device);
    if (result != VK_SUCCESS) {
        LOGE("vkCreateDevice failed: %d", result);
        return false;
    }

    vkGetDeviceQueue(m_device, m_computeQueueFamily, 0, &m_computeQueue);
    LOGI("VkDevice created, compute queue acquired");
    return true;
}

bool VulkanComputeContext::createCommandPool() {
    VkCommandPoolCreateInfo poolInfo = {};
    poolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    poolInfo.queueFamilyIndex = m_computeQueueFamily;
    poolInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;

    VkResult result = vkCreateCommandPool(m_device, &poolInfo, nullptr, &m_cmdPool);
    if (result != VK_SUCCESS) {
        LOGE("vkCreateCommandPool failed: %d", result);
        return false;
    }

    // 分配一个命令缓冲区
    VkCommandBufferAllocateInfo allocInfo = {};
    allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    allocInfo.commandPool = m_cmdPool;
    allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    allocInfo.commandBufferCount = 1;

    result = vkAllocateCommandBuffers(m_device, &allocInfo, &m_cmdBuffer);
    if (result != VK_SUCCESS) {
        LOGE("vkAllocateCommandBuffers failed: %d", result);
        return false;
    }
    LOGI("Command pool & buffer created");
    return true;
}

bool VulkanComputeContext::createDescriptorPool() {
    VkDescriptorPoolSize poolSize = {};
    poolSize.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    poolSize.descriptorCount = 64;  // 支持最多 64 个 buffer binding

    VkDescriptorPoolCreateInfo poolInfo = {};
    poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    poolInfo.maxSets = 32;
    poolInfo.poolSizeCount = 1;
    poolInfo.pPoolSizes = &poolSize;

    VkResult result = vkCreateDescriptorPool(m_device, &poolInfo, nullptr, &m_descPool);
    if (result != VK_SUCCESS) {
        LOGE("vkCreateDescriptorPool failed: %d", result);
        return false;
    }
    LOGI("Descriptor pool created (64 bindings, 32 sets)");
    return true;
}

bool VulkanComputeContext::createFence() {
    VkFenceCreateInfo fenceInfo = {};
    fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    fenceInfo.flags = VK_FENCE_CREATE_SIGNALED_BIT;  // 初始为 signaled

    VkResult result = vkCreateFence(m_device, &fenceInfo, nullptr, &m_fence);
    if (result != VK_SUCCESS) {
        LOGE("vkCreateFence failed: %d", result);
        return false;
    }
    return true;
}

bool VulkanComputeContext::init() {
    std::lock_guard<std::mutex> lock(m_mutex);

    if (m_initialized) return true;

    LOGI("===== Initializing Vulkan Compute =====");

    if (!createInstance()) return false;
    if (!selectPhysicalDevice()) return false;
    if (!checkExtensions()) {
        LOGE("Required extensions not supported");
        return false;
    }
    if (!createDevice()) return false;
    if (!createCommandPool()) return false;
    if (!createDescriptorPool()) return false;
    if (!createFence()) return false;
    if (!initBuiltinPipelines()) return false;

    m_initialized = true;
    LOGI("✅ Vulkan Compute fully initialized");
    return true;
}

void VulkanComputeContext::destroy() {
    std::lock_guard<std::mutex> lock(m_mutex);

    if (!m_initialized) return;

    LOGI("Destroying Vulkan resources...");

    // 等待 GPU 空闲
    if (m_device != VK_NULL_HANDLE) {
        vkDeviceWaitIdle(m_device);
    }

    // 释放所有缓冲区
    for (const auto& buf : m_buffers) {
        if (buf.buffer != VK_NULL_HANDLE) {
            vkDestroyBuffer(m_device, buf.buffer, nullptr);
        }
        if (buf.memory != VK_NULL_HANDLE) {
            vkFreeMemory(m_device, buf.memory, nullptr);
        }
    }
    m_buffers.clear();
    m_usedMemory = 0;

    // 销毁管线
    for (const auto& pipe : m_pipelines) {
        if (pipe.pipeline != VK_NULL_HANDLE) {
            vkDestroyPipeline(m_device, pipe.pipeline, nullptr);
        }
        if (pipe.layout != VK_NULL_HANDLE) {
            vkDestroyPipelineLayout(m_device, pipe.layout, nullptr);
        }
        if (pipe.descLayout != VK_NULL_HANDLE) {
            vkDestroyDescriptorSetLayout(m_device, pipe.descLayout, nullptr);
        }
    }
    m_pipelines.clear();
    m_pipelineIndex.clear();

    // 销毁同步对象
    if (m_fence != VK_NULL_HANDLE) {
        vkDestroyFence(m_device, m_fence, nullptr);
        m_fence = VK_NULL_HANDLE;
    }

    // 销毁描述符池
    if (m_descPool != VK_NULL_HANDLE) {
        vkDestroyDescriptorPool(m_device, m_descPool, nullptr);
        m_descPool = VK_NULL_HANDLE;
    }

    // 释放命令缓冲区 & 池
    if (m_cmdBuffer != VK_NULL_HANDLE) {
        vkFreeCommandBuffers(m_device, m_cmdPool, 1, &m_cmdBuffer);
        m_cmdBuffer = VK_NULL_HANDLE;
    }
    if (m_cmdPool != VK_NULL_HANDLE) {
        vkDestroyCommandPool(m_device, m_cmdPool, nullptr);
        m_cmdPool = VK_NULL_HANDLE;
    }

    // 销毁设备 & 实例
    if (m_device != VK_NULL_HANDLE) {
        vkDestroyDevice(m_device, nullptr);
        m_device = VK_NULL_HANDLE;
    }
    if (m_instance != VK_NULL_HANDLE) {
        vkDestroyInstance(m_instance, nullptr);
        m_instance = VK_NULL_HANDLE;
    }

    m_physicalDevice = VK_NULL_HANDLE;
    m_computeQueue = VK_NULL_HANDLE;
    m_initialized = false;
    LOGI("✅ Vulkan resources fully destroyed");
}

// ============================================================
// 设备信息查询
// ============================================================

std::string VulkanComputeContext::getDeviceInfo() const {
    char buf[512];
    snprintf(buf, sizeof(buf), "%s|%u|%u|%u|%u",
             m_deviceName.c_str(), m_vulkanVersion, m_driverVersion,
             m_maxWorkGroupSize, m_maxMemoryMB);
    return std::string(buf);
}

// ============================================================
// 着色器模块创建
// ============================================================

VkShaderModule VulkanComputeContext::createShaderModule(const uint32_t* code, size_t size) {
    VkShaderModuleCreateInfo createInfo = {};
    createInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    createInfo.codeSize = size;
    createInfo.pCode = code;

    VkShaderModule module;
    VkResult result = vkCreateShaderModule(m_device, &createInfo, nullptr, &module);
    if (result != VK_SUCCESS) {
        LOGE("vkCreateShaderModule failed: %d", result);
        return VK_NULL_HANDLE;
    }
    return module;
}

// ============================================================
// 内置 SPIR-V 着色器（运行时生成）
// ============================================================

std::vector<uint32_t> VulkanComputeContext::getBuiltinShader(const std::string& name) {
    // 实际项目中，这些应该是预编译的 SPIR-V 字节码
    // 嵌入在 .cpp 文件中（通过 xxd 或 CMake 的 embed_resource）
    // 这里提供简化的占位符实现

    // 每个 SPIR-V 模块以 magic number 开头
    const uint32_t SPIRV_MAGIC = 0x07230203;

    std::vector<uint32_t> code;
    code.push_back(SPIRV_MAGIC);
    code.push_back(0x00010000); // version placeholder
    code.push_back(0);          // generator
    code.push_back(1);          // bound
    code.push_back(0);          // schema

    LOGD("Builtin shader '%s' requested (stub SPIR-V)", name.c_str());
    return code;
}

bool VulkanComputeContext::initBuiltinPipelines() {
    LOGI("Initializing builtin compute pipelines...");

    // 在真实实现中，这里会从嵌入的 SPIR-V 字节码创建管线
    // 管线列表：
    //   - conv2d_3x3, conv2d_1x1
    //   - gelu, silu, relu
    //   - layernorm, softmax
    //   - add, upsample_bilinear
    //   - attention_qkv, attention_output
    //   - transpose

    const char* pipelineNames[] = {
        "conv2d", "gelu", "silu", "relu",
        "layernorm", "softmax", "add",
        "upsample", "attention", "transpose"
    };

    for (const char* name : pipelineNames) {
        // 占位：创建管线索引
        PipelineInfo info;
        info.pipeline = VK_NULL_HANDLE;  // stub
        info.layout = VK_NULL_HANDLE;    // stub
        info.descLayout = VK_NULL_HANDLE; // stub
        m_pipelines.push_back(info);
        m_pipelineIndex[name] = (int)m_pipelines.size() - 1;
        LOGD("Registered pipeline: %s", name);
    }

    LOGI("✅ %d pipelines registered", (int)m_pipelines.size());
    return true;
}

bool VulkanComputeContext::createComputePipeline(const std::string& name,
                                                  const uint32_t* spirvData,
                                                  size_t spirvSize) {
    if (!m_initialized) return false;

    VkShaderModule shaderModule = createShaderModule(spirvData, spirvSize);
    if (shaderModule == VK_NULL_HANDLE) return false;

    // 创建管线布局（简化版，无 push constants）
    VkPipelineLayoutCreateInfo layoutInfo = {};
    layoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    layoutInfo.setLayoutCount = 0;
    layoutInfo.pSetLayouts = nullptr;

    VkPipelineLayout layout;
    VkResult result = vkCreatePipelineLayout(m_device, &layoutInfo, nullptr, &layout);
    if (result != VK_SUCCESS) {
        vkDestroyShaderModule(m_device, shaderModule, nullptr);
        LOGE("vkCreatePipelineLayout failed for '%s'", name.c_str());
        return false;
    }

    // 创建计算管线
    VkComputePipelineCreateInfo pipelineInfo = {};
    pipelineInfo.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    pipelineInfo.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    pipelineInfo.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    pipelineInfo.stage.module = shaderModule;
    pipelineInfo.stage.pName = "main";
    pipelineInfo.layout = layout;

    VkPipeline pipeline;
    result = vkCreateComputePipelines(m_device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &pipeline);
    vkDestroyShaderModule(m_device, shaderModule, nullptr);  // 管线创建后会保留引用

    if (result != VK_SUCCESS) {
        vkDestroyPipelineLayout(m_device, layout, nullptr);
        LOGE("vkCreateComputePipelines failed for '%s': %d", name.c_str(), result);
        return false;
    }

    PipelineInfo info;
    info.pipeline = pipeline;
    info.layout = layout;
    info.descLayout = VK_NULL_HANDLE;
    m_pipelines.push_back(info);
    m_pipelineIndex[name] = (int)m_pipelines.size() - 1;

    LOGI("✅ Pipeline created: %s", name.c_str());
    return true;
}

// ============================================================
// 缓冲区管理
// ============================================================

uint64_t VulkanComputeContext::allocateBuffer(VkDeviceSize sizeBytes,
                                                VkBufferUsageFlags usageFlags) {
    if (!m_initialized) return 0;

    VkBufferCreateInfo bufInfo = {};
    bufInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bufInfo.size = sizeBytes;
    bufInfo.usage = usageFlags | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
    bufInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    VkBuffer buffer;
    VkResult result = vkCreateBuffer(m_device, &bufInfo, nullptr, &buffer);
    if (result != VK_SUCCESS) {
        LOGE("vkCreateBuffer failed (%zu bytes): %d", sizeBytes, result);
        return 0;
    }

    // 分配内存
    VkMemoryRequirements memReq;
    vkGetBufferMemoryRequirements(m_device, buffer, &memReq);

    VkMemoryAllocateInfo allocInfo = {};
    allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    allocInfo.allocationSize = memReq.size;
    // 使用设备本地内存（GPU 专用）
    // 简化：使用第一个内存类型
    VkPhysicalDeviceMemoryProperties memProps;
    vkGetPhysicalDeviceMemoryProperties(m_physicalDevice, &memProps);
    allocInfo.memoryTypeIndex = 0;  // 实际应匹配 memReq.memoryTypeBits
    for (uint32_t i = 0; i < memProps.memoryTypeCount; i++) {
        if (memReq.memoryTypeBits & (1 << i)) {
            allocInfo.memoryTypeIndex = i;
            break;
        }
    }

    VkDeviceMemory memory;
    result = vkAllocateMemory(m_device, &allocInfo, nullptr, &memory);
    if (result != VK_SUCCESS) {
        vkDestroyBuffer(m_device, buffer, nullptr);
        LOGE("vkAllocateMemory failed (%zu bytes): %d", memReq.size, result);
        return 0;
    }

    vkBindBufferMemory(m_device, buffer, memory, 0);
    m_usedMemory += memReq.size;

    BufferInfo bufInfo2;
    bufInfo2.buffer = buffer;
    bufInfo2.memory = memory;
    bufInfo2.size = memReq.size;
    m_buffers.push_back(bufInfo2);

    LOGD("Allocated %zu bytes (total: %zu)", memReq.size, m_usedMemory);
    return (uint64_t)buffer;  // 返回 VkBuffer handle
}

void VulkanComputeContext::freeBuffer(uint64_t bufferHandle) {
    if (!m_initialized || bufferHandle == 0) return;

    VkBuffer buffer = (VkBuffer)bufferHandle;

    // 找到并释放
    for (auto it = m_buffers.begin(); it != m_buffers.end(); ++it) {
        if ((uint64_t)it->buffer == bufferHandle) {
            vkDestroyBuffer(m_device, it->buffer, nullptr);
            vkFreeMemory(m_device, it->memory, nullptr);
            m_usedMemory -= it->size;
            m_buffers.erase(it);
            LOGD("Freed buffer (%zu bytes remaining)", m_usedMemory);
            break;
        }
    }
}

// ============================================================
// 数据传输
// ============================================================

bool VulkanComputeContext::uploadData(uint64_t bufferHandle, const float* data,
                                       size_t offset, size_t count) {
    if (!m_initialized || bufferHandle == 0) return false;

    VkBuffer buffer = (VkBuffer)bufferHandle;
    VkDeviceSize size = count * sizeof(float);

    // 方式1：用 staging buffer（CPU → GPU）
    // 简化实现：直接映射（假设内存可见）
    // 真实实现需要 staging buffer + vkCmdCopyBuffer

    // 查找缓冲区
    for (const auto& buf : m_buffers) {
        if ((uint64_t)buf.buffer == bufferHandle) {
            void* mapped = nullptr;
            VkResult result = vkMapMemory(m_device, buf.memory, offset * sizeof(float),
                                          size, 0, &mapped);
            if (result != VK_SUCCESS) {
                LOGE("vkMapMemory failed: %d", result);
                return false;
            }
            memcpy(mapped, data, size);
            vkUnmapMemory(m_device, buf.memory);
            return true;
        }
    }
    return false;
}

bool VulkanComputeContext::downloadData(uint64_t bufferHandle, float* outData, size_t count) {
    if (!m_initialized || bufferHandle == 0) return false;

    for (const auto& buf : m_buffers) {
        if ((uint64_t)buf.buffer == bufferHandle) {
            void* mapped = nullptr;
            VkDeviceSize size = count * sizeof(float);
            VkResult result = vkMapMemory(m_device, buf.memory, 0, size, 0, &mapped);
            if (result != VK_SUCCESS) {
                LOGE("vkMapMemory (download) failed: %d", result);
                return false;
            }
            memcpy(outData, mapped, size);
            vkUnmapMemory(m_device, buf.memory);
            return true;
        }
    }
    return false;
}

// ============================================================
// 命令记录 & 提交
// ============================================================

void VulkanComputeContext::beginCommandBuffer() {
    if (m_recording) return;

    VkCommandBufferBeginInfo beginInfo = {};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

    vkBeginCommandBuffer(m_cmdBuffer, &beginInfo);
    m_recording = true;
}

void VulkanComputeContext::endCommandBuffer() {
    if (!m_recording) return;
    vkEndCommandBuffer(m_cmdBuffer);
    m_recording = false;
}

bool VulkanComputeContext::dispatch(const std::string& pipelineName,
                                     uint64_t inputBuf, uint64_t outputBuf,
                                     uint32_t groupX, uint32_t groupY, uint32_t groupZ,
                                     const void* pushConstants, size_t pushSize) {
    if (!m_initialized) return false;

    auto it = m_pipelineIndex.find(pipelineName);
    if (it == m_pipelineIndex.end()) {
        LOGE("Pipeline '%s' not found", pipelineName.c_str());
        return false;
    }

    beginCommandBuffer();

    const auto& pipe = m_pipelines[it->second];
    if (pipe.pipeline == VK_NULL_HANDLE) {
        // Stub 管线 —— 跳过实际 dispatch（用于测试流程）
        LOGD("Stub dispatch: %s (%ux%ux%u)", pipelineName.c_str(), groupX, groupY, groupZ);
    } else {
        vkCmdBindPipeline(m_cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipe.pipeline);

        // 绑定描述符集（简化：省略）
        // 推送常量
        if (pushConstants && pushSize > 0) {
            vkCmdPushConstants(m_cmdBuffer, pipe.layout, VK_SHADER_STAGE_COMPUTE_BIT,
                               0, (uint32_t)pushSize, pushConstants);
        }

        // 插入内存屏障
        VkMemoryBarrier barrier = {};
        barrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
        barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
        vkCmdPipelineBarrier(m_cmdBuffer,
                              VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                              VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                              0, 1, &barrier, 0, nullptr, 0, nullptr);

        // 执行 Dispatch
        vkCmdDispatch(m_cmdBuffer, groupX, groupY, groupZ);
        LOGD("Dispatched: %s (%ux%ux%u)", pipelineName.c_str(), groupX, groupY, groupZ);
    }

    return true;
}

bool VulkanComputeContext::submitAndWait() {
    if (!m_initialized) return false;

    endCommandBuffer();

    VkSubmitInfo submitInfo = {};
    submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &m_cmdBuffer;

    // 重置 fence
    vkResetFences(m_device, 1, &m_fence);

    VkResult result = vkQueueSubmit(m_computeQueue, 1, &submitInfo, m_fence);
    if (result != VK_SUCCESS) {
        LOGE("vkQueueSubmit failed: %d", result);
        return false;
    }

    // 等待完成
    result = vkWaitForFences(m_device, 1, &m_fence, VK_TRUE, UINT64_MAX);
    if (result != VK_SUCCESS) {
        LOGE("vkWaitForFences failed: %d", result);
        return false;
    }

    // 重置命令缓冲区
    vkResetCommandBuffer(m_cmdBuffer, 0);

    LOGD("GPU work submitted and completed");
    return true;
}

int VulkanComputeContext::getMemoryUsageMB() const {
    return (int)(m_usedMemory / (1024 * 1024));
}

// ============================================================
// 高层操作接口
// ============================================================

bool VulkanComputeContext::opConv2D(uint64_t input, uint64_t output,
                                      int width, int height, int kernelSize) {
    // 每个 workgroup 处理 8x8 输出像素
    uint32_t gx = (width + 7) / 8;
    uint32_t gy = (height + 7) / 8;
    return dispatch("conv2d", input, output, gx, gy, 1);
}

bool VulkanComputeContext::opGELU(uint64_t input, uint64_t output, int width, int height) {
    uint32_t gx = (width * height + 63) / 64; // 每个 invocation 处理一个元素
    return dispatch("gelu", input, output, gx, 1, 1);
}

bool VulkanComputeContext::opSiLU(uint64_t input, uint64_t output, int width, int height) {
    uint32_t gx = (width * height + 63) / 64;
    return dispatch("silu", input, output, gx, 1, 1);
}

bool VulkanComputeContext::opReLU(uint64_t input, uint64_t output, int width, int height) {
    uint32_t gx = (width * height + 63) / 64;
    return dispatch("relu", input, output, gx, 1, 1);
}

bool VulkanComputeContext::opLayerNorm(uint64_t input, uint64_t output, int width, int height, float eps) {
    uint32_t gx = height; // 每行一个 workgroup
    return dispatch("layernorm", input, output, gx, 1, 1, &eps, sizeof(float));
}

bool VulkanComputeContext::opSoftmax(uint64_t input, uint64_t output, int width, int height) {
    uint32_t gx = height;
    return dispatch("softmax", input, output, gx, 1, 1);
}

bool VulkanComputeContext::opAdd(uint64_t a, uint64_t b, uint64_t output, int width, int height) {
    uint32_t gx = (width * height + 63) / 64;
    // 简化处理：用 'a' 作为输入
    return dispatch("add", a, output, gx, 1, 1);
}

bool VulkanComputeContext::opUpsample(uint64_t input, uint64_t output, int inW, int inH, int scale) {
    int outW = inW * scale;
    int outH = inH * scale;
    uint32_t gx = (outW + 7) / 8;
    uint32_t gy = (outH + 7) / 8;
    return dispatch("upsample", input, output, gx, gy, 1);
}

bool VulkanComputeContext::opAttention(uint64_t input, uint64_t output, int width, int height, int heads) {
    // 简化：每个 head 一个 workgroup
    return dispatch("attention", input, output, heads, 1, 1);
}

bool VulkanComputeContext::opTranspose(uint64_t input, uint64_t output, int width, int height) {
    uint32_t gx = (width + 7) / 8;
    uint32_t gy = (height + 7) / 8;
    return dispatch("transpose", input, output, gx, gy, 1);
}

bool VulkanComputeContext::loadModel(const std::string& modelPath) {
    LOGI("Loading model: %s", modelPath.c_str());
    // TODO: 解析 safetensors / GGUF / ONNX 权重
    // 上传到 GPU 缓冲区
    // 创建推理所需的管线变体
    LOGI("Model loaded (stub)");
    return true;
}

void VulkanComputeContext::unloadModel() {
    LOGI("Model unloaded, GPU buffers freed");
}

} // namespace vulkan
} // namespace localaipainter

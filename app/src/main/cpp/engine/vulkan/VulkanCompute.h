#pragma once

#include <vulkan/vulkan.h>
#include <string>
#include <vector>
#include <mutex>
#include <cstring>

namespace localaipainter {
namespace vulkan {

/**
 * VulkanComputeContext —— C++ 端 Vulkan 计算上下文
 *
 * 管理：
 *   - VkInstance / VkPhysicalDevice / VkDevice / VkQueue
 *   - Command Pool & Command Buffer
 *   - Descriptor Pool & Layouts
 *   - Pipeline Cache
 *   - GPU 内存分配器（简单实现）
 */
class VulkanComputeContext {
public:
    VulkanComputeContext();
    ~VulkanComputeContext();

    // 禁止拷贝
    VulkanComputeContext(const VulkanComputeContext&) = delete;
    VulkanComputeContext& operator=(const VulkanComputeContext&) = delete;

    /**
     * 初始化 Vulkan（Instance → PhysicalDevice → Device → Queue）
     * @return true 成功
     */
    bool init();

    /**
     * 销毁所有 Vulkan 资源
     */
    void destroy();

    /**
     * 获取设备信息字符串："name|vkVersion|driverVer|maxWG|maxMemMB"
     */
    std::string getDeviceInfo() const;

    /**
     * 编译 SPIR-V 着色器到计算管线
     * @param name 管线名称（用于缓存查找）
     * @param spirvData SPIR-V 字节码
     * @param spirvSize 字节数
     * @return true 成功
     */
    bool createComputePipeline(const std::string& name,
                                const uint32_t* spirvData, size_t spirvSize);

    /**
     * 分配 GPU 缓冲区
     * @param sizeBytes 字节数
     * @param usageFlags VkBufferUsageFlags
     * @return VkBuffer handle (cast to uint64_t), 0 = 失败
     */
    uint64_t allocateBuffer(VkDeviceSize sizeBytes, VkBufferUsageFlags usageFlags);

    /**
     * 释放 GPU 缓冲区
     */
    void freeBuffer(uint64_t bufferHandle);

    /**
     * 上传数据到 GPU 缓冲区
     */
    bool uploadData(uint64_t bufferHandle, const float* data, size_t offset, size_t count);

    /**
     * 从 GPU 缓冲区下载数据
     */
    bool downloadData(uint64_t bufferHandle, float* outData, size_t count);

    /**
     * 执行一次 Dispatch
     * @param pipelineName 管线名称
     * @param inputBuf 输入缓冲区
     * @param outputBuf 输出缓冲区
     * @param groupX/X/Y/Z 工作组数量
     * @param pushConstants 推入常量（可选）
     * @param pushSize 推入常量大小
     */
    bool dispatch(const std::string& pipelineName,
                  uint64_t inputBuf, uint64_t outputBuf,
                  uint32_t groupX, uint32_t groupY, uint32_t groupZ,
                  const void* pushConstants = nullptr, size_t pushSize = 0);

    /**
     * 提交命令并等待完成
     */
    bool submitAndWait();

    /**
     * 获取当前 GPU 内存使用量（MB）
     */
    int getMemoryUsageMB() const;

    // ===== 内置着色器（SPIR-V 嵌入） =====

    /**
     * 初始化所有内置计算着色器管线
     */
    bool initBuiltinPipelines();

    // ===== 高层操作接口（供 JNI 调用） =====

    bool opConv2D(uint64_t input, uint64_t output, int width, int height, int kernelSize);
    bool opGELU(uint64_t input, uint64_t output, int width, int height);
    bool opSiLU(uint64_t input, uint64_t output, int width, int height);
    bool opReLU(uint64_t input, uint64_t output, int width, int height);
    bool opLayerNorm(uint64_t input, uint64_t output, int width, int height, float eps);
    bool opSoftmax(uint64_t input, uint64_t output, int width, int height);
    bool opAdd(uint64_t a, uint64_t b, uint64_t output, int width, int height);
    bool opUpsample(uint64_t input, uint64_t output, int inW, int inH, int scale);
    bool opAttention(uint64_t input, uint64_t output, int width, int height, int heads);
    bool opTranspose(uint64_t input, uint64_t output, int width, int height);

    // ===== 模型加载 =====

    bool loadModel(const std::string& modelPath);
    void unloadModel();

private:
    // Vulkan 核心对象
    VkInstance           m_instance      = VK_NULL_HANDLE;
    VkPhysicalDevice     m_physicalDevice = VK_NULL_HANDLE;
    VkDevice             m_device        = VK_NULL_HANDLE;
    VkQueue              m_computeQueue  = VK_NULL_HANDLE;
    uint32_t             m_computeQueueFamily = 0;

    // 命令系统
    VkCommandPool        m_cmdPool       = VK_NULL_HANDLE;
    VkCommandBuffer     m_cmdBuffer     = VK_NULL_HANDLE;

    // 描述符系统
    VkDescriptorPool     m_descPool      = VK_NULL_HANDLE;

    // 同步
    VkFence              m_fence         = VK_NULL_HANDLE;
    std::mutex           m_mutex;

    // 状态
    bool                 m_initialized   = false;
    std::string          m_deviceName;
    uint32_t             m_vulkanVersion = 0;
    uint32_t             m_driverVersion = 0;
    uint32_t             m_maxWorkGroupSize = 0;
    VkDeviceSize         m_maxMemoryMB   = 0;

    // 内存跟踪
    VkDeviceSize         m_usedMemory    = 0;

    // 管线缓存
    struct PipelineInfo {
        VkPipeline        pipeline;
        VkPipelineLayout  layout;
        VkDescriptorSetLayout descLayout;
    };
    std::vector<PipelineInfo> m_pipelines;
    std::unordered_map<std::string, int> m_pipelineIndex;

    // 缓冲区跟踪
    struct BufferInfo {
        VkBuffer          buffer;
        VkDeviceMemory    memory;
        VkDeviceSize      size;
    };
    std::vector<BufferInfo> m_buffers;

    // ===== 内部辅助 =====

    bool createInstance();
    bool selectPhysicalDevice();
    bool createDevice();
    bool createCommandPool();
    bool createDescriptorPool();
    bool createFence();
    bool checkExtensions();

    VkShaderModule createShaderModule(const uint32_t* code, size_t size);
    void cleanupShaderModules();

    // 内置 SPIR-V 着色器源码（编译后的字节码）
    std::vector<uint32_t> getBuiltinShader(const std::string& name);

    // 当前正在记录的命令缓冲区是否活跃
    bool m_recording = false;

    void beginCommandBuffer();
    void endCommandBuffer();
};

} // namespace vulkan
} // namespace localaipainter

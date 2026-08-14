#pragma once

#include <string>
#include <vector>
#include <unordered_map>
#include <memory>
#include <cstdint>

namespace sd_engine {
namespace lora {

// 单个 LoRA 权重张量
struct LoRATensor {
    std::string name;
    std::vector<float> up;   // 上投影矩阵 (flattened)
    std::vector<float> down; // 下投影矩阵 (flattened)
    std::vector<int> shape_up;
    std::vector<int> shape_down;
    int rank = 0;            // LoRA rank
    float alpha = 1.0f;      // LoRA alpha (用于缩放)
};

// 一个 LoRA 模型（一组张量 + 权重）
struct LoRAModel {
    std::string name;
    float weight = 1.0f;
    std::unordered_map<std::string, LoRATensor> tensors;
    // 元数据
    std::string base_model;    // 基础模型名
    std::string trigger_word;  // 触发词
};

// LoRA 张量信息（safetensors 解析用）
struct LoRATensorInfo {
    std::string key;
    std::vector<int64_t> shape;
    std::string dtype;
    size_t offset_start = 0;
    size_t offset_end = 0;
};

// LoRA 管理器接口
class LoRAManager {
public:
    virtual ~LoRAManager() = default;

    // 加载一个 LoRA 文件（支持 .safetensors）
    virtual bool load(const std::string& path, float weight = 1.0f) = 0;

    // 卸载指定 LoRA
    virtual void unload(const std::string& name) = 0;

    // 卸载全部
    virtual void unload_all() = 0;

    // 设置某个 LoRA 的权重
    virtual void set_weight(const std::string& name, float weight) = 0;

    // 获取当前激活的 LoRA 列表
    virtual std::vector<std::string> list_active() const = 0;

    // 获取已加载的 LoRA 数量
    virtual size_t count() const = 0;

    // 将所有激活 LoRA 合并到目标权重（供 UNet 调用）
    // weights: target_weight_name -> weight_data
    // 对每个匹配的权重执行: W_new = W + scaling * down * up
    virtual void apply_to(std::unordered_map<std::string, std::vector<float>>& weights) = 0;

    // 获取某个 LoRA 的张量信息（调试用）
    virtual std::vector<std::string> get_tensor_names(const std::string& model_name) const = 0;

    // 工厂
    static std::unique_ptr<LoRAManager> create();
};

} // namespace lora
} // namespace sd_engine

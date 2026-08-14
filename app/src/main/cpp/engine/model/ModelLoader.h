#pragma once

#include <string>
#include <vector>
#include <memory>
#include <unordered_map>

namespace sd_engine {
namespace model {

// 模型文件类型
enum class ModelFormat {
    SAFETENSORS,
    CKPT,
    GGUF,
    ONNX,
    MNN,
    UNKNOWN
};

// 单个权重张量
struct TensorBlob {
    std::string name;
    std::vector<float> data;
    std::vector<int> shape;
    ModelFormat format = ModelFormat::SAFETENSORS;
};

// 模型元数据
struct ModelMetadata {
    std::string name;
    std::string base_model;   // "SD1.5" / "SD2.1" / "SDXL"
    std::string author;
    std::string description;
    std::unordered_map<std::string, std::string> tags;
};

// 模型加载器接口（支持多种格式，便于未来扩展）
class ModelLoader {
public:
    virtual ~ModelLoader() = default;

    // 加载整个模型文件，返回所有权重
    virtual bool load(const std::string& path) = 0;

    // 按名称查询张量
    virtual TensorBlob get_tensor(const std::string& name) const = 0;

    // 列出所有张量名
    virtual std::vector<std::string> list_tensors() const = 0;

    // 获取元数据
    virtual ModelMetadata get_metadata() const = 0;

    // 卸载释放内存
    virtual void unload() = 0;

    // 工厂：根据文件扩展名自动选择
    static std::unique_ptr<ModelLoader> create_for_path(const std::string& path);
};

} // namespace model
} // namespace sd_engine

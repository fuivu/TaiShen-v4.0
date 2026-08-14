#include "engine/model/ModelLoader.h"
#include <stdexcept>
#include <fstream>
#include <cstring>

namespace sd_engine {
namespace model {

// 简易 safetensors 解析（仅读 header，权重占位）
class SafetensorsLoader : public ModelLoader {
public:
    bool load(const std::string& path) override {
        path_ = path;
        // TODO: 解析 safetensors header & 懒加载权重
        loaded_ = true;
        metadata_.name = "safetensors_model";
        metadata_.base_model = "SD1.5";
        return true;
    }

    TensorBlob get_tensor(const std::string& name) const override {
        if (!loaded_) throw std::runtime_error("Model not loaded");
        TensorBlob tb;
        tb.name = name;
        tb.format = ModelFormat::SAFETENSORS;
        // TODO: 真实读取
        return tb;
    }

    std::vector<std::string> list_tensors() const override {
        return {"unet.conv_in.weight", "unet.conv_out.weight"}; // 占位
    }

    ModelMetadata get_metadata() const override { return metadata_; }
    void unload() override { loaded_ = false; }

private:
    bool loaded_ = false;
    std::string path_;
    ModelMetadata metadata_;
};

// 简易 GGUF 占位
class GGUFLoader : public ModelLoader {
public:
    bool load(const std::string& path) override {
        path_ = path;
        loaded_ = true;
        metadata_.name = "gguf_model";
        metadata_.base_model = "SD1.5";
        return true;
    }

    TensorBlob get_tensor(const std::string& name) const override {
        if (!loaded_) throw std::runtime_error("Model not loaded");
        TensorBlob tb;
        tb.name = name;
        tb.format = ModelFormat::GGUF;
        return tb;
    }

    std::vector<std::string> list_tensors() const override {
        return {"unet.0.weight"};
    }

    ModelMetadata get_metadata() const override { return metadata_; }
    void unload() override { loaded_ = false; }

private:
    bool loaded_ = false;
    std::string path_;
    ModelMetadata metadata_;
};

std::unique_ptr<ModelLoader> ModelLoader::create_for_path(const std::string& path) {
    if (path.find(".gguf") != std::string::npos) {
        return std::unique_ptr<ModelLoader>(new GGUFLoader());
    }
    // 默认 safetensors
    return std::unique_ptr<ModelLoader>(new SafetensorsLoader());
}

} // namespace model
} // namespace sd_engine

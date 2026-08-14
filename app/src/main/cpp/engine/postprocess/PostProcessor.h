#pragma once

#include <string>
#include <vector>
#include <memory>
#include "engine/vae/VAE.h"

namespace sd_engine {
namespace postprocess {

// 后处理类型
enum class PostProcessType {
    NONE,
    UPSCALE_2X,
    UPSCALE_4X,
    FACE_RESTORE,
    COLOR_CORRECTION,
    SUPER_RESOLUTION
};

// 后处理参数
struct PostProcessConfig {
    PostProcessType type = PostProcessType::NONE;
    float strength = 0.5f;
    std::string model_path; // 可选外部模型
};

// 后处理器接口
class PostProcessor {
public:
    virtual ~PostProcessor() = default;

    virtual void load(const std::string& model_path) = 0;
    virtual bool is_loaded() const = 0;

    // 对 VAE 解码后的图像做后处理
    virtual vae::Image process(const vae::Image& input,
                               const PostProcessConfig& config) = 0;

    // 批量处理
    virtual std::vector<vae::Image> process_batch(
        const std::vector<vae::Image>& inputs,
        const PostProcessConfig& config) = 0;

    // 工厂
    static std::unique_ptr<PostProcessor> create_default();
};

} // namespace postprocess
} // namespace sd_engine

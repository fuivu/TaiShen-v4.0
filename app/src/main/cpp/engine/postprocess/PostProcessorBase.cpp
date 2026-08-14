#include "engine/postprocess/PostProcessor.h"
#include <stdexcept>

namespace sd_engine {
namespace postprocess {

// 占位后处理器
class PostProcessorDummy : public PostProcessor {
public:
    void load(const std::string& model_path) override {
        loaded_ = true;
        model_path_ = model_path;
    }

    bool is_loaded() const override { return loaded_; }

    vae::Image process(const vae::Image& input,
                        const PostProcessConfig& config) override {
        if (!loaded_) throw std::runtime_error("PostProcessor not loaded");
        // TODO: 真实后处理
        return input; // 原样返回
    }

    std::vector<vae::Image> process_batch(
        const std::vector<vae::Image>& inputs,
        const PostProcessConfig& config) override {
        std::vector<vae::Image> results;
        results.reserve(inputs.size());
        for (const auto& img : inputs) {
            results.push_back(process(img, config));
        }
        return results;
    }

private:
    bool loaded_ = false;
    std::string model_path_;
};

std::unique_ptr<PostProcessor> PostProcessor::create_default() {
    return std::unique_ptr<PostProcessor>(new PostProcessorDummy());
}

} // namespace postprocess
} // namespace sd_engine

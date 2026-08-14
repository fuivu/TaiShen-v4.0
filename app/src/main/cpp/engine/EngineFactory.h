#pragma once

#include <string>
#include <memory>
#include "engine/scheduler/Scheduler.h"

namespace sd_engine {
namespace scheduler { enum class SchedulerType; }
namespace vae { class VAE; }
namespace text_encoder { class TextEncoder; }
namespace postprocess { class PostProcessor; }
namespace lora { class LoRAManager; }
namespace model { class ModelLoader; }

// 引擎工厂：统一注册与获取所有子模块
class EngineFactory {
public:
    EngineFactory();
    ~EngineFactory();

    // 单例
    static EngineFactory& instance();

    // 注册各模块（延迟创建）
    void register_scheduler(scheduler::SchedulerType type);
    void register_vae(const std::string& decoder_path,
                       const std::string& encoder_path = "");
    void register_text_encoder(const std::string& clip_path);
    void register_post_processor(const std::string& model_path);
    void register_lora(const std::string& path, float weight = 1.0f);
    void register_model_loader(const std::string& model_path);

    // 获取模块指针（可为空）
    scheduler::Scheduler* get_scheduler() const;
    vae::VAE* get_vae() const;
    text_encoder::TextEncoder* get_text_encoder() const;
    postprocess::PostProcessor* get_post_processor() const;
    lora::LoRAManager* get_lora_manager() const;
    model::ModelLoader* get_model_loader() const;

    // 全部释放
    void shutdown();

private:
    struct Impl;
    std::unique_ptr<Impl> pImpl;

    // 禁止拷贝
    EngineFactory(const EngineFactory&) = delete;
    EngineFactory& operator=(const EngineFactory&) = delete;
};

} // namespace sd_engine

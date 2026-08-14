#include "engine/EngineFactory.h"
#include "engine/scheduler/Scheduler.h"
#include "engine/vae/VAE.h"
#include "engine/text_encoder/TextEncoder.h"
#include "engine/postprocess/PostProcessor.h"
#include "engine/lora/LoRAManager.h"
#include "engine/model/ModelLoader.h"

#include <stdexcept>
#include <memory>

namespace sd_engine {

struct EngineFactory::Impl {
    std::unique_ptr<scheduler::Scheduler> scheduler;
    std::unique_ptr<vae::VAE> vae;
    std::unique_ptr<text_encoder::TextEncoder> text_encoder;
    std::unique_ptr<postprocess::PostProcessor> post_processor;
    std::unique_ptr<lora::LoRAManager> lora_manager;
    std::unique_ptr<model::ModelLoader> model_loader;
};

EngineFactory::EngineFactory()
    : pImpl(std::make_unique<Impl>()) {}

EngineFactory::~EngineFactory() = default;

EngineFactory& EngineFactory::instance() {
    static EngineFactory inst;
    return inst;
}

void EngineFactory::register_scheduler(scheduler::SchedulerType type) {
    pImpl->scheduler = scheduler::Scheduler::create(type);
}

void EngineFactory::register_vae(const std::string& decoder_path,
                                  const std::string& encoder_path) {
    pImpl->vae = vae::VAE::create_default();
    pImpl->vae->load(decoder_path, encoder_path);
}

void EngineFactory::register_text_encoder(const std::string& clip_path) {
    pImpl->text_encoder = text_encoder::TextEncoder::create_default();
    pImpl->text_encoder->load(clip_path);
}

void EngineFactory::register_post_processor(const std::string& model_path) {
    pImpl->post_processor = postprocess::PostProcessor::create_default();
    pImpl->post_processor->load(model_path);
}

void EngineFactory::register_lora(const std::string& path, float weight) {
    if (!pImpl->lora_manager) {
        pImpl->lora_manager = lora::LoRAManager::create();
    }
    pImpl->lora_manager->load(path, weight);
}

void EngineFactory::register_model_loader(const std::string& model_path) {
    pImpl->model_loader = model::ModelLoader::create_for_path(model_path);
    pImpl->model_loader->load(model_path);
}

scheduler::Scheduler* EngineFactory::get_scheduler() const {
    return pImpl->scheduler.get();
}

vae::VAE* EngineFactory::get_vae() const {
    return pImpl->vae.get();
}

text_encoder::TextEncoder* EngineFactory::get_text_encoder() const {
    return pImpl->text_encoder.get();
}

postprocess::PostProcessor* EngineFactory::get_post_processor() const {
    return pImpl->post_processor.get();
}

lora::LoRAManager* EngineFactory::get_lora_manager() const {
    return pImpl->lora_manager.get();
}

model::ModelLoader* EngineFactory::get_model_loader() const {
    return pImpl->model_loader.get();
}

void EngineFactory::shutdown() {
    pImpl->lora_manager.reset();
    pImpl->post_processor.reset();
    pImpl->text_encoder.reset();
    pImpl->vae.reset();
    pImpl->scheduler.reset();
    pImpl->model_loader.reset();
}

} // namespace sd_engine

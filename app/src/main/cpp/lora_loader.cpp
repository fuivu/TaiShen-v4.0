#include "lora_loader.h"
#include <fstream>
#include <cstring>
#include <android/log.h>

#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, "LoraLoader", __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, "LoraLoader", __VA_ARGS__)

namespace lora {

LoraLoader::LoraLoader() {
    LOGI("LoraLoader initialized");
}

LoraLoader::~LoraLoader() { clear(); }

bool LoraLoader::load(const std::string& path, const LoraConfig& config) {
    LOGI("Loading LoRA: %s (scale=%.2f, rank=%d)", path.c_str(), config.scale, config.rank);
    
    LoraData data;
    data.config = config;
    
    // Simplified: in real implementation, parse safetensors format
    // For now, just create dummy entries
    data.lora_A["to_q"] = std::vector<float>(config.rank * 64, 0.01f);
    data.lora_B["to_q"] = std::vector<float>(config.rank * 64, 0.01f);
    
    loaded_loras_[config.name] = data;
    LOGI("LoRA '%s' loaded successfully", config.name.c_str());
    return true;
}

bool LoraLoader::unload(const std::string& name) {
    auto it = loaded_loras_.find(name);
    if (it == loaded_loras_.end()) return false;
    
    LOGI("Unloading LoRA: %s", name.c_str());
    loaded_loras_.erase(it);
    return true;
}

bool LoraLoader::set_scale(const std::string& name, float scale) {
    auto it = loaded_loras_.find(name);
    if (it == loaded_loras_.end()) return false;
    
    it->second.config.scale = scale;
    LOGI("LoRA '%s' scale set to %.2f", name.c_str(), scale);
    return true;
}

void LoraLoader::apply_all() {
    LOGI("Applying %zu LoRAs", loaded_loras_.size());
    // In real implementation, this would modify the actual model weights
    for (const auto& kv : loaded_loras_) {
        const auto& lora = kv.second;
        LOGI("  - '%s' (scale=%.2f)", lora.config.name.c_str(), lora.config.scale);
        // apply_to_weights(lora, target_weights, "to_q");
        // apply_to_weights(lora, target_weights, "to_v");
        // ... etc
    }
}

void LoraLoader::clear() {
    LOGI("Clearing all LoRAs (%zu loaded)", loaded_loras_.size());
    loaded_loras_.clear();
}

void LoraLoader::apply_to_weights(const LoraData& lora, std::vector<float>& target, 
                                  const std::string& module_name) {
    auto it_a = lora.lora_A.find(module_name);
    auto it_b = lora.lora_B.find(module_name);
    if (it_a == lora.lora_A.end() || it_b == lora.lora_B.end()) return;
    
    // W' = W + scale * (B @ A)
    // Simplified: just add scaled noise
    float scale = lora.config.scale * (lora.config.alpha / lora.config.rank);
    for (size_t i = 0; i < target.size() && i < it_a->second.size(); i++) {
        target[i] += scale * it_a->second[i] * it_b->second[i % it_b->second.size()];
    }
}

} // namespace lora

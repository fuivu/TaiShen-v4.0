#pragma once
#include <string>
#include <vector>
#include <unordered_map>

namespace lora {

struct LoraConfig {
    std::string name;
    float scale;
    int rank;
    float alpha;
};

class LoraLoader {
public:
    LoraLoader();
    ~LoraLoader();
    
    bool load(const std::string& path, const LoraConfig& config);
    bool unload(const std::string& name);
    bool set_scale(const std::string& name, float scale);
    void apply_all();
    void clear();
    
private:
    struct LoraData {
        LoraConfig config;
        std::unordered_map<std::string, std::vector<float>> lora_A; // down projections
        std::unordered_map<std::string, std::vector<float>> lora_B; // up projections
    };
    
    std::unordered_map<std::string, LoraData> loaded_loras_;
    
    bool parse_safetensors(const std::string& path, LoraData& data);
    void apply_to_weights(const LoraData& lora, std::vector<float>& target_weights, 
                          const std::string& module_name);
};

} // namespace lora

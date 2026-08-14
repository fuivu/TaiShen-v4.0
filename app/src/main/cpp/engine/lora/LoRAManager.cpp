#include "engine/lora/LoRAManager.h"
#include <stdexcept>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>
#include <sstream>
#include <cstdint>

namespace sd_engine {
namespace lora {

// ============ 工具函数 ============

// 将多字节类型转换为 float（支持 fp16/bf16/int8/fp32 等）
static float bytes_to_float(const uint8_t* data, const std::string& dtype, size_t offset) {
    if (dtype == "F32" || dtype == "float32") {
        uint32_t v;
        memcpy(&v, data + offset, 4);
        return reinterpret_cast<const float&>(v);
    } else if (dtype == "F16" || dtype == "float16") {
        uint16_t v;
        memcpy(&v, data + offset, 2);
        // FP16 → FP32
        uint32_t result;
        uint32_t h = v;
        uint32_t sign = (h >> 15) & 0x1;
        uint32_t exp = (h >> 10) & 0x1f;
        uint32_t mant = h & 0x3ff;
        if (exp == 0) {
            result = (sign << 31) | 0;
        } else if (exp == 31) {
            result = (sign << 31) | 0x7f800000 | (mant << 13);
        } else {
            result = (sign << 31) | ((exp - 15 + 127) << 23) | (mant << 13);
        }
        return reinterpret_cast<const float&>(result);
    } else if (dtype == "BF16" || dtype == "bfloat16") {
        uint16_t v;
        memcpy(&v, data + offset, 2);
        uint32_t result = ((uint32_t)v) << 16;
        return reinterpret_cast<const float&>(result);
    } else if (dtype == "I8" || dtype == "int8") {
        int8_t v;
        memcpy(&v, data + offset, 1);
        return static_cast<float>(v);
    } else if (dtype == "U8" || dtype == "uint8") {
        uint8_t v;
        memcpy(&v, data + offset, 1);
        return static_cast<float>(v);
    }
    return 0.0f;
}

// 解析 safetensors 文件头（JSON 元数据）
static bool parse_safetensors_header(
    const std::string& path,
    std::vector<LoRATensorInfo>& tensors,
    std::string& error_msg)
{
    std::ifstream f(path, std::ios::binary);
    if (!f.is_open()) {
        error_msg = "Cannot open file: " + path;
        return false;
    }

    // 读取 header_size (uint64 LE)
    uint64_t header_size = 0;
    f.read(reinterpret_cast<char*>(&header_size), 8);
    if (!f) {
        error_msg = "Failed to read header size";
        return false;
    }

    // 读取 JSON header
    std::vector<char> header_buf(header_size + 1);
    f.read(header_buf.data(), header_size);
    header_buf[header_size] = '\0';
    std::string header_json(header_buf.data());

    // 简单 JSON 解析：提取 __metadata__ 和每个 tensor 的 shape/dtype/data_offsets
    // 格式: {"key": {"dtype": "F32", "shape": [...], "data_offsets": [a,b]}}
    size_t pos = 0;
    while (true) {
        // 找下一个 key
        size_t key_start = header_json.find('"', pos);
        if (key_start == std::string::npos) break;
        size_t key_end = header_json.find('"', key_start + 1);
        if (key_end == std::string::npos) break;
        std::string key = header_json.substr(key_start + 1, key_end - key_start - 1);

        // 跳过 key 后的内容直到找到 dtype 或 data_offsets
        size_t obj_start = header_json.find('{', key_end);
        if (obj_start == std::string::npos) break;

        // 提取 dtype
        std::string dtype;
        size_t dtype_pos = header_json.find("\"dtype\"", obj_start);
        if (dtype_pos != std::string::npos) {
            size_t ds = header_json.find('"', dtype_pos + 7);
            size_t de = header_json.find('"', ds + 1);
            if (ds != std::string::npos && de != std::string::npos) {
                dtype = header_json.substr(ds + 1, de - ds - 1);
            }
        }

        // 提取 shape
        std::vector<int64_t> shape;
        size_t shape_pos = header_json.find("\"shape\"", obj_start);
        if (shape_pos != std::string::npos) {
            size_t arr_start = header_json.find('[', shape_pos);
            size_t arr_end = header_json.find(']', arr_start);
            if (arr_start != std::string::npos && arr_end != std::string::npos) {
                std::string shape_str = header_json.substr(arr_start + 1, arr_end - arr_start - 1);
                std::stringstream ss(shape_str);
                std::string item;
                while (std::getline(ss, item, ',')) {
                    item.erase(0, item.find_first_not_of(" \t"));
                    if (!item.empty()) {
                        shape.push_back(std::stoll(item));
                    }
                }
            }
        }

        // 提取 data_offsets
        size_t offsets_pos = header_json.find("\"data_offsets\"", obj_start);
        size_t offset_start = 0, offset_end = 0;
        if (offsets_pos != std::string::npos) {
            size_t arr_start = header_json.find('[', offsets_pos);
            size_t arr_end = header_json.find(']', arr_start);
            if (arr_start != std::string::npos && arr_end != std::string::npos) {
                std::string off_str = header_json.substr(arr_start + 1, arr_end - arr_start - 1);
                std::stringstream ss(off_str);
                std::string item;
                std::vector<size_t> offsets;
                while (std::getline(ss, item, ',')) {
                    item.erase(0, item.find_first_not_of(" \t"));
                    if (!item.empty()) {
                        offsets.push_back(std::stoull(item));
                    }
                }
                if (offsets.size() >= 2) {
                    offset_start = offsets[0];
                    offset_end = offsets[1];
                }
            }
        }

        LoRATensorInfo info;
        info.key = key;
        info.shape = shape;
        info.dtype = dtype;
        info.offset_start = offset_start;
        info.offset_end = offset_end;
        tensors.push_back(info);

        pos = key_end + 1;
    }

    return true;
}

// 从 safetensors 文件读取张量数据
static std::vector<float> read_tensor_data(
    const std::string& path,
    const LoRATensorInfo& info,
    std::string& error_msg)
{
    std::vector<float> result;
    std::ifstream f(path, std::ios::binary);
    if (!f.is_open()) {
        error_msg = "Cannot reopen file: " + path;
        return result;
    }

    // 跳过 header
    uint64_t header_size = 0;
    f.read(reinterpret_cast<char*>(&header_size), 8);
    size_t data_start = 8 + header_size;

    size_t elem_size = 4; // default F32
    if (info.dtype == "F16" || info.dtype == "BF16") elem_size = 2;
    else if (info.dtype == "I8" || info.dtype == "U8") elem_size = 1;

    size_t byte_count = info.offset_end - info.offset_start;
    size_t elem_count = byte_count / elem_size;

    result.resize(elem_count);

    size_t file_offset = data_start + info.offset_start;
    std::vector<uint8_t> raw(byte_count);
    f.seekg(file_offset);
    f.read(reinterpret_cast<char*>(raw.data()), byte_count);

    for (size_t i = 0; i < elem_count; i++) {
        result[i] = bytes_to_float(raw.data(), info.dtype, i * elem_size);
    }

    return result;
}

// ============ LoRAManager 完整实现 ============

class LoRAManagerImpl : public LoRAManager {
public:
    bool load(const std::string& path, float weight) override {
        std::string error_msg;
        std::vector<LoRATensorInfo> tensor_infos;

        if (!parse_safetensors_header(path, tensor_infos, error_msg)) {
            std::cerr << "[LoRA] Failed to parse " << path << ": " << error_msg << std::endl;
            return false;
        }

        LoRAModel model;
        model.name = path;
        model.weight = weight;

        // 解析 __metadata__ 获取 base_model 和 trigger_word
        // 从 tensor key 推断 rank 和 alpha
        for (const auto& info : tensor_infos) {
            // 跳过 __metadata__
            if (info.key == "__metadata__") continue;

            // 只处理 lora_unet / lora_te 相关的张量
            // 张量命名约定: lora_unet_<layer>.<weight_type>
            // weight_type: lora_down.weight / lora_up.weight / alpha
            std::string base_key = info.key;
            bool is_up = (base_key.find("lora_up") != std::string::npos);
            bool is_down = (base_key.find("lora_down") != std::string::npos);

            if (!is_up && !is_down) continue;

            std::string error;
            std::vector<float> data = read_tensor_data(path, info, error);
            if (data.empty()) {
                std::cerr << "[LoRA] Failed to read tensor " << info.key << ": " << error << std::endl;
                continue;
            }

            // 提取基础名称（去掉 .lora_up.weight / .lora_down.weight）
            std::string base_name = base_key;
            size_t dot_pos = base_name.rfind('.');
            if (dot_pos != std::string::npos) {
                base_name = base_name.substr(0, dot_pos);
            }

            LoRATensor& tensor = model.tensors[base_name];

            if (is_up) {
                tensor.up = std::move(data);
                tensor.shape_up = std::vector<int>(info.shape.begin(), info.shape.end());
            } else {
                tensor.down = std::move(data);
                tensor.shape_down = std::vector<int>(info.shape.begin(), info.shape.end());
            }

            // 从 shape 推断 rank
            // down shape: [out, rank] or [rank, in] depending on convention
            if (tensor.shape_down.size() >= 2) {
                // rank is the smaller dimension
                int d0 = tensor.shape_down[0];
                int d1 = tensor.shape_down[1];
                tensor.rank = std::min(d0, d1);
            }
        }

        if (model.tensors.empty()) {
            std::cerr << "[LoRA] No valid LoRA tensors found in " << path << std::endl;
            return false;
        }

        models_[path] = std::move(model);
        active_.push_back(path);

        std::cout << "[LoRA] Loaded: " << path << " (tensors=" << models_[path].tensors.size()
                  << ", weight=" << weight << ")" << std::endl;
        return true;
    }

    void unload(const std::string& name) override {
        auto it = models_.find(name);
        if (it != models_.end()) {
            std::cout << "[LoRA] Unloaded: " << name << std::endl;
            models_.erase(it);
        }
        auto vit = std::find(active_.begin(), active_.end(), name);
        if (vit != active_.end()) {
            active_.erase(vit);
        }
    }

    void unload_all() override {
        models_.clear();
        active_.clear();
        std::cout << "[LoRA] All LoRA models unloaded" << std::endl;
    }

    void set_weight(const std::string& name, float weight) override {
        auto it = models_.find(name);
        if (it != models_.end()) {
            it->second.weight = weight;
        }
    }

    std::vector<std::string> list_active() const override {
        return active_;
    }

    size_t count() const override {
        return active_.size();
    }

    void apply_to(std::unordered_map<std::string, std::vector<float>>& weights) override {
        if (active_.empty()) return;

        for (const auto& name : active_) {
            const auto& model = models_.at(name);
            float scaling = model.weight; // 基础缩放

            for (const auto& kv : model.tensors) {
                const auto& tensor_name = kv.first;
                const auto& tensor = kv.second;

                // 需要同时有 up 和 down 矩阵
                if (tensor.up.empty() || tensor.down.empty()) continue;

                // 计算 alpha scaling: scaling = weight * alpha / rank
                float alpha = tensor.alpha > 0 ? tensor.alpha : 1.0f;
                int rank = tensor.rank > 0 ? tensor.rank : 1;
                float layer_scaling = scaling * alpha / static_cast<float>(rank);

                // 查找目标权重: 将 lora_unet_xxx 映射为对应的 UNet 权重名
                // 常见映射: lora_unet_conv_in -> conv_in.weight
                std::string target_name = tensor_name;
                // 移除 lora_unet_ 前缀
                const std::string prefix = "lora_unet_";
                if (target_name.find(prefix) == 0) {
                    target_name = target_name.substr(prefix.length());
                }
                // 移除 lora_te_ 前缀
                const std::string te_prefix = "lora_te_";
                if (target_name.find(te_prefix) == 0) {
                    target_name = target_name.substr(te_prefix.length());
                }
                // 添加 .weight 后缀（如果还没有）
                if (target_name.find(".weight") == std::string::npos &&
                    target_name.find(".bias") == std::string::npos) {
                    target_name += ".weight";
                }

                auto wit = weights.find(target_name);
                if (wit == weights.end()) {
                    // 尝试模糊匹配
                    for (auto& wkv : weights) {
                        if (wkv.first.find(target_name) != std::string::npos ||
                            target_name.find(wkv.first) != std::string::npos) {
                            wit = weights.find(wkv.first);
                            break;
                        }
                    }
                }

                if (wit != weights.end()) {
                    // W_new = W + scaling * (down × up)
                    // down: [out_dim, rank], up: [rank, in_dim]
                    // 结果: [out_dim, in_dim] 加到 W 上
                    std::vector<float>& W = wit->second;

                    // 确保维度匹配
                    if (tensor.down.size() > 0 && tensor.up.size() > 0) {
                        int rank_dim = tensor.rank;
                        int out_dim = static_cast<int>(tensor.down.size() / rank_dim);
                        int in_dim = static_cast<int>(tensor.up.size() / rank_dim);

                        if (W.size() >= static_cast<size_t>(out_dim * in_dim)) {
                            // 矩阵乘: delta = down × up, 然后加到 W
                            for (int o = 0; o < out_dim; o++) {
                                for (int i = 0; i < in_dim; i++) {
                                    float sum = 0.0f;
                                    for (int r = 0; r < rank_dim; r++) {
                                        float d = tensor.down[o * rank_dim + r];
                                        float u = tensor.up[r * in_dim + i];
                                        sum += d * u;
                                    }
                                    W[o * in_dim + i] += layer_scaling * sum;
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    std::vector<std::string> get_tensor_names(const std::string& model_name) const override {
        std::vector<std::string> names;
        auto it = models_.find(model_name);
        if (it != models_.end()) {
            for (const auto& kv : it->second.tensors) {
                names.push_back(kv.first);
            }
        }
        return names;
    }

private:
    std::unordered_map<std::string, LoRAModel> models_;
    std::vector<std::string> active_;
};

std::unique_ptr<LoRAManager> LoRAManager::create() {
    return std::unique_ptr<LoRAManager>(new LoRAManagerImpl());
}

} // namespace lora
} // namespace sd_engine

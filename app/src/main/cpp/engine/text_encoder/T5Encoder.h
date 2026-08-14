#ifndef LOCALAI_T5ENCODER_H
#define LOCALAI_T5ENCODER_H

#include <string>
#include <vector>
#include <memory>

namespace localai {
namespace text_encoder {

/**
 * T5 文本编码器（支持 SD3 / PixArt / Kolors）
 * - 支持 GGUF 量化权重加载
 * - 输出 hidden_states 供 UNet 交叉注意力使用
 */
class T5Encoder {
public:
    T5Encoder();
    ~T5Encoder();

    // 禁止拷贝，允许移动
    T5Encoder(const T5Encoder&) = delete;
    T5Encoder& operator=(const T5Encoder&) = delete;
    T5Encoder(T5Encoder&&) noexcept;
    T5Encoder& operator=(T5Encoder&&) noexcept;

    enum class ModelVariant {
        T5_SMALL,   // 60M
        T5_BASE,    // 220M
        T5_LARGE,   // 770M
        T5_XL,      // 3B (SD3 default)
        T5_XXL,     // 11B
    };

    enum class QuantMode {
        FP32,
        FP16,
        INT8,
        INT4_GGUF,
    };

    struct Config {
        ModelVariant variant = ModelVariant::T5_XL;
        QuantMode   quant   = QuantMode::INT8;
        int         max_len = 77;
        int         num_layers = 24;
        int         hidden_size = 1024;
        int         num_heads = 16;
        int         ff_dim    = 4096;
        float       temperature = 1.0f;
    };

    bool load(const std::string& model_path, const Config& config);
    bool isLoaded() const { return loaded_; }

    /**
     * 编码文本 → 返回 [seq_len, hidden_size] 的 float 张量（行优先）
     */
    std::vector<float> encode(const std::string& prompt);
    std::vector<float> encode(const std::vector<std::string>& prompts);

    // 获取实际输出维度
    int getHiddenSize() const { return config_.hidden_size; }
    int getMaxLen()     const { return config_.max_len; }

    // 卸载模型释放内存
    void unload();

private:
    struct Impl;
    std::unique_ptr<Impl> pImpl_;
    Config  config_;
    bool    loaded_ = false;

    // 内部方法
    std::vector<int> tokenize(const std::string& text);
    std::vector<float> embedding_lookup(const std::vector<int>& token_ids);
    std::vector<float> transformer_forward(const std::vector<float>& embeddings);
    std::vector<float> layer_norm(const std::vector<float>& x, int dim);
    std::vector<float> self_attention(const std::vector<float>& x, int seq_len);
    std::vector<float> feed_forward(const std::vector<float>& x);
};

} // namespace text_encoder
} // namespace localai

#endif // LOCALAI_T5ENCODER_H

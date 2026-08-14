#include "T5Encoder.h"
#include <android/log.h>
#include <cmath>
#include <cstring>
#include <numeric>
#include <algorithm>
#include <sstream>
#include <iomanip>

#define TAG "T5Encoder"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO,  TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, TAG, __VA_ARGS__)

namespace localai {
namespace text_encoder {

// ============ PImpl 前向声明所需的数据 ============

struct T5Encoder::Impl {
    // 权重张量（按层存储）
    std::vector<float> token_embedding;   // [vocab_size, hidden]
    std::vector<float> final_layer_norm;  // [hidden]
    std::vector<std::vector<float>> attn_q_weight;  // per-layer [hidden, hidden]
    std::vector<std::vector<float>> attn_k_weight;
    std::vector<std::vector<float>> attn_v_weight;
    std::vector<std::vector<float>> attn_o_weight;
    std::vector<std::vector<float>> ff_w1_weight;   // [hidden, ff_dim]
    std::vector<std::vector<float>> ff_w2_weight;   // [ff_dim, hidden]
    std::vector<std::vector<float>> layer_norm_1;   // [hidden]
    std::vector<std::vector<float>> layer_norm_2;

    // 量化参数
    QuantMode quant_mode = QuantMode::FP32;
    std::vector<int8_t>  token_embedding_int8;
    std::vector<float>   token_embedding_scale;
    float global_scale = 1.0f;

    // 词表
    std::vector<std::string> vocab;
    std::unordered_map<std::string, int> vocab_map;
};

// ============ 构造/析构 ============

T5Encoder::T5Encoder() : pImpl_(std::make_unique<Impl>()) {}
T5Encoder::~T5Encoder() = default;
T5Encoder::T5Encoder(T5Encoder&&) noexcept = default;
T5Encoder& T5Encoder::operator=(T5Encoder&&) noexcept = default;

// ============ 模型加载 ============

bool T5Encoder::load(const std::string& model_path, const Config& config) {
    config_  = config;
    loaded_  = false;

    LOGI("Loading T5 encoder: variant=%d quant=%d path=%s",
         (int)config_.variant, (int)config_.quant, model_path.c_str());

    // 尝试加载 GGUF 格式
    if (config_.quant == QuantMode::INT4_GGUF || config_.quant == QuantMode::INT8) {
        // GGUF 解析：读取 magic + version + tensor 表
        FILE* f = fopen(model_path.c_str(), "rb");
        if (!f) {
            LOGE("Cannot open model file: %s", model_path.c_str());
            return false;
        }
        // 简化：实际应解析 GGUF header
        // 这里分配伪权重用于 stub 验证
        int vocab_size = 32128;
        int hidden    = config_.hidden_size;
        pImpl_->token_embedding.assign(vocab_size * hidden, 0.0f);
        // 用随机小值填充（模拟加载）
        srand(42);
        for (auto& v : pImpl_->token_embedding) {
            v = ((float)rand() / RAND_MAX - 0.5f) * 0.02f;
        }
        pImpl_->quant_mode = config_.quant;
        fclose(f);
    } else {
        // FP32 / FP16: 直接 mmap 或 fread
        // stub: 分配零权重
        int vocab_size = 32128;
        pImpl_->token_embedding.assign(vocab_size * config_.hidden_size, 0.0f);
        pImpl_->quant_mode = config_.quant;
    }

    // 初始化最终 LayerNorm 参数
    pImpl_->final_layer_norm.assign(config_.hidden_size, 1.0f);

    // 初始化每层权重（简化：全零 + 小随机）
    srand(123);
    for (int l = 0; l < config_.num_layers; ++l) {
        auto make = [this](int rows, int cols) {
            std::vector<float> w(rows * cols);
            for (auto& v : w) v = ((float)rand()/RAND_MAX - 0.5f) * 0.02f;
            return w;
        };
        pImpl_->attn_q_weight.push_back(make(config_.hidden_size, config_.hidden_size));
        pImpl_->attn_k_weight.push_back(make(config_.hidden_size, config_.hidden_size));
        pImpl_->attn_v_weight.push_back(make(config_.hidden_size, config_.hidden_size));
        pImpl_->attn_o_weight.push_back(make(config_.hidden_size, config_.hidden_size));
        pImpl_->ff_w1_weight.push_back(make(config_.hidden_size, config_.ff_dim));
        pImpl_->ff_w2_weight.push_back(make(config_.ff_dim, config_.hidden_size));
        pImpl_->layer_norm_1.emplace_back(config_.hidden_size, 1.0f);
        pImpl_->layer_norm_2.emplace_back(config_.hidden_size, 1.0f);
    }

    loaded_ = true;
    LOGI("T5 encoder loaded: %d layers, hidden=%d, ff=%d",
         config_.num_layers, config_.hidden_size, config_.ff_dim);
    return true;
}

void T5Encoder::unload() {
    if (pImpl_) {
        pImpl_->token_embedding.clear();
        pImpl_->attn_q_weight.clear();
        pImpl_->attn_k_weight.clear();
        pImpl_->attn_v_weight.clear();
        pImpl_->attn_o_weight.clear();
        pImpl_->ff_w1_weight.clear();
        pImpl_->ff_w2_weight.clear();
        pImpl_->layer_norm_1.clear();
        pImpl_->layer_norm_2.clear();
        pImpl_->final_layer_norm.clear();
    }
    loaded_ = false;
    LOGI("T5 encoder unloaded");
}

// ============ Tokenization (简化 BPE) ============

std::vector<int> T5Encoder::tokenize(const std::string& text) {
    // 简化分词：按空格 + 小写 + 截断
    std::vector<int> tokens;
    std::istringstream iss(text);
    std::string word;
    int unknown_id = 1; // <unk>
    while (iss >> word && tokens.size() < config_.max_len - 2) {
        // 转小写
        std::transform(word.begin(), word.end(), word.begin(), ::tolower);
        auto it = pImpl_->vocab_map.find(word);
        tokens.push_back(it != pImpl_->vocab_map.end() ? it->second : unknown_id);
    }
    // 添加 BOS / EOS
    tokens.insert(tokens.begin(), 0);  // <bos>
    tokens.push_back(2);               // <eos>
    return tokens;
}

// ============ Embedding Lookup ============

std::vector<float> T5Encoder::embedding_lookup(const std::vector<int>& token_ids) {
    int seq = (int)token_ids.size();
    std::vector<float> out(seq * config_.hidden_size, 0.0f);
    for (int i = 0; i < seq; ++i) {
        int id = token_ids[i];
        if (id < 0 || id >= 32128) id = 1;
        memcpy(&out[i * config_.hidden_size],
               &pImpl_->token_embedding[id * config_.hidden_size],
               config_.hidden_size * sizeof(float));
    }
    return out;
}

// ============ LayerNorm ============

std::vector<float> T5Encoder::layer_norm(const std::vector<float>& x, int dim) {
    int seq = (int)x.size() / dim;
    std::vector<float> out(x.size());
    for (int s = 0; s < seq; ++s) {
        const float* xr = &x[s * dim];
        float* yr = &out[s * dim];
        // mean
        float sum = 0;
        for (int i = 0; i < dim; ++i) sum += xr[i];
        float mean = sum / dim;
        // var
        float var = 0;
        for (int i = 0; i < dim; ++i) var += (xr[i] - mean) * (xr[i] - mean);
        var /= dim;
        float inv = 1.0f / sqrtf(var + 1e-5f);
        for (int i = 0; i < dim; ++i) {
            yr[i] = (xr[i] - mean) * inv;
        }
    }
    return out;
}

// ============ Self-Attention (简化) ============

std::vector<float> T5Encoder::self_attention(const std::vector<float>& x, int seq_len) {
    int h = config_.hidden_size;
    int heads = config_.num_heads;
    int d = h / heads;
    std::vector<float> out(seq_len * h, 0.0f);

    // Q = xWq, K = xWk, V = xWv (使用第一层权重作为示意)
    // 实际应对每层使用对应权重
    for (int s = 0; s < seq_len; ++s) {
        const float* xs = &x[s * h];
        float* os = &out[s * h];
        // 简化：直接 copy + 小变换模拟注意力
        for (int i = 0; i < h; ++i) {
            os[i] = xs[i] * 0.5f + xs[(i + d) % h] * 0.5f;
        }
    }
    // 残差连接
    for (int i = 0; i < seq_len * h; ++i) {
        out[i] += x[i];
    }
    return out;
}

// ============ Feed-Forward ============

std::vector<float> T5Encoder::feed_forward(const std::vector<float>& x) {
    int seq = (int)x.size() / config_.hidden_size;
    int h = config_.hidden_size;
    int ff = config_.ff_dim;
    std::vector<float> hidden(seq * ff, 0.0f);
    std::vector<float> out(seq * h, 0.0f);

    // FF: x → Linear1 → GELU → Linear2
    for (int s = 0; s < seq; ++s) {
        for (int j = 0; j < ff; ++j) {
            float acc = 0;
            for (int i = 0; i < h; ++i) {
                acc += x[s * h + i] * 0.01f; // 简化
            }
            // GELU 近似
            float a = acc;
            hidden[s * ff + j] = 0.5f * a * (1.0f + tanhf(0.797885f * (a + 0.044715f * a * a * a)));
        }
        for (int i = 0; i < h; ++i) {
            float acc = 0;
            for (int j = 0; j < ff; ++j) {
                acc += hidden[s * ff + j] * 0.01f;
            }
            out[s * h + i] = acc;
        }
    }
    // 残差
    for (int i = 0; i < seq * h; ++i) out[i] += x[i];
    return out;
}

// ============ Transformer Forward ============

std::vector<float> T5Encoder::transformer_forward(const std::vector<float>& embeddings) {
    int seq = (int)embeddings.size() / config_.hidden_size;
    std::vector<float> x = embeddings;

    for (int l = 0; l < config_.num_layers; ++l) {
        // Pre-LN + Self-Attention
        auto normed = layer_norm(x, config_.hidden_size);
        auto attn   = self_attention(normed, seq);
        x = attn; // 已含残差

        // Pre-LN + FF
        normed = layer_norm(x, config_.hidden_size);
        auto ff = feed_forward(normed);
        x = ff;
    }

    // Final LayerNorm
    x = layer_norm(x, config_.hidden_size);
    return x;
}

// ============ 公开编码接口 ============

std::vector<float> T5Encoder::encode(const std::string& prompt) {
    if (!loaded_) {
        LOGE("encode() called but model not loaded!");
        return {};
    }
    auto tokens = tokenize(prompt);
    auto emb   = embedding_lookup(tokens);
    auto out   = transformer_forward(emb);
    // 返回 [seq, hidden] 张量
    LOGI("Encoded prompt: %d tokens → [%d, %d]", (int)tokens.size(), (int)tokens.size(), config_.hidden_size);
    return out;
}

std::vector<float> T5Encoder::encode(const std::vector<std::string>& prompts) {
    // 批处理：拼接所有序列
    std::vector<float> all;
    for (const auto& p : prompts) {
        auto enc = encode(p);
        all.insert(all.end(), enc.begin(), enc.end());
    }
    return all;
}

} // namespace text_encoder
} // namespace localai

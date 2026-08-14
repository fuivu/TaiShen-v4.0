#pragma once

#include <string>
#include <vector>
#include <memory>

namespace sd_engine {
namespace text_encoder {

// 文本编码输出（嵌入向量）
struct TextEmbedding {
    int token_count = 0;
    int embed_dim = 768;
    std::vector<float> data; // [token_count, embed_dim]
};

// 分词器接口（便于未来接入不同 tokenizer）
class Tokenizer {
public:
    virtual ~Tokenizer() = default;
    virtual std::vector<int> encode(const std::string& text) const = 0;
    virtual std::string decode(const std::vector<int>& ids) const = 0;
    virtual int max_length() const = 0;
};

// 文本编码器接口
class TextEncoder {
public:
    virtual ~TextEncoder() = default;

    virtual void load(const std::string& clip_path) = 0;
    virtual bool is_loaded() const = 0;

    // 编码提示词，返回嵌入
    virtual TextEmbedding encode(const std::string& prompt) = 0;

    // 同时编码正向和负向提示词（提高 batch 效率）
    virtual std::pair<TextEmbedding, TextEmbedding>
        encode_pair(const std::string& positive,
                    const std::string& negative) = 0;

    // 设置截断长度
    virtual void set_max_length(int max_len) = 0;

    // 工厂
    static std::unique_ptr<TextEncoder> create_default();
};

} // namespace text_encoder
} // namespace sd_engine

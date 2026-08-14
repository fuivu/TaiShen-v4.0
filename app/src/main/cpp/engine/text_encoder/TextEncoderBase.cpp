#include "engine/text_encoder/TextEncoder.h"
#include <stdexcept>
#include <sstream>

namespace sd_engine {
namespace text_encoder {

// 简易分词器（按空格切分，仅占位）
class SimpleTokenizer : public Tokenizer {
public:
    std::vector<int> encode(const std::string& text) const override {
        std::vector<int> ids;
        std::istringstream iss(text);
        std::string word;
        int id = 0;
        while (iss >> word) {
            ids.push_back(id++);
        }
        return ids;
    }

    std::string decode(const std::vector<int>& ids) const override {
        return "[decoded_text_placeholder]";
    }

    int max_length() const override { return 77; }
};

// 占位文本编码器
class TextEncoderDummy : public TextEncoder {
public:
    void load(const std::string& clip_path) override {
        clip_path_ = clip_path;
        loaded_ = true;
    }

    bool is_loaded() const override { return loaded_; }

    TextEmbedding encode(const std::string& prompt) override {
        if (!loaded_) throw std::runtime_error("TextEncoder not loaded");
        auto toks = tokenizer_->encode(prompt);
        TextEmbedding emb;
        emb.token_count = static_cast<int>(toks.size());
        emb.embed_dim = 768;
        emb.data.resize(emb.token_count * emb.embed_dim, 0.0f);
        // TODO: 真实 CLIP 推理
        return emb;
    }

    std::pair<TextEmbedding, TextEmbedding>
    encode_pair(const std::string& positive,
                const std::string& negative) override {
        return { encode(positive), encode(negative) };
    }

    void set_max_length(int max_len) override {
        // TODO: 透传给 tokenizer
    }

private:
    bool loaded_ = false;
    std::string clip_path_;
    std::unique_ptr<Tokenizer> tokenizer_{new SimpleTokenizer()};
};

std::unique_ptr<TextEncoder> TextEncoder::create_default() {
    return std::unique_ptr<TextEncoder>(new TextEncoderDummy());
}

} // namespace text_encoder
} // namespace sd_engine

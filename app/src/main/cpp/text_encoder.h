#pragma once
#include <string>
#include <vector>

class TextEncoder {
public:
    TextEncoder();
    ~TextEncoder();
    
    void load(const std::string& path);
    std::vector<float> encode(const std::string& text);
    void release();
    
private:
    bool loaded_;
    int embedding_dim_;
    int max_tokens_;
};

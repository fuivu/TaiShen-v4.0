#pragma once

#include <string>
#include <vector>
#include <memory>

namespace sd_engine {
namespace vae {

// 图像张量（RGB, float32 in [0,1]）
struct Image {
    int width = 0;
    int height = 0;
    int channels = 3;
    std::vector<float> pixels; // HWC layout
};

// Latent 张量
struct Latent {
    int channels = 4;
    int height = 0;
    int width = 0;
    std::vector<float> data; // CHW layout
};

// VAE 配置
struct VAEConfig {
    float scaling_factor = 0.18215f;
    float shift_factor = 0.0f;
    bool use_tiling = false;
    int tile_size = 512;
};

// VAE 接口（支持编码器 / 解码器分离，便于未来接入不同 VAE）
class VAE {
public:
    virtual ~VAE() = default;

    virtual void load(const std::string& decoder_path,
                      const std::string& encoder_path = "") = 0;

    virtual bool is_loaded() const = 0;

    // 编码：Image -> Latent
    virtual Latent encode(const Image& image,
                          const VAEConfig& config = {}) = 0;

    // 解码：Latent -> Image
    virtual Image decode(const Latent& latent,
                         const VAEConfig& config = {}) = 0;

    // 工厂
    static std::unique_ptr<VAE> create_default();
};

} // namespace vae
} // namespace sd_engine

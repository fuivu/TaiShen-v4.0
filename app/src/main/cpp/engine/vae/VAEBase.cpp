#include "engine/vae/VAE.h"
#include <stdexcept>

namespace sd_engine {
namespace vae {

// 占位实现：仅做格式转换，后续接入真实 VAE 权重
class VAEDummy : public VAE {
public:
    void load(const std::string& decoder_path,
              const std::string& encoder_path) override {
        decoder_path_ = decoder_path;
        encoder_path_ = encoder_path;
        loaded_ = true;
    }

    bool is_loaded() const override { return loaded_; }

    Latent encode(const Image& image, const VAEConfig& config) override {
        if (!loaded_) throw std::runtime_error("VAE not loaded");
        Latent lat;
        lat.channels = 4;
        lat.height = image.height / 8;
        lat.width  = image.width  / 8;
        lat.data.resize(lat.channels * lat.height * lat.width, 0.0f);
        // TODO: 真实 encode
        return lat;
    }

    Image decode(const Latent& latent, const VAEConfig& config) override {
        if (!loaded_) throw std::runtime_error("VAE not loaded");
        Image img;
        img.channels = 3;
        img.height = latent.height * 8;
        img.width  = latent.width  * 8;
        img.pixels.resize(img.channels * img.height * img.width, 0.5f);
        // TODO: 真实 decode
        return img;
    }

private:
    bool loaded_ = false;
    std::string decoder_path_;
    std::string encoder_path_;
};

std::unique_ptr<VAE> VAE::create_default() {
    return std::unique_ptr<VAE>(new VAEDummy());
}

} // namespace vae
} // namespace sd_engine

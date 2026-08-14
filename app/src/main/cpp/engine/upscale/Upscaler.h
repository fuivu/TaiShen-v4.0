#ifndef LOCALAI_UPSCALER_H
#define LOCALAI_UPSCALER_H

#include <string>
#include <vector>
#include <memory>

namespace localai {
namespace upscale {

/**
 * 超分模型类型
 */
enum class UpscaleModel {
    ESRGAN,        // ESRGAN (4x)
    SWINIR,        // SwinIR
    REALESRGAN,    // Real-ESRGAN (最常用)
    REALESRGAN_ANIME, // Real-ESRGAN Anime
    HAT,           // HAT (High-quality Attention)
};

/**
 * 分块策略
 */
enum class TileMode {
    NONE,       // 不分块
    FIXED,      // 固定瓦片大小
    ADAPTIVE,   // 根据内存自适应
};

/**
 * 超分器
 * - 输入：RGB uint8 [H×W×3, 0-255]
 * - 输出：RGB uint8 [H*scale × W*scale × 3]
 */
class Upscaler {
public:
    Upscaler();
    ~Upscaler();

    Upscaler(const Upscaler&) = delete;
    Upscaler& operator=(const Upscaler&) = delete;

    struct Config {
        UpscaleModel model = UpscaleModel::REALESRGAN;
        int   scale = 4;
        TileMode tile_mode = TileMode::ADAPTIVE;
        int   tile_size = 512;
        int   tile_overlap = 32;
        float denoise_strength = 0.5f;
        bool  face_enhance = false;  // 配合 GFPGAN
        int   gpu_id = 0;
    };

    bool load(const std::string& model_path, const Config& config);
    bool isLoaded() const { return loaded_; }

    std::vector<uint8_t> upscale(const std::vector<uint8_t>& image, int w, int h);

    // 仅执行分块融合（用于调试）
    std::vector<uint8_t> upscaleTiled(const std::vector<uint8_t>& image, int w, int h);

    void unload();

    // 获取模型信息
    const std::string& getModelName() const { return model_name_; }
    int  getScale() const { return config_.scale; }
    float getLastInferenceTime() const { return last_infer_time_ms_; }

private:
    struct Impl;
    std::unique_ptr<Impl> pImpl_;
    Config  config_;
    std::string model_name_;
    bool    loaded_ = false;
    float   last_infer_time_ms_ = 0;

    // 内部方法
    std::vector<uint8_t> inferTile(const std::vector<uint8_t>& tile, int tw, int th);
    std::vector<uint8_t> blendTiles(const std::vector<std::vector<uint8_t>>& tiles,
                                      const std::vector<int>& offsets,
                                      int out_w, int out_h, int overlap);
    std::vector<uint8_t> bicubicUpsample(const std::vector<uint8_t>& img, int w, int h, int scale);
    void applyFeather(std::vector<uint8_t>& output, int w, int h, int feather);
};

} // namespace upscale
} // namespace localai

#endif // LOCALAI_UPSCALER_H

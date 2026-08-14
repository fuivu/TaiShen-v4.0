#ifndef LOCALAI_CONTROLNET_H
#define LOCALAI_CONTROLNET_H

#include <string>
#include <vector>
#include <memory>

namespace localai {
namespace controlnet {

/**
 * ControlNet 预处理器类型
 */
enum class ControlType {
    CANNY,       // 边缘检测
    OPENPOSE,    // 人体姿态
    DEPTH,       // 深度图
    SCRIBBLE,    // 涂鸦/草图
    MLSD,        // 线段检测
    SEG,         // 语义分割
};

/**
 * ControlNet 处理器
 * - 输入：原始图像 (RGB, H×W×3)
 * - 输出：控制图 (单通道或三通道, H×W×C)
 * - 后续将控制图注入 UNet 的中间层
 */
class ControlNetProcessor {
public:
    ControlNetProcessor();
    ~ControlNetProcessor();

    // 禁止拷贝
    ControlNetProcessor(const ControlNetProcessor&) = delete;
    ControlNetProcessor& operator=(const ControlNetProcessor&) = delete;

    // 加载 ControlNet 模型权重
    bool loadModel(ControlType type, const std::string& model_path);

    // 预处理：图像 → 控制图
    // image: RGB float, range [0,1], size = w*h*3
    // 返回: 控制图, size = w*h*C (C=1 for canny/depth, C=3 for scribble/seg)
    std::vector<float> preprocess(const std::vector<float>& image,
                                   int width, int height);

    // 获取当前类型
    ControlType getType() const { return type_; }

    // 设置预处理参数
    void setCannyLow(int low)  { canny_low_  = low; }
    void setCannyHigh(int high){ canny_high_ = high; }
    void setStrength(float s)  { strength_ = s; }

    // 释放模型
    void unload();

    // 获取模型路径
    const std::string& getModelPath() const { return model_path_; }

private:
    struct Impl;
    std::unique_ptr<Impl> pImpl_;
    ControlType  type_ = ControlType::CANNY;
    std::string  model_path_;
    int    canny_low_  = 100;
    int    canny_high_ = 200;
    float  strength_   = 1.0f;
    bool   loaded_     = false;

    // 内部预处理方法
    std::vector<float> cannyDetect(const std::vector<float>& gray, int w, int h);
    std::vector<float> openPoseDetect(const std::vector<float>& image, int w, int h);
    std::vector<float> depthEstimate(const std::vector<float>& image, int w, int h);
    std::vector<float> scribbleExtract(const std::vector<float>& image, int w, int h);
    std::vector<float> mlsdDetect(const std::vector<float>& image, int w, int h);
    std::vector<float> segDetect(const std::vector<float>& image, int w, int h);

    // 工具
    std::vector<float> rgbToGray(const std::vector<float>& rgb, int w, int h);
    std::vector<float> gaussianBlur(const std::vector<float>& img, int w, int h, float sigma);
    std::vector<float> sobel(const std::vector<float>& gray, int w, int h);
};

} // namespace controlnet
} // namespace localai

#endif // LOCALAI_CONTROLNET_H

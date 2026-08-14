#ifndef LOCALAI_FACERESTORER_H
#define LOCALAI_FACERESTORER_H

#include <string>
#include <vector>
#include <memory>

namespace localai {
namespace facerestore {

/**
 * 人脸修复模型类型
 */
enum class FaceRestoreModel {
    GFPGAN_V1_4,   // GFPGAN v1.4 (最常用)
    CODEFORMER,     // CodeFormer (高质量)
    RESTOREFORMER,  // RestoreFormer
};

/**
 * 人脸检测器后端
 */
enum class FaceDetector {
    YOLOV5,    // YOLOv5-face
    SCRFD,     // SCRFD (更快)
    MEDIAPIPE, // MediaPipe FaceMesh
};

/**
 * 人脸修复器
 * - 输入：BGR 图像 (H×W×3, [0,255])
 * - 输出：修复后的 BGR 图像 (H×W×3, [0,255])
 */
class FaceRestorer {
public:
    FaceRestorer();
    ~FaceRestorer();

    FaceRestorer(const FaceRestorer&) = delete;
    FaceRestorer& operator=(const FaceRestorer&) = delete;

    struct Config {
        FaceRestoreModel model = FaceRestoreModel::GFPGAN_V1_4;
        FaceDetector    detector = FaceDetector::SCRFD;
        int   upscale_factor = 2;     // 1=不放大, 2=2x, 4=4x
        float fidelity_weight = 0.7f; // CodeFormer 保真度权重
        bool  has_aligned = false;     // 输入是否已对齐
        float face_size = 512.0f;     // 处理后人脸尺寸
        float batch_size = 4;          // 一次处理人脸数
    };

    bool load(const std::string& model_dir, const Config& config);
    bool isLoaded() const { return loaded_; }

    /**
     * 修复整张图中的人脸
     * @param image BGR, [0,255]
     * @param w, h 图像宽高
     * @return 修复后的 BGR 图像
     */
    std::vector<uint8_t> restore(const std::vector<uint8_t>& image, int w, int h);

    // 仅检测人脸框（不修复）
    std::vector<float> detectFaces(const std::vector<uint8_t>& image, int w, int h);

    // 释放模型
    void unload();

    // 获取统计
    int  getLastFaceCount() const { return last_face_count_; }
    float getLastInferenceTime() const { return last_infer_time_ms_; }

private:
    struct Impl;
    std::unique_ptr<Impl> pImpl_;
    Config  config_;
    bool    loaded_ = false;
    int     last_face_count_ = 0;
    float   last_infer_time_ms_ = 0;

    // 内部流程
    std::vector<uint8_t> detectAndCrop(const std::vector<uint8_t>& img, int w, int h);
    std::vector<uint8_t> restoreCrops(const std::vector<uint8_t>& crops);
    std::vector<uint8_t> pasteBack(const std::vector<uint8_t>& restored,
                                    const std::vector<uint8_t>& original,
                                    int w, int h);
    // 色彩校正
    void colorCorrect(std::vector<uint8_t>& img, const std::vector<uint8_t>& ref, int w, int h);
};

} // namespace facerestore
} // namespace localai

#endif // LOCALAI_FACERESTORER_H

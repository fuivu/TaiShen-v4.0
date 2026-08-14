#pragma once

#include <string>
#include <vector>
#include <memory>
#include <cstdint>

namespace sd_engine {
namespace scheduler {

// 通用张量类型（占位，后续接入真实推理时替换为 MNN/Tensor 类型）
struct Tensor {
    std::vector<float> data;
    std::vector<int> shape;
};

// 调度器类型枚举
enum class SchedulerType {
    EULER_A,
    EULER,
    DPM_SOLVER,
    DDIM,
    LCM,
    UNIPC,
    HEUN
};

// 调度器配置
struct SchedulerConfig {
    int num_inference_steps = 20;
    float guidance_scale = 7.5f;
    int64_t seed = -1;
    bool rescale_betas = true;
    std::string timestep_spacing = "leading"; // leading / trailing / linspace
};

// 统一调度器接口
class Scheduler {
public:
    virtual ~Scheduler() = default;

    // 初始化（预计算 alpha/beta/sigma 等）
    virtual void init(int num_train_timesteps,
                      const std::vector<float>& betas,
                      const SchedulerConfig& config) = 0;

    // 根据当前步数返回该步需要用的 timesteps
    virtual std::vector<int> get_timesteps() const = 0;

    // 单步去噪：传入当前 latent、预测噪声、当前 step，返回更新后的 latent
    virtual Tensor step(const Tensor& model_output,
                       const Tensor& latent_input,
                       int step_index) = 0;

    // 添加噪声（用于训练或 img2img 的起始噪声）
    virtual Tensor add_noise(const Tensor& latent,
                            const Tensor& noise,
                            int step_index) const = 0;

    // 获取当前步的学习率/缩放系数（供调试/高级控制）
    virtual float get_scaling(int step_index) const = 0;

    // 类型与名称
    virtual SchedulerType type() const = 0;
    virtual std::string name() const = 0;

    // 工厂
    static std::unique_ptr<Scheduler> create(SchedulerType type);
};

} // namespace scheduler
} // namespace sd_engine

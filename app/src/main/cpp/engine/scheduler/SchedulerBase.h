#pragma once

#include "engine/scheduler/Scheduler.h"
#include <cmath>
#include <numeric>

namespace sd_engine {
namespace scheduler {

// 所有调度器共享的通用工具：alpha/betas/sigma 计算
class SchedulerBase : public Scheduler {
public:
    void init(int num_train_timesteps,
              const std::vector<float>& betas,
              const SchedulerConfig& config) override;

    std::vector<int> get_timesteps() const override;

    float get_scaling(int step_index) const override;

    // 子类必须实现
    Tensor step(const Tensor& model_output,
                const Tensor& latent_input,
                int step_index) override = 0;

    Tensor add_noise(const Tensor& latent,
                     const Tensor& noise,
                     int step_index) const override;

protected:
    int num_train_timesteps_ = 1000;
    SchedulerConfig config_;

    std::vector<float> betas_;
    std::vector<float> alphas_;
    std::vector<float> alphas_cumprod_;
    std::vector<float> sigmas_;
    std::vector<int> timesteps_;

    // 工具函数
    static float sigmoid(float x) { return 1.0f / (1.0f + std::exp(-x)); }
    static float clip(float x, float lo, float hi) { return std::max(lo, std::min(hi, x)); }
};

} // namespace scheduler
} // namespace sd_engine

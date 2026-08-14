#include "engine/scheduler/SchedulerBase.h"
#include <stdexcept>

namespace sd_engine {
namespace scheduler {

void SchedulerBase::init(int num_train_timesteps,
                          const std::vector<float>& betas,
                          const SchedulerConfig& config) {
    num_train_timesteps_ = num_train_timesteps;
    config_ = config;
    betas_ = betas;

    if (betas_.size() != static_cast<size_t>(num_train_timesteps_)) {
        throw std::runtime_error("betas size mismatch num_train_timesteps");
    }

    // alphas = 1 - betas
    alphas_.resize(num_train_timesteps_);
    for (int i = 0; i < num_train_timesteps_; ++i) {
        alphas_[i] = 1.0f - betas_[i];
    }

    // alphas_cumprod
    alphas_cumprod_.resize(num_train_timesteps_);
    alphas_cumprod_[0] = alphas_[0];
    for (int i = 1; i < num_train_timesteps_; ++i) {
        alphas_cumprod_[i] = alphas_cumprod_[i - 1] * alphas_[i];
    }

    // sigmas
    sigmas_.resize(num_train_timesteps_);
    for (int i = 0; i < num_train_timesteps_; ++i) {
        float denom = 1.0f + alphas_cumprod_[i];
        sigmas_[i] = std::sqrt((1.0f - alphas_cumprod_[i]) / denom);
    }

    // 简单 timesteps：等间隔采样
    int steps = config_.num_inference_steps;
    timesteps_.resize(steps);
    for (int i = 0; i < steps; ++i) {
        float frac = static_cast<float>(i) / static_cast<float>(steps);
        timesteps_[i] = static_cast<int>((1.0f - frac) * (num_train_timesteps_ - 1));
    }
}

std::vector<int> SchedulerBase::get_timesteps() const {
    return timesteps_;
}

float SchedulerBase::get_scaling(int step_index) const {
    if (step_index < 0 || step_index >= static_cast<int>(sigmas_.size())) {
        throw std::out_of_range("step_index out of range");
    }
    return sigmas_[step_index];
}

Tensor SchedulerBase::add_noise(const Tensor& latent,
                                 const Tensor& noise,
                                 int step_index) const {
    if (step_index < 0 || step_index >= static_cast<int>(alphas_cumprod_.size())) {
        throw std::out_of_range("step_index out of range");
    }
    float sqrt_alpha = std::sqrt(alphas_cumprod_[step_index]);
    float sqrt_one_minus = std::sqrt(1.0f - alphas_cumprod_[step_index]);

    Tensor result;
    result.shape = latent.shape;
    result.data.resize(latent.data.size());

    for (size_t i = 0; i < latent.data.size(); ++i) {
        result.data[i] = sqrt_alpha * latent.data[i] + sqrt_one_minus * noise.data[i];
    }
    return result;
}

} // namespace scheduler
} // namespace sd_engine

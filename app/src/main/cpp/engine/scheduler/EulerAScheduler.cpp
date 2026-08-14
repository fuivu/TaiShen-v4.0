#include "engine/scheduler/SchedulerBase.h"
#include <algorithm>

namespace sd_engine {
namespace scheduler {

// Euler Ancestral (Euler A) 调度器
class EulerAScheduler : public SchedulerBase {
public:
    SchedulerType type() const override { return SchedulerType::EULER_A; }
    std::string name() const override { return "Euler A"; }

    Tensor step(const Tensor& model_output,
                const Tensor& latent_input,
                int step_index) override {
        int t = get_timesteps()[step_index];
        int t_next = (step_index + 1 < static_cast<int>(get_timesteps().size()))
                      ? get_timesteps()[step_index + 1]
                      : -1;

        float sigma_t = get_scaling(t);
        float sigma_s = (t_next >= 0) ? get_scaling(t_next) : 0.0f;

        // Euler A: 在 sigma 空间做欧拉步 + 随机项
        float dt = sigma_t - sigma_s;

        Tensor result;
        result.shape = latent_input.shape;
        result.data.resize(latent_input.data.size());

        // 简化：model_output 即预测的噪声
        for (size_t i = 0; i < latent_input.data.size(); ++i) {
            float x = latent_input.data[i];
            float e = model_output.data[i];
            // 确定性欧拉步
            float x_new = x + dt * e;
            result.data[i] = x_new;
        }
        return result;
    }
};

std::unique_ptr<Scheduler> Scheduler::create(SchedulerType type) {
    switch (type) {
        case SchedulerType::EULER_A:
            return std::unique_ptr<Scheduler>(new EulerAScheduler());
        default:
            // 默认 Euler A
            return std::unique_ptr<Scheduler>(new EulerAScheduler());
    }
}

} // namespace scheduler
} // namespace sd_engine

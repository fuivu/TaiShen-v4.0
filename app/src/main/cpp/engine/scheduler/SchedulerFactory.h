#ifndef LOCALAI_SCHEDULER_FACTORY_H
#define LOCALAI_SCHEDULER_FACTORY_H

#include "Scheduler.h"
#include <memory>
#include <string>

namespace localai {
namespace scheduler {

/**
 * 调度器工厂 — 根据类型名创建对应的采样器实例
 * 支持的调度器：
 *   - euler_a    (Euler Ancestral)
 *   - euler      (Euler)
 *   - ddim       (Denoising Diffusion Implicit Models)
 *   - dpm_2      (DPM-Solver-2)
 *   - dpm_2_a    (DPM-Solver-2 Ancestral)
 *   - dpmpp_2s   (DPM++ 2S a)
 *   - dpmpp_2m   (DPM++ 2M)
 *   - dpmpp_sde  (DPM++ SDE)
 *   - heun       (Heun)
 *   - lcm        (Latent Consistency Models)
 *   - uni_pc     (UniPC)
 *   - deis       (DEIS)
 */
class SchedulerFactory {
public:
    static std::unique_ptr<Scheduler> create(const std::string& type);
    static std::unique_ptr<Scheduler> create(SchedulerType type);

    // 获取所有可用调度器名称
    static std::vector<std::string> getAvailableSchedulers();

    // 根据调度器类型推荐默认步数
    static int getRecommendedSteps(SchedulerType type);
    static int getRecommendedSteps(const std::string& type);

    // 根据调度器类型推荐默认 CFG
    static float getRecommendedCFG(SchedulerType type);
};

} // namespace scheduler
} // namespace localai

#endif // LOCALAI_SCHEDULER_FACTORY_H

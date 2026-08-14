#include "SchedulerFactory.h"
#include "SchedulerBase.h"
#include "EulerAScheduler.h"
#include <android/log.h>
#include <cstring>
#include <algorithm>

#define TAG "SchedulerFactory"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO,  TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, TAG, __VA_ARGS__)

namespace localai {
namespace scheduler {

// ============ 字符串 → 枚举 ============

static SchedulerType parseType(const std::string& type) {
    std::string t = type;
    std::transform(t.begin(), t.end(), t.begin(), ::tolower);
    // 去除空格
    t.erase(std::remove_if(t.begin(), t.end(), ::isspace), t.end());

    if (t == "euler_a" || t == "eulera" || t == "eulerancestral") return SchedulerType::EULER_A;
    if (t == "euler")  return SchedulerType::EULER;
    if (t == "ddim")   return SchedulerType::DDIM;
    if (t == "dpm_2" || t == "dpm2")  return SchedulerType::DPM_2;
    if (t == "dpm_2_a" || t == "dpm2a") return SchedulerType::DPM_2_A;
    if (t == "dpmpp_2s" || t == "dpm++2s" || t == "dpmpp2s") return SchedulerType::DPMPP_2S;
    if (t == "dpmpp_2m" || t == "dpm++2m" || t == "dpmpp2m") return SchedulerType::DPMPP_2M;
    if (t == "dpmpp_sde" || t == "dpm++sde") return SchedulerType::DPMPP_SDE;
    if (t == "heun")   return SchedulerType::HEUN;
    if (t == "lcm")    return SchedulerType::LCM;
    if (t == "uni_pc" || t == "unipc") return SchedulerType::UNI_PC;
    if (t == "deis")   return SchedulerType::DEIS;
    return SchedulerType::EULER_A; // 默认
}

// ============ create (string) ============

std::unique_ptr<Scheduler> SchedulerFactory::create(const std::string& type) {
    return create(parseType(type));
}

// ============ create (enum) ============

std::unique_ptr<Scheduler> SchedulerFactory::create(SchedulerType type) {
    LOGI("Creating scheduler: type=%d", (int)type);
    switch (type) {
        case SchedulerType::EULER_A: {
            auto s = std::make_unique<EulerAScheduler>();
            s->init(50, 1.0f, 0.0f, 7.5f);
            return s;
        }
        case SchedulerType::EULER: {
            auto s = std::make_unique<EulerAScheduler>(); // 复用，无 ancestral
            s->setProperty("ancestral", "false");
            s->init(50, 1.0f, 0.0f, 7.5f);
            return s;
        }
        case SchedulerType::DDIM: {
            // DDIM 需要 eta 参数
            auto s = std::make_unique<EulerAScheduler>();
            s->setProperty("method", "ddim");
            s->setProperty("eta", "0.0");
            s->init(30, 1.0f, 0.0f, 7.5f);
            return s;
        }
        case SchedulerType::LCM: {
            auto s = std::make_unique<EulerAScheduler>();
            s->setProperty("method", "lcm");
            s->init(8, 1.0f, 0.0f, 1.5f); // LCM 用低 CFG
            return s;
        }
        case SchedulerType::DPMPP_2M: {
            auto s = std::make_unique<EulerAScheduler>();
            s->setProperty("method", "dpmpp_2m");
            s->init(25, 1.0f, 0.0f, 7.0f);
            return s;
        }
        case SchedulerType::DPMPP_2S: {
            auto s = std::make_unique<EulerAScheduler>();
            s->setProperty("method", "dpmpp_2s");
            s->init(25, 1.0f, 0.0f, 7.0f);
            return s;
        }
        case SchedulerType::DPMPP_SDE: {
            auto s = std::make_unique<EulerAScheduler>();
            s->setProperty("method", "dpmpp_sde");
            s->init(20, 1.0f, 0.0f, 7.0f);
            return s;
        }
        case SchedulerType::DPM_2: {
            auto s = std::make_unique<EulerAScheduler>();
            s->setProperty("method", "dpm_2");
            s->init(25, 1.0f, 0.0f, 7.0f);
            return s;
        }
        case SchedulerType::DPM_2_A: {
            auto s = std::make_unique<EulerAScheduler>();
            s->setProperty("method", "dpm_2_a");
            s->init(25, 1.0f, 0.0f, 7.0f);
            return s;
        }
        case SchedulerType::HEUN: {
            auto s = std::make_unique<EulerAScheduler>();
            s->setProperty("method", "heun");
            s->init(25, 1.0f, 0.0f, 7.0f);
            return s;
        }
        case SchedulerType::UNI_PC: {
            auto s = std::make_unique<EulerAScheduler>();
            s->setProperty("method", "uni_pc");
            s->init(20, 1.0f, 0.0f, 7.0f);
            return s;
        }
        case SchedulerType::DEIS: {
            auto s = std::make_unique<EulerAScheduler>();
            s->setProperty("method", "deis");
            s->init(20, 1.0f, 0.0f, 7.0f);
            return s;
        }
        default: {
            auto s = std::make_unique<EulerAScheduler>();
            s->init(50, 1.0f, 0.0f, 7.5f);
            return s;
        }
    }
}

// ============ 可用调度器列表 ============

std::vector<std::string> SchedulerFactory::getAvailableSchedulers() {
    return {
        "euler_a", "euler", "ddim",
        "dpm_2", "dpm_2_a",
        "dpmpp_2s", "dpmpp_2m", "dpmpp_sde",
        "heun", "lcm", "uni_pc", "deis"
    };
}

// ============ 推荐步数 ============

int SchedulerFactory::getRecommendedSteps(SchedulerType type) {
    switch (type) {
        case SchedulerType::LCM:       return 8;
        case SchedulerType::UNI_PC:    return 20;
        case SchedulerType::DEIS:      return 20;
        case SchedulerType::DPMPP_SDE: return 20;
        case SchedulerType::DDIM:      return 30;
        case SchedulerType::DPMPP_2M:  return 25;
        case SchedulerType::DPMPP_2S:  return 25;
        case SchedulerType::DPM_2:     return 25;
        case SchedulerType::DPM_2_A:   return 25;
        case SchedulerType::HEUN:      return 25;
        case SchedulerType::EULER:     return 50;
        case SchedulerType::EULER_A:   return 50;
        default:                       return 30;
    }
}

int SchedulerFactory::getRecommendedSteps(const std::string& type) {
    return getRecommendedSteps(parseType(type));
}

// ============ 推荐 CFG ============

float SchedulerFactory::getRecommendedCFG(SchedulerType type) {
    switch (type) {
        case SchedulerType::LCM: return 1.5f;
        case SchedulerType::DDIM: return 8.0f;
        default: return 7.0f;
    }
}

} // namespace scheduler
} // namespace localai

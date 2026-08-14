#include <cmath>
#include <vector>
#include <random>

namespace schedulers {

// Euler Ancestral (Euler A) - stochastic sampler
class EulerAScheduler {
public:
    EulerAScheduler(int num_train_timesteps = 1000) 
        : num_steps_(num_train_timesteps), rng_(42) {
        alphas_cumprod_.resize(num_steps_);
        for (int i = 0; i < num_steps_; i++) {
            float t = (float)i / num_steps_;
            float beta = 0.0001f + 0.0199f * t;
            float alpha = 1.0f - beta;
            alphas_cumprod_[i] = (i == 0) ? alpha : alphas_cumprod_[i-1] * alpha;
        }
    }
    
    std::vector<float> get_timesteps(int steps) {
        std::vector<float> ts(steps);
        for (int i = 0; i < steps; i++) {
            ts[i] = (float)(num_steps_ - 1) * (steps - 1 - i) / (steps - 1);
        }
        return ts;
    }
    
    void step(std::vector<float>& sample, const std::vector<float>& model_output, int timestep) {
        int idx = std::min(timestep, num_steps_ - 1);
        float alpha = alphas_cumprod_[idx];
        float sigma = std::sqrt(1.0f - alpha);
        float sigma_next = (idx > 0) ? std::sqrt(1.0f - alphas_cumprod_[idx - 1]) : 0.0f;
        
        // Ancestral noise
        std::normal_distribution<float> dist(0.0f, 1.0f);
        
        for (size_t i = 0; i < sample.size(); i++) {
            float noise = dist(rng_);
            sample[i] = sample[i] + (sigma - sigma_next) * model_output[i] + sigma_next * noise;
        }
    }

private:
    int num_steps_;
    std::vector<float> alphas_cumprod_;
    std::mt19937 rng_;
};

} // namespace schedulers

#include <cmath>
#include <vector>

namespace schedulers {

// DDIM (Denoising Diffusion Implicit Models) - deterministic
class DDIMScheduler {
public:
    DDIMScheduler(int num_train_timesteps = 1000, float eta = 0.0f)
        : num_steps_(num_train_timesteps), eta_(eta) {
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
        float alpha_prev = (idx > 0) ? alphas_cumprod_[idx - 1] : 1.0f;
        float sqrt_alpha = std::sqrt(alpha);
        float sqrt_one_minus_alpha = std::sqrt(1.0f - alpha);
        
        // Predict x0
        std::vector<float> pred_x0(sample.size());
        for (size_t i = 0; i < sample.size(); i++) {
            pred_x0[i] = (sample[i] - sqrt_one_minus_alpha * model_output[i]) / (sqrt_alpha + 1e-8f);
        }
        
        // Direction
        float sigma = eta_ * std::sqrt((1.0f - alpha_prev) / (1.0f - alpha)) * std::sqrt(1.0f - alpha / alpha_prev);
        
        for (size_t i = 0; i < sample.size(); i++) {
            sample[i] = std::sqrt(alpha_prev) * pred_x0[i] + sigma * model_output[i];
        }
    }

private:
    int num_steps_;
    float eta_;
    std::vector<float> alphas_cumprod_;
};

} // namespace schedulers

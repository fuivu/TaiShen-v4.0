#include <cmath>
#include <vector>

namespace schedulers {

// LCM (Latent Consistency Model) scheduler
class LCMScheduler {
public:
    LCMScheduler(int num_train_timesteps = 1000) 
        : num_steps_(num_train_timesteps) {
        // Precompute alphas_cumprod
        alphas_cumprod_.resize(num_steps_);
        for (int i = 0; i < num_steps_; i++) {
            float t = (float)i / num_steps_;
            float beta = 0.0001f + 0.0199f * t;
            float alpha = 1.0f - beta;
            alphas_cumprod_[i] = (i == 0) ? alpha : alphas_cumprod_[i-1] * alpha;
        }
    }
    
    std::vector<float> get_timesteps(int num_inference_steps) {
        std::vector<float> timesteps(num_inference_steps);
        for (int i = 0; i < num_inference_steps; i++) {
            timesteps[i] = (float)(num_steps_ - 1) * (num_inference_steps - 1 - i) / (num_inference_steps - 1);
        }
        return timesteps;
    }
    
    void step(std::vector<float>& sample, const std::vector<float>& model_output, int timestep) {
        int idx = std::min(timestep, num_steps_ - 1);
        float alpha = alphas_cumprod_[idx];
        float sqrt_alpha = std::sqrt(alpha);
        float sqrt_one_minus_alpha = std::sqrt(1.0f - alpha);
        
        for (size_t i = 0; i < sample.size(); i++) {
            sample[i] = sqrt_alpha * sample[i] + sqrt_one_minus_alpha * model_output[i];
        }
    }

private:
    int num_steps_;
    std::vector<float> alphas_cumprod_;
};

} // namespace schedulers

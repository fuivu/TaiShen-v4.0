#pragma once
/**
 * Double-Buffered Pipeline + DMA Overlap
 * GPU 计算与权重搬运并行（Hexagon-MLIR 风格）
 * 天玑 CIM 存算一体模式
 */
#include <vector>
#include <thread>
#include <atomic>
#include <functional>
#include <cstdint>

namespace locai::pipeline {

// ─── 双缓冲块 ───────────────────────────────────────
struct BufferBlock {
    void*  gpu_ptr;       // GPU 显存指针
    void*  cpu_ptr;       // CPU 主机指针
    size_t size;
    int    block_id;
    bool   in_use;
    bool   dma_active;
};

// ─── DMA 引擎 ────────────────────────────────────────
class DmaEngine {
public:
    DmaEngine(int num_blocks = 4);
    ~DmaEngine();

    void  submit_copy(int block_id, const void* src, size_t bytes);
    void  wait_complete(int block_id);
    void  wait_all();
    float bandwidth_gbps() const;

private:
    std::vector<std::thread>   workers_;
    std::atomic<int>           active_count_;
    int                        num_blocks_;
    // 模拟 DMA 带宽 (天玑 9400+ UFS 4.0 ≈ 6.4 GB/s)
    float                      peak_bw_gbps_ = 6.4f;
};

// ─── 双缓冲管线 ─────────────────────────────────────
class DoubleBufferedPipeline {
public:
    struct Config {
        int   num_blocks        = 4;
        size_t block_size       = 2 * 1024 * 1024;  // 2MB per block
        bool  enable_cim_mode   = false;  // 天玑 CIM 存算一体
        bool  enable_speculative = false; // 投机解码
        int   prefetch_distance = 2;
    };

    explicit DoubleBufferedPipeline(const Config& cfg);
    ~DoubleBufferedPipeline();

    // 初始化管线
    void init();

    // 提交一个计算块（异步）
    void submit_compute(int block_id,
                        std::function<void(void* gpu_ptr)> kernel_fn);

    // 提交 DMA 搬运（异步，与 compute 重叠）
    void submit_dma(int block_id, const void* src, size_t bytes);

    // 等待所有完成
    void synchronize();

    // 投机解码：草稿模型 + 验证
    struct SpeculativeConfig {
        int draft_tokens   = 4;
        int verify_tokens  = 8;
        float accept_threshold = 0.8f;
    };
    void enable_speculative(const SpeculativeConfig& sc);
    int  run_speculative_decode(const std::vector<int>& prompt_tokens,
                                 std::vector<int>& output_tokens);

    // 统计
    float compute_time_ms() const;
    float dma_time_ms() const;
    float overlap_efficiency() const;  // 0~1，越高越好
    float power_saving_estimate() const; // CIM 模式功耗节省

private:
    Config          cfg_;
    DmaEngine*      dma_;
    std::vector<BufferBlock> blocks_;
    std::atomic<bool> running_;
    // 统计
    mutable std::atomic<float> compute_ms_;
    mutable std::atomic<float> dma_ms_;
    SpeculativeConfig spec_cfg_;
    bool              spec_enabled_ = false;

    void prefetch_next_block(int current_id);
};

} // namespace locai::pipeline

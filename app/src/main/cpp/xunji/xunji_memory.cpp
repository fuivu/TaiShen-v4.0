/*
 * ═════════════════════════════════════════════════════════════
 *  迅集 (XunJi) — 内存管理实现  v4.0 TaiShen
 *  RAII · 内存池 · 泄漏追踪 · /proc/meminfo 解析
 * ═════════════════════════════════════════════════════════════
 */
#include "xunji_memory.h"
#include <fstream>
#include <sstream>
#include <chrono>
#include <thread>
#include <cstdio>

namespace xunji {

// ─── MemoryPool ──────────────────────────────────────────────
constexpr size_t MemoryPool::BUCKETS[6];

MemoryPool::MemoryPool() : totalAlloc_(0) {}
MemoryPool::~MemoryPool() { trim(); }

int MemoryPool::bucketIndex(size_t n) const {
    for (int i = 0; i < 6; ++i) if (n <= BUCKETS[i]) return i;
    return -1; // too large → direct malloc
}

void* MemoryPool::alloc(size_t n) {
    std::lock_guard<std::mutex> lk(mu_);
    int bi = bucketIndex(n);
    if (bi < 0) {
        void* p = malloc(n);
        if (p) totalAlloc_ += n;
        return p;
    }
    // find free
    for (auto& b : buckets_[bi]) {
        if (!b.in_use) { b.in_use = true; totalAlloc_ += b.size; return b.ptr; }
    }
    // create new
    void* p = malloc(BUCKETS[bi]);
    if (!p) return nullptr;
    buckets_[bi].push_back({p, true, BUCKETS[bi]});
    totalAlloc_ += BUCKETS[bi];
    return p;
}

void MemoryPool::free(void* p, size_t /*n*/) {
    if (!p) return;
    std::lock_guard<std::mutex> lk(mu_);
    for (int bi = 0; bi < 6; ++bi) {
        for (auto& b : buckets_[bi]) {
            if (b.ptr == p) {
                b.in_use = false;
                totalAlloc_ -= b.size;
                return;
            }
        }
    }
    // not pooled → direct free
    ::free(p);
}

void MemoryPool::trim() {
    std::lock_guard<std::mutex> lk(mu_);
    for (int bi = 0; bi < 6; ++bi) {
        auto it = buckets_[bi].begin();
        while (it != buckets_[bi].end()) {
            if (!it->in_use) { ::free(it->ptr); it = buckets_[bi].erase(it); }
            else ++it;
        }
    }
}

size_t MemoryPool::totalAllocated() const {
    std::lock_guard<std::mutex> lk(mu_);
    return totalAlloc_;
}

// ─── /proc/meminfo 解析 ─────────────────────────────────────
static long parse_meminfo(const char* key) {
    std::ifstream f("/proc/meminfo");
    std::string line;
    while (std::getline(f, line)) {
        if (line.rfind(key, 0) == 0) {
            std::istringstream iss(line);
            std::string k; long v;
            iss >> k >> v;
            return v; // kB
        }
    }
    return 0;
}

static long mem_total_kb()    { return parse_meminfo("MemTotal:"); }
static long mem_free_kb()     { return parse_meminfo("MemFree:"); }
static long mem_avail_kb()    { return parse_meminfo("MemAvailable:"); }
static long mem_buffers_kb()  { return parse_meminfo("Buffers:"); }
static long mem_cached_kb()   { return parse_meminfo("Cached:"); }

MemPressure xunji_mem_pressure() {
    long total = mem_total_kb();
    long avail = mem_avail_kb();
    if (total <= 0) return MemPressure::MEDIUM;
    float ratio = (float)avail / (float)total;
    if (ratio < 0.05f) return MemPressure::CRITICAL;
    if (ratio < 0.10f) return MemPressure::HIGH;
    if (ratio < 0.20f) return MemPressure::MEDIUM;
    return MemPressure::LOW;
}

long xunji_mem_available_kb() { return mem_avail_kb(); }

// ─── 安全分配 ───────────────────────────────────────────────
void* xunji_malloc(size_t n) {
    const int maxAttempts = 5;
    for (int i = 0; i < maxAttempts; ++i) {
        void* p = malloc(n);
        if (p) return p;
        long delay = 50L << i; // 50,100,200,400,800
        if (delay > 1000) delay = 1000;
        std::this_thread::sleep_for(std::chrono::milliseconds(delay));
    }
    return nullptr;
}

void xunji_free(void* p) { if (p) free(p); }

bool xunji_try_alloc(size_t n, void** out) {
    // 检查可用内存
    long avail = mem_avail_kb() * 1024;
    long reserved = 200L * 1024 * 1024; // 200MB 预留
    if ((long)n > avail - reserved) return false;
    void* p = xunji_malloc(n);
    if (!p) return false;
    *out = p; return true;
}

// ─── 泄漏追踪 ────────────────────────────────────────────────
LeakTracker& LeakTracker::instance() {
    static LeakTracker inst; return inst;
}

void LeakTracker::recordAlloc(void* p, size_t n, const char* file, int line) {
    std::lock_guard<std::mutex> lk(mu_);
    char buf[256]; snprintf(buf, sizeof(buf), "%s:%d (%zu bytes)", file, line, n);
    allocs_[p] = buf;
}

void LeakTracker::recordFree(void* p) {
    std::lock_guard<std::mutex> lk(mu_);
    allocs_.erase(p);
}

void LeakTracker::report() const {
    std::lock_guard<std::mutex> lk(mu_);
    if (allocs_.empty()) { printf("[LeakTracker] No leaks.\n"); return; }
    printf("[LeakTracker] %zu leaks:\n", allocs_.size());
    for (const auto& kv : allocs_) printf("  %p -> %s\n", kv.first, kv.second.c_str());
}

LeakTracker::~LeakTracker() { report(); }

// ─── 后台监控线程 ───────────────────────────────────────────
static std::thread monitor_thread;
static volatile bool monitor_running = false;

static void monitor_loop() {
    while (monitor_running) {
        auto p = xunji_mem_pressure();
        long avail = xunji_mem_available_kb();
        if (p == MemPressure::CRITICAL) {
            printf("[XunJi] CRITICAL memory: %ld kB available\n", avail);
        }
        std::this_thread::sleep_for(std::chrono::seconds(5));
    }
}

void xunji_start_monitor() {
    if (monitor_running) return;
    monitor_running = true;
    monitor_thread = std::thread(monitor_loop);
    monitor_thread.detach();
}

void xunji_stop_monitor() { monitor_running = false; }

void xunji_report_leaks() { LeakTracker::instance().report(); }

} // namespace xunji

/*
 * ═════════════════════════════════════════════════════════════
 *  迅集 (XunJi) — 内存管理  v4.0 TaiShen
 *  SafeAlloc<T> RAII · QuantAllocator · 6 级内存池 · 泄漏追踪
 * ═════════════════════════════════════════════════════════════
 */
#pragma once
#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <vector>
#include <mutex>
#include <unordered_map>
#include <string>

namespace xunji {

// ─── 压力等级 ──────────────────────────────────────────────
enum class MemPressure { LOW, MEDIUM, HIGH, CRITICAL };

// ─── SafeAlloc<T> RAII 模板 ─────────────────────────────────
template<typename T>
class SafeAlloc {
public:
    SafeAlloc() noexcept : ptr_(nullptr), size_(0) {}
    explicit SafeAlloc(size_t count) : ptr_(nullptr), size_(0) { allocate(count); }
    ~SafeAlloc() { release(); }

    // 禁止拷贝
    SafeAlloc(const SafeAlloc&) = delete;
    SafeAlloc& operator=(const SafeAlloc&) = delete;

    // 允许移动
    SafeAlloc(SafeAlloc&& o) noexcept : ptr_(o.ptr_), size_(o.size_) {
        o.ptr_ = nullptr; o.size_ = 0;
    }
    SafeAlloc& operator=(SafeAlloc&& o) noexcept {
        if (this != &o) { release(); ptr_ = o.ptr_; size_ = o.size_;
            o.ptr_ = nullptr; o.size_ = 0; }
        return *this;
    }

    bool allocate(size_t count) {
        release();
        if (count == 0) return true;
        // 指数退避重试
        const int maxAttempts = 5;
        for (int attempt = 0; attempt < maxAttempts; ++attempt) {
            ptr_ = static_cast<T*>(malloc(count * sizeof(T)));
            if (ptr_) { size_ = count; return true; }
            long delay = 50L << attempt; // 50,100,200,400,800
            if (delay > 1000) delay = 1000;
            // cross-platform sleep stub
#ifdef _WIN32
            Sleep((DWORD)delay);
#else
            struct timespec ts = { delay/1000, (delay%1000)*1000000 };
            nanosleep(&ts, nullptr);
#endif
        }
        return false;
    }

    void fill(const T& val) { for (size_t i = 0; i < size_; ++i) ptr_[i] = val; }
    void zero() { if (ptr_ && size_ > 0) memset(ptr_, 0, size_ * sizeof(T)); }

    T* get() noexcept { return ptr_; }
    const T* get() const noexcept { return ptr_; }
    size_t size() const noexcept { return size_; }
    bool empty() const noexcept { return ptr_ == nullptr; }

    T& operator[](size_t i) noexcept { return ptr_[i]; }
    const T& operator[](size_t i) const noexcept { return ptr_[i]; }

    T* release() noexcept { T* p = ptr_; ptr_ = nullptr; size_ = 0; return p; }

private:
    void free_ptr() { if (ptr_) { free(ptr_); ptr_ = nullptr; size_ = 0; } }
    T* ptr_;
    size_t size_;
};

// ─── QuantAllocator<T> STL 兼容 ─────────────────────────────
template<typename T>
struct QuantAllocator {
    using value_type = T;
    QuantAllocator() = default;
    template<typename U> QuantAllocator(const QuantAllocator<U>&) {}
    T* allocate(std::size_t n) {
        T* p = static_cast<T*>(malloc(n * sizeof(T)));
        if (!p) throw std::bad_alloc();
        return p;
    }
    void deallocate(T* p, std::size_t) { free(p); }
    template<typename U> bool operator==(const QuantAllocator<U>&) const { return true; }
    template<typename U> bool operator!=(const QuantAllocator<U>&) const { return false; }
};

// ─── 内存池 (6 级: 4KB/16KB/64KB/256KB/1MB/4MB) ──────────
class MemoryPool {
public:
    static constexpr size_t BUCKETS[6] = { 4*1024, 16*1024, 64*1024, 256*1024, 1024*1024, 4*1024*1024 };
    MemoryPool();
    ~MemoryPool();
    void* alloc(size_t n);
    void  free(void* p, size_t n);
    void  trim();
    size_t totalAllocated() const;
private:
    struct Bucket { void* ptr; bool in_use; size_t size; };
    std::vector<Bucket> buckets_[6];
    mutable std::mutex mu_;
    size_t totalAlloc_;
    int bucketIndex(size_t n) const;
};

// ─── 泄漏追踪器 (Debug) ─────────────────────────────────────
class LeakTracker {
public:
    static LeakTracker& instance();
    void  recordAlloc(void* p, size_t n, const char* file, int line);
    void  recordFree(void* p);
    void  report() const;
    ~LeakTracker();
private:
    LeakTracker() = default;
    mutable std::mutex mu_;
    std::unordered_map<void*, std::string> allocs_;
};

// ─── C API ───────────────────────────────────────────────────
extern "C" {
    void* xunji_malloc(size_t n);
    void  xunji_free(void* p);
    bool  xunji_try_alloc(size_t n, void** out_ptr);
    MemPressure xunji_mem_pressure();
    long  xunji_mem_available_kb();
    void  xunji_report_leaks();
    void  xunji_start_monitor();
    void  xunji_stop_monitor();
}

} // namespace xunji

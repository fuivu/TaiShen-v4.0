#include "memory_pool.h"
#include <cstdlib>
#include <cstring>
#include <android/log.h>

#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, "MemoryPool", __VA_ARGS__)

MemoryPool::MemoryPool(size_t max_pool_mb) 
    : max_pool_bytes_(max_pool_mb * 1024 * 1024), used_bytes_(0) {
    LOGI("MemoryPool created: %zu MB", max_pool_mb);
}

MemoryPool::~MemoryPool() { clear(); }

void* MemoryPool::allocate(const std::string& name, size_t size_bytes) {
    // Check if exists
    auto it = allocations_.find(name);
    if (it != allocations_.end()) {
        if (it->second.size >= size_bytes) {
            return it->second.ptr;
        }
        // Reallocate
        free(name);
    }
    
    // Check capacity
    if (used_bytes_ + size_bytes > max_pool_bytes_) {
        LOGI("WARNING: Pool full! Used: %zuMB, Requested: %zuMB, Max: %zuMB",
             used_bytes_ / (1024*1024), size_bytes / (1024*1024), max_pool_bytes_ / (1024*1024));
        // Try to free oldest (simple LRU would go here)
    }
    
    void* ptr = aligned_alloc(64, size_bytes);
    if (ptr) {
        memset(ptr, 0, size_bytes);
        allocations_[name] = {ptr, size_bytes};
        used_bytes_ += size_bytes;
        LOGI("Allocated '%s': %zu bytes (total used: %zuMB)", 
             name.c_str(), size_bytes, used_bytes_ / (1024*1024));
    }
    return ptr;
}

void MemoryPool::free(const std::string& name) {
    auto it = allocations_.find(name);
    if (it != allocations_.end()) {
        used_bytes_ -= it->second.size;
        ::free(it->second.ptr);
        allocations_.erase(it);
    }
}

void MemoryPool::clear() {
    for (auto& kv : allocations_) {
        ::free(kv.second.ptr);
    }
    allocations_.clear();
    used_bytes_ = 0;
}

size_t MemoryPool::getUsedMB() const {
    return used_bytes_ / (1024 * 1024);
}

size_t MemoryPool::getAvailableMB() const {
    size_t avail = max_pool_bytes_ > used_bytes_ ? max_pool_bytes_ - used_bytes_ : 0;
    return avail / (1024 * 1024);
}

void MemoryPool::resize(size_t new_max_mb) {
    max_pool_bytes_ = new_max_mb * 1024 * 1024;
    LOGI("Pool resized to %zu MB", new_max_mb);
}

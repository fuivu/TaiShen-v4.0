#pragma once
#include <unordered_map>
#include <string>
#include <cstddef>

class MemoryPool {
public:
    explicit MemoryPool(size_t max_pool_mb);
    ~MemoryPool();
    
    void* allocate(const std::string& name, size_t size_bytes);
    void free(const std::string& name);
    void clear();
    
    size_t getUsedMB() const;
    size_t getAvailableMB() const;
    void resize(size_t new_max_mb);
    
private:
    struct Allocation {
        void* ptr;
        size_t size;
    };
    
    std::unordered_map<std::string, Allocation> allocations_;
    size_t max_pool_bytes_;
    size_t used_bytes_;
};

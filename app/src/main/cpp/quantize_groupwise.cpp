#include <vector>
#include <cmath>
#include <algorithm>

namespace quantize {

// INT4 Group-wise quantization
// Each group of 32 elements shares scale & zero-point
// Packs 2 nibbles per byte

struct GroupWiseQuantResult {
    std::vector<uint8_t> packed_data;  // 2x INT4 per byte
    std::vector<float> scales;
    std::vector<int> zero_points;
    int group_size;
    int num_groups;
};

GroupWiseQuantResult quantize_int4_groupwise(const float* weights, size_t count, int group_size = 32) {
    int num_groups = (count + group_size - 1) / group_size;
    
    GroupWiseQuantResult result;
    result.packed_data.resize((count + 1) / 2);
    result.scales.resize(num_groups);
    result.zero_points.resize(num_groups);
    result.group_size = group_size;
    result.num_groups = num_groups;
    
    for (int g = 0; g < num_groups; g++) {
        int start = g * group_size;
        int end = std::min((int)count, start + group_size);
        
        // Find min/max in group
        float gmin = weights[start], gmax = weights[start];
        for (int i = start + 1; i < end; i++) {
            if (weights[i] < gmin) gmin = weights[i];
            if (weights[i] > gmax) gmax = weights[i];
        }
        
        float scale = (gmax > gmin) ? (gmax - gmin) / 15.0f : 1.0f;
        int zp = (int)roundf(-gmin / scale);
        zp = std::max(0, std::min(15, zp));
        
        result.scales[g] = scale;
        result.zero_points[g] = zp;
        
        for (int i = start; i < end; i++) {
            int q = (int)roundf(weights[i] / scale + zp);
            q = std::max(0, std::min(15, q));
            
            int byte_idx = i / 2;
            if (i % 2 == 0) {
                result.packed_data[byte_idx] = (uint8_t)(q & 0x0F);
            } else {
                result.packed_data[byte_idx] |= (uint8_t)((q & 0x0F) << 4);
            }
        }
    }
    
    return result;
}

} // namespace quantize

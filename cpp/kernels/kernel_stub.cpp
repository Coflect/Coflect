#include <cstddef>

namespace coflect {

// Placeholder kernel stub. Replace with real optimized kernels.
float sum_f32(const float* data, std::size_t n) {
    float out = 0.0f;
    for (std::size_t i = 0; i < n; ++i) {
        out += data[i];
    }
    return out;
}

}  // namespace coflect

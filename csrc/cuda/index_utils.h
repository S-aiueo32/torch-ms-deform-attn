#pragma once

#include <algorithm>
#include <cstdint>
#include <initializer_list>
#include <limits>
#include <vector>
#include <c10/util/Exception.h>

namespace ms_deform_attn {

// The CUDA kernels keep individual dimensions as int but use int64_t for
// tensor offsets. Leave headroom for one out-of-image interpolation neighbor.
inline int64_t checked_index_product(std::initializer_list<int64_t> factors) {
    constexpr int64_t limit = std::numeric_limits<int64_t>::max() / 2;
    int64_t result = 1;
    for (const int64_t factor : factors) {
        TORCH_CHECK(factor > 0 && result <= limit / factor,
                    "CUDA tensor size exceeds the safe 64-bit indexing range");
        result *= factor;
    }
    return result;
}

inline int cuda_blocks(int64_t work, int threads) {
    TORCH_CHECK(work > 0 && threads > 0 && threads <= 1024,
                "Invalid CUDA launch dimensions");
    // Do not cap the grid: backward kernels advance gradient pointers inside
    // CUDA_KERNEL_LOOP and require at most one iteration per thread.
    // Division first avoids overflow from rounding up with work + threads - 1.
    const int64_t blocks = (work - 1) / threads + 1;
    TORCH_CHECK(blocks <= std::numeric_limits<int>::max(),
                "CUDA work exceeds the grid dimension limit; reduce im2col_step");
    return static_cast<int>(blocks);
}

// Shared by the native CUDA entry points and CPU-only regression tests.
// dims = [batch, spatial_size, heads, channels, levels, queries, points].
// result = [chunk, value/sample/weight elements per batch, chunk output, blocks].
inline std::vector<int64_t> check_cuda_indexing(const std::vector<int64_t>& dims,
                                               int64_t step) {
    TORCH_CHECK(dims.size() == 7, "CUDA indexing expects seven dimensions");
    for (const int64_t dim : dims) {
        TORCH_CHECK(dim > 0 && dim <= std::numeric_limits<int>::max(),
                    "CUDA dimensions must be positive and fit in int32");
    }
    TORCH_CHECK(step > 0 && step <= std::numeric_limits<int>::max(),
                "im2col_step must be positive and fit in int32");
    const auto N = dims[0], S = dims[1], M = dims[2], D = dims[3];
    const auto L = dims[4], Q = dims[5], P = dims[6];
    const int64_t chunk = std::min(N, step);
    const int64_t per_value = checked_index_product({S, M, D});
    const int64_t per_weights = checked_index_product({Q, M, L, P});
    const int64_t per_locations = checked_index_product({per_weights, 2});
    checked_index_product({N, per_value});
    checked_index_product({N, per_locations});
    checked_index_product({N, Q, M, D});
    const int64_t work = checked_index_product({chunk, Q, M, D});
    const int blocks = cuda_blocks(work, static_cast<int>(std::min(D, int64_t(1024))));
    return {chunk, per_value, per_locations, per_weights, work, blocks};
}

} // namespace ms_deform_attn

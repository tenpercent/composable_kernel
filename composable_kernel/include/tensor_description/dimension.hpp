#ifndef CK_DIMENSION_HPP
#define CK_DIMENSION_HPP

#include "common_header.hpp"

namespace ck {

template <index_t Length>
struct Dimension
{
    __host__ __device__ static constexpr auto GetLength() { return Number<Length>{}; }
};

template <index_t Length, index_t Stride>
struct NativeDimension : Dimension<Length>
{
    __host__ __device__ static constexpr auto GetStride() { return Number<Stride>{}; }

    __host__ __device__ static constexpr index_t GetOffset(index_t id) { return id * Stride; }

    __host__ __device__ static constexpr index_t GetOffsetDiff(index_t id_diff)
    {
        return id_diff * Stride;
    }
};

} // namespace ck
#endif
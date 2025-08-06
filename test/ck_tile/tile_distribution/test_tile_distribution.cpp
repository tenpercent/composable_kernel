// SPDX-License-Identifier: MIT
// Copyright (c) Advanced Micro Devices, Inc. All rights reserved.

#include <gtest/gtest.h>
#include "ck_tile/host.hpp"
#include "ck_tile/core.hpp"
#include "ck_tile/host/kernel_launch.hpp"

namespace
{

struct HArgs
{
    ck_tile::index_t height;
    ck_tile::index_t width;

    ck_tile::index_t dim_block_h;
    ck_tile::index_t dim_block_w;
};

template<typename T>
struct Kernel
{
    using DataType = T;
    static constexpr ck_tile::index_t kBlockSize = 256;
    static constexpr ck_tile::index_t kBlockPerCu = 1;

    static constexpr ck_tile::index_t kBlockTileHeight = 64;
    static constexpr ck_tile::index_t kBlockTileWidth = 32;

    struct KArgs
    {
        DataType* output_ptr;
        ck_tile::index_t height;
        ck_tile::index_t width;
    };

    CK_TILE_HOST static constexpr dim3 BlockSize()
    {
        return dim3(kBlockSize, 1, 1);
    }

    CK_TILE_HOST static constexpr auto GridSize(const HArgs& host_args)
    {
        const size_t grid_size_x = (host_args.height + host_args.dim_block_h - 1) / host_args.dim_block_h;
        const size_t grid_size_y = (host_args.width + host_args.dim_block_w - 1) / host_args.dim_block_w;
        const size_t grid_size_z = 1;
        return dim3(grid_size_x, grid_size_y, grid_size_z);
    }

    CK_TILE_HOST static KArgs MakeKargs(DataType* output_ptr,
                                        const HArgs& host_args)
    {
        return {output_ptr, host_args.height, host_args.width};
    }

    CK_TILE_DEVICE void operator() (const KArgs kargs) const
    {
        ck_tile::ignore = kargs;
    }

};

}

class TestTileDistribution: public ::testing::Test
{
    protected:
    void Run(ck_tile::index_t height, ck_tile::index_t width)
    {
        using DataType = ck_tile::fp16_t;
        ck_tile::HostTensor<DataType> output_host({height, width});
        output_host.SetZero();
        ck_tile::DeviceMem output_device(output_host.get_element_space_size_in_bytes());

        output_device.ToDevice(output_host.data());

        using K = Kernel<DataType>;

        auto host_args = HArgs {height, width, K::kBlockTileHeight, K::kBlockTileWidth};

        ck_tile::launch_kernel(ck_tile::stream_config{}, 
                               ck_tile::make_kernel<K::kBlockSize, K::kBlockPerCu>(
                                   K{}, 
                                   K::GridSize(host_args), 
                                   K::BlockSize(), 
                                   0, 
                                   K::MakeKargs(static_cast<DataType*>(output_device.GetDeviceBuffer()), host_args)));

        output_device.FromDevice(output_host.data());
        std::cout << output_host << std::endl;
    }
};

TEST_F(TestTileDistribution, Dummy)
{
    Run(1024, 1024);
}

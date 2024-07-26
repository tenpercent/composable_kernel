// SPDX-License-Identifier: MIT
// Copyright (c) 2024, Advanced Micro Devices, Inc. All rights reserved.

#include <iostream>
#include <numeric>
#include <initializer_list>
#include <cstdlib>

#include "ck/ck.hpp"
#include "ck/tensor_operation/gpu/device/gemm_specialization.hpp"
#include "ck/tensor_operation/gpu/device/impl/device_gemm_multiple_d_xdl_cshuffle_v3.hpp"
#include "ck/tensor_operation/gpu/element/element_wise_operation.hpp"
#include "ck/tensor_operation/gpu/element/unary_element_wise_operation.hpp"

#include "ck/library/utility/device_memory.hpp"
#include "ck/library/utility/host_tensor.hpp"
#include "ck/library/utility/host_tensor_generator.hpp"
#include "ck/library/utility/literals.hpp"
#include "ck/library/reference_tensor_operation/cpu/reference_gemm.hpp"
#include "ck/library/utility/check_err.hpp"

#include "ck/utility/blkgemmpipe_scheduler.hpp"

template <ck::index_t... Is>
using S = ck::Sequence<Is...>;

using F16 = ck::half_t;
using FP8 = ck::f8_t;
using F32 = float;

using Row = ck::tensor_layout::gemm::RowMajor;
using Col = ck::tensor_layout::gemm::ColumnMajor;

using ADataType        = F16;
using BDataType        = F16;
using AccDataType      = F32;
using CShuffleDataType = F32;
using D0DataType       = F16;
using DsDataType       = ck::Tuple<D0DataType>;
using EDataType        = F16;

using ALayout = Row;
using BLayout = Row;
using D0Layout = Row;
using DsLayout = ck::Tuple<D0Layout>;
using ELayout  = Row;

using PassThrough = ck::tensor_operation::element_wise::PassThrough;
using Bilinear = ck::tensor_operation::element_wise::Bilinear;

using AElementOp   = PassThrough;
using BElementOp   = PassThrough;
using CDEElementOp = Bilinear;

using BlockGemmPipelineScheduler = ck::BlockGemmPipelineScheduler;
using GemmSpecialization = ck::tensor_operation::device::GemmSpecialization;
using BlockGemmPipelineVersion = ck::BlockGemmPipelineVersion;

using DeviceOpInstance = ck::tensor_operation::device::DeviceGemmMultiD_Xdl_CShuffle_V3<
// clang-format off
            /* a_layout */ ALayout,
            /* b_layout */ BLayout,
            /* ds_layouts */ DsLayout,
            /* c_layout */ ELayout,
            /* a_element_dtype */ ADataType,
            /* b_element_dtype */ BDataType,
            /* ds_element_dtypes */ DsDataType,
            /* c_element_dtype */ EDataType,
            /* acc_dtype */ AccDataType,
            /* c_shuffle_dtype */ CShuffleDataType,
            /* a_elementwise_op */ PassThrough,
            /* b_elementwise_op */ PassThrough,
            /* c_elementwise_op */ Bilinear,
            /* gemm_specialization */ GemmSpecialization::KPadding,
            /* block_size */ 128,
            /* m_per_block */ 64,
            /* n_per_block */ 16,
            /* k_per_block */ 64,
            /* a_k1 */ 8,
            /* b_k1 */ 4,
            /* m_per_xdl */ 16,
            /* n_per_xdl */ 16,
            /* m_xdl_per_wave */ 2,
            /* n_xdl_per_wave */ 1,
            /* a_block_transfer_thread_cluster_lengths_ak0_m_ak1 */ S<8, 16, 1>,
            /* a_block_transfer_thread_cluster_arrange_order */ S<1, 0, 2>,
            /* a_block_transfer_src_access_order */ S<1, 0, 2>,
            /* a_block_transfer_src_vector_dim */ 2,
            /* a_block_transfer_src_scalar_per_vector */ 8,
            /* a_block_transfer_dst_scalar_per_vector_ak1 */ 8,
            /* a_block_lds_extra_m */ 0,
            /* b_block_transfer_thread_cluster_lengths_bk0_n_bk1 */ S<16, 8, 1>,
            /* b_block_transfer_thread_cluster_arrange_order */ S<0, 2, 1>,
            /* b_block_transfer_src_access_order */ S<0, 2, 1>,
            /* b_block_transfer_src_vector_dim */ 1,
            /* b_block_transfer_src_scalar_per_vector */ 2,
            /* b_block_transfer_dst_scalar_per_vector_bk1 */ 4,
            /* b_block_lds_extra_n */ 0,
            /* c_shuffle_m_xdl_per_wave_per_shuffle */ 1,
            /* c_shuffle_n_xdl_per_wave_per_shuffle */ 1,
            /* c_shuffle_block_transfer_cluster_lengths_m_block_m_per_block_n_block_n_per_block */ S<1, 16, 1, 8>,
            /* c_shuffle_block_transfer_scalar_per_vector_n_per_block */ S<2, 2>,
            /* block_gemm_pipeline_scheduler */ BlockGemmPipelineScheduler::Interwave,
            /* block_gemm_pipeline_version */ BlockGemmPipelineVersion::v2>;
// clang-format on

int main(int argc, char* argv[])
{
    /* A: (M, K)
       B: (K, N)
       D: (N,)
       E: (M, N) := alpha * A @ B + beta * D (broadcasted along M)
    */
    bool do_verification = true;
    int init_method      = 1;
    bool time_kernel     = false;

    // GEMM shape
    ck::index_t M = 3840;
    ck::index_t N = 4096;
    ck::index_t K = 4096;

    ck::index_t StrideA = std::is_same_v<ck::tensor_layout::gemm::RowMajor, ALayout> ? K : M;
    ck::index_t StrideB = std::is_same_v<ck::tensor_layout::gemm::RowMajor, BLayout> ? N : K;
    ck::index_t StrideD = 0;
    ck::index_t StrideE = std::is_same_v<ck::tensor_layout::gemm::RowMajor, ELayout> ? N : M;

    if(argc == 1)
    {
        // use default case
    }
    else if(argc == 4)
    {
        do_verification = std::stoi(argv[1]);
        init_method     = std::stoi(argv[2]);
        time_kernel     = std::stoi(argv[3]);
    }
    else if(argc == 11)
    {
        do_verification = std::stoi(argv[1]);
        init_method     = std::stoi(argv[2]);
        time_kernel     = std::stoi(argv[3]);

        M = std::stoi(argv[4]);
        N = std::stoi(argv[5]);
        K = std::stoi(argv[6]);

        StrideA = std::stoi(argv[7]);
        StrideB = std::stoi(argv[8]);
        StrideD = std::stoi(argv[9]);
        StrideE = std::stoi(argv[10]);
    }
    else
    {
        printf("arg1: verification (0=no, 1=yes)\n");
        printf("arg2: initialization (0=no init, 1=integer value, 2=decimal value)\n");
        printf("arg3: time kernel (0=no, 1=yes)\n");
        printf("arg4 to 9: M (256x), N(128x), K(32x), StrideA, StrideB, StrideD, StrideE\n");
        exit(0);
    }

    auto f_host_tensor_descriptor =
        [](std::size_t row, std::size_t col, std::size_t stride, auto layout) {
            using namespace ck::literals;

            if(std::is_same<decltype(layout), ck::tensor_layout::gemm::RowMajor>::value)
            {
                return HostTensorDescriptor({row, col}, {stride, 1_uz});
            }
            else
            {
                return HostTensorDescriptor({row, col}, {1_uz, stride});
            }
        };

    Tensor<ADataType> a0_m_k(f_host_tensor_descriptor(M, K, StrideA, ALayout{}));
    Tensor<BDataType> b0_k_n(f_host_tensor_descriptor(K, N, StrideB, BLayout{}));
    Tensor<D0DataType> d0_m_n(f_host_tensor_descriptor(M, N, StrideD, D0Layout{}));
    Tensor<EDataType> e_m_n_host_result(f_host_tensor_descriptor(M, N, StrideE, ELayout{}));
    Tensor<EDataType> e_m_n_device_result(f_host_tensor_descriptor(M, N, StrideE, ELayout{}));

    std::cout << "a0_m_k: " << a0_m_k.mDesc << std::endl;
    std::cout << "b0_k_n: " << b0_k_n.mDesc << std::endl;
    std::cout << "d0_m_n: " << d0_m_n.mDesc << std::endl;
    std::cout << "e_m_n: " << e_m_n_host_result.mDesc << std::endl;

    switch(init_method)
    {
    case 0: break;
    case 1:
        a0_m_k.GenerateTensorValue(GeneratorTensor_2<ADataType>{-2, 2});
        b0_k_n.GenerateTensorValue(GeneratorTensor_2<BDataType>{0, 2});
        d0_m_n.GenerateTensorValue(GeneratorTensor_2<D0DataType>{0, 2});
        break;
    default:
        a0_m_k.GenerateTensorValue(GeneratorTensor_3<ADataType>{0.0, 1.0});
        b0_k_n.GenerateTensorValue(GeneratorTensor_3<BDataType>{-0.5, 0.5});
        d0_m_n.GenerateTensorValue(GeneratorTensor_3<D0DataType>{-0.5, 0.5});
    }

    DeviceMem a0_device_buf(sizeof(ADataType) * a0_m_k.mDesc.GetElementSpaceSize());
    DeviceMem b0_device_buf(sizeof(BDataType) * b0_k_n.mDesc.GetElementSpaceSize());
    DeviceMem d0_device_buf(sizeof(D0DataType) * d0_m_n.mDesc.GetElementSpaceSize());
    DeviceMem e_device_buf(sizeof(EDataType) * e_m_n_device_result.mDesc.GetElementSpaceSize());

    a0_device_buf.ToDevice(a0_m_k.mData.data());
    b0_device_buf.ToDevice(b0_k_n.mData.data());
    d0_device_buf.ToDevice(d0_m_n.mData.data());
    e_device_buf.ToDevice(e_m_n_device_result.mData.data());

    auto a_element_op   = AElementOp{};
    auto b_element_op   = BElementOp{};
    auto cde_element_op = CDEElementOp{ /* alpha */ 2.0, /* beta */ 0.125 };

    constexpr ck::index_t NumDTensor = DsDataType::Size();

    // do GEMM
    auto device_op = DeviceOpInstance{};
    auto invoker   = device_op.MakeInvoker();
    auto argument =
        device_op.MakeArgument(a0_device_buf.GetDeviceBuffer(),
                               b0_device_buf.GetDeviceBuffer(),
                               std::array<const void*, NumDTensor>{d0_device_buf.GetDeviceBuffer()},
                               e_device_buf.GetDeviceBuffer(),
                               M,
                               N,
                               K,
                               StrideA,
                               StrideB,
                               std::array<ck::index_t, NumDTensor>{StrideD},
                               StrideE,
                               a_element_op,
                               b_element_op,
                               cde_element_op);

    if(!device_op.IsSupportedArgument(argument))
    {
        throw std::runtime_error(
            "wrong! device_gemm with the specified compilation parameters does "
            "not support this GEMM problem");
    }

    float ave_time = invoker.Run(argument, StreamConfig{nullptr, time_kernel, 20, 50});

    std::size_t flop = std::size_t(2) * M * N * K;
    std::size_t num_btype =
        sizeof(ADataType) * M * K + sizeof(BDataType) * K * N + sizeof(EDataType) * M * N;

    float tflops = static_cast<float>(flop) / 1.E9 / ave_time;

    float gb_per_sec = num_btype / 1.E6 / ave_time;

    std::cout << "Perf: " << ave_time << " ms, " << tflops << " TFlops, " << gb_per_sec << " GB/s"
              << std::endl;

    e_device_buf.FromDevice(e_m_n_device_result.mData.data());

    if(do_verification)
    {
        Tensor<CShuffleDataType> c_m_n({M, N});

        using ReferenceGemmInstance = ck::tensor_operation::host::ReferenceGemm<ADataType,
                                                                                BDataType,
                                                                                CShuffleDataType,
                                                                                AccDataType,
                                                                                PassThrough,
                                                                                PassThrough,
                                                                                PassThrough>;
        auto ref_gemm               = ReferenceGemmInstance{};
        auto ref_invoker            = ref_gemm.MakeInvoker();

        auto ref_argument = ref_gemm.MakeArgument(
            a0_m_k, b0_k_n, c_m_n, PassThrough{}, PassThrough{}, PassThrough{});

        ref_invoker.Run(ref_argument);

        for(int m = 0; m < M; ++m)
        {
            for(int n = 0; n < N; ++n)
            {
                cde_element_op(e_m_n_host_result(m, n), c_m_n(m, n), d0_m_n(m, n));
            }
        }

        e_device_buf.FromDevice(e_m_n_device_result.mData.data());

        return ck::utils::check_err(e_m_n_device_result, e_m_n_host_result) ? 0 : 1;
    }

    return 0;
}

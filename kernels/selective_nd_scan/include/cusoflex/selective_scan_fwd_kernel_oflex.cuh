#pragma once

#include <c10/util/BFloat16.h>
#include <c10/util/Half.h>
#include <c10/cuda/CUDAException.h>  // For C10_CUDA_CHECK and C10_CUDA_KERNEL_LAUNCH_CHECK

#include <cub/block/block_load.cuh>
#include <cub/block/block_shuffle.cuh>
#include <cub/block/block_store.cuh>
#include <cub/block/block_scan.cuh>

#include "selective_scan_oflex.h"
#include "selective_scan_common.h"
#include "static_switch.h"

template<int kNThreads_, int kNItemsX_, int kNItemsY_, bool kIsEvenLen_, typename input_t_, typename weight_t_, typename output_t_>
struct Selective_Scan_fwd_kernel_traits {
    using input_t = input_t_;
    using weight_t = weight_t_;
    using output_t = output_t_;
    static constexpr int kNThreads = kNThreads_;
    // Setting MinBlocksPerMP to be 3 (instead of 2) for 128 threads improves occupancy.
    static constexpr int kMinBlocks = kNThreads < 128 ? 5 : 3;
    static constexpr int kNItemsX = kNItemsX_;
    static constexpr int kNItemsY = kNItemsY_;
    static constexpr int kNBytes = sizeof(input_t);
    static_assert((kNBytes == 2 && kNItemsX % 8 == 0) || (kNBytes == 4 && kNItemsX % 4 == 0));
    static constexpr int kNElts = kNBytes == 2 ? 8 : 4; // float4
    static constexpr int kNLoads = kNItemsX / kNElts;
    static constexpr bool kIsEvenLen = kIsEvenLen_;
    static constexpr bool kDirectIO = kIsEvenLen && kNLoads == 1;
    static constexpr int kNLoadsOutput = sizeof(output_t) * kNLoads / kNBytes;
    static constexpr bool kDirectIOOutput = kDirectIO && (kNLoadsOutput == 1);
    static constexpr int kNLoadsH = sizeof(weight_t) * kNLoads / kNBytes;
    static constexpr bool kDirectIOH = kDirectIO && (kNLoadsH == 1);

    using vec_t = typename BytesToType<kNBytes * kNElts>::Type;
    using scan_t = float2;
    using BlockLoadT = cub::BlockLoad<input_t, kNThreads, kNItemsX, cub::BLOCK_LOAD_WARP_TRANSPOSE>;
    using BlockLoadVecT = cub::BlockLoad<vec_t, kNThreads, kNLoads,
        !kDirectIO ? cub::BLOCK_LOAD_WARP_TRANSPOSE : cub::BLOCK_LOAD_DIRECT>;
    using BlockLoadWeightT = cub::BlockLoad<input_t, kNThreads, kNItemsX, cub::BLOCK_LOAD_WARP_TRANSPOSE>;
    using BlockLoadWeightVecT = cub::BlockLoad<vec_t, kNThreads, kNLoads,
        !kDirectIO ? cub::BLOCK_LOAD_WARP_TRANSPOSE  : cub::BLOCK_LOAD_DIRECT>;
    using BlockStoreT = cub::BlockStore<input_t, kNThreads, kNItemsX, cub::BLOCK_STORE_WARP_TRANSPOSE>;
    using BlockStoreVecT = cub::BlockStore<vec_t, kNThreads, kNLoads,
        !kDirectIO ? cub::BLOCK_STORE_WARP_TRANSPOSE : cub::BLOCK_STORE_DIRECT>;
    using BlockStoreOutputT = cub::BlockStore<output_t, kNThreads, kNItemsX, cub::BLOCK_STORE_WARP_TRANSPOSE>;
    using BlockStoreOutputVecT = cub::BlockStore<vec_t, kNThreads, kNLoadsOutput,
        !kDirectIOOutput ? cub::BLOCK_STORE_WARP_TRANSPOSE  : cub::BLOCK_STORE_DIRECT>;
    
    using BlockLoadHT = cub::BlockLoad<weight_t, kNThreads, kNItemsX, cub::BLOCK_LOAD_WARP_TRANSPOSE>;
    using BlockLoadHVecT = cub::BlockLoad<vec_t, kNThreads, kNLoadsH, cub::BLOCK_LOAD_WARP_TRANSPOSE>;
    using BlockStoreHT = cub::BlockStore<weight_t, kNThreads, kNItemsX, cub::BLOCK_STORE_WARP_TRANSPOSE>;
    using BlockStoreHVecT = cub::BlockStore<vec_t, kNThreads, kNLoadsH,
        !kDirectIOH ? cub::BLOCK_STORE_WARP_TRANSPOSE  : cub::BLOCK_STORE_DIRECT>;
    
    // using BlockScanT = cub::BlockScan<scan_t, kNThreads, cub::BLOCK_SCAN_RAKING_MEMOIZE>;
    // using BlockScanT = cub::BlockScan<scan_t, kNThreads, cub::BLOCK_SCAN_RAKING>;
    using BlockScanT = cub::BlockScan<scan_t, kNThreads, cub::BLOCK_SCAN_WARP_SCANS>;
    using BlockShuffleT = cub::BlockShuffle<weight_t, kNThreads>;
    static constexpr int kSmemIOSize = std::max({
        sizeof(typename BlockLoadT::TempStorage),
        sizeof(typename BlockLoadVecT::TempStorage),
        2 * sizeof(typename BlockLoadWeightT::TempStorage),
        2 * sizeof(typename BlockLoadWeightVecT::TempStorage),
        sizeof(typename BlockStoreT::TempStorage),
        sizeof(typename BlockStoreVecT::TempStorage),
        sizeof(typename BlockStoreOutputT::TempStorage),
        sizeof(typename BlockStoreOutputVecT::TempStorage),

        sizeof(typename BlockLoadHT::TempStorage),
        sizeof(typename BlockLoadHVecT::TempStorage),
        sizeof(typename BlockStoreHT::TempStorage),
        sizeof(typename BlockStoreHVecT::TempStorage)
    });
    static constexpr int kSmemSize = kSmemIOSize + sizeof(typename BlockScanT::TempStorage) + sizeof(typename BlockShuffleT::TempStorage);
};

template<typename Ktraits>
__global__ __launch_bounds__(Ktraits::kNThreads, Ktraits::kMinBlocks)
void selective_scan_fwd_kernel(SSMParamsBase params) {
    constexpr int kNThreads = Ktraits::kNThreads;
    constexpr int kNItemsX = Ktraits::kNItemsX;
    constexpr int kNItemsY = Ktraits::kNItemsY;
    using input_t = typename Ktraits::input_t;
    using weight_t = typename Ktraits::weight_t;
    using output_t = typename Ktraits::output_t;
    using scan_t = typename Ktraits::scan_t;

    // Shared memory.
    extern __shared__ char smem_[];
    auto& smem_load = reinterpret_cast<typename Ktraits::BlockLoadT::TempStorage&>(smem_);
    auto& smem_load_weight = reinterpret_cast<typename Ktraits::BlockLoadWeightT::TempStorage&>(smem_);
    auto& smem_load_weight1 = *reinterpret_cast<typename Ktraits::BlockLoadWeightT::TempStorage*>(smem_ + sizeof(typename Ktraits::BlockLoadWeightT::TempStorage));
    auto& smem_store = reinterpret_cast<typename Ktraits::BlockStoreT::TempStorage&>(smem_);
    auto& smem_store1 = reinterpret_cast<typename Ktraits::BlockStoreOutputT::TempStorage&>(smem_);
    auto& smem_load_h = reinterpret_cast<typename Ktraits::BlockLoadHT::TempStorage&>(smem_);
    auto& smem_store_h = reinterpret_cast<typename Ktraits::BlockStoreHT::TempStorage&>(smem_);
    auto& smem_scan = *reinterpret_cast<typename Ktraits::BlockScanT::TempStorage*>(smem_ + Ktraits::kSmemIOSize);
    auto& smem_shuffle = *reinterpret_cast<typename Ktraits::BlockShuffleT::TempStorage*>(reinterpret_cast<char *>(&smem_scan) + sizeof(typename Ktraits::BlockScanT::TempStorage));

    const int batch_id = blockIdx.x;
    const int dim_id = blockIdx.y;
    const int group_id = dim_id / (params.dim_ngroups_ratio);
    const int delta_group_id = dim_id / (params.dim_deltagroups_ratio);
    input_t *u = reinterpret_cast<input_t *>(params.u_ptr) + batch_id * params.u_batch_stride
        + dim_id * params.u_d_stride;
    input_t *delta = reinterpret_cast<input_t *>(params.delta_ptr) + batch_id * params.delta_batch_stride
        + delta_group_id * params.delta_d_stride;
    weight_t *A = reinterpret_cast<weight_t *>(params.A_ptr) + dim_id * params.A_d_stride;
    input_t *Bvar = reinterpret_cast<input_t *>(params.B_ptr) + batch_id * params.B_batch_stride + group_id * params.B_group_stride;
    input_t *Cvar = reinterpret_cast<input_t *>(params.C_ptr) + batch_id * params.C_batch_stride + group_id * params.C_group_stride;
    scan_t *x = reinterpret_cast<scan_t *>(params.x_ptr) + (batch_id * params.dim + dim_id) * params.dstate * params.nChunksX * params.nrows;
    scan_t *x_sd = nullptr;
    if (params.x_sd_ptr != nullptr) {
        x_sd = reinterpret_cast<scan_t *>(params.x_sd_ptr) + (batch_id * params.dim + dim_id) * params.dstate * params.nChunksX * params.nrows;
    }
    weight_t *x_v = reinterpret_cast<weight_t *>(params.x_v_ptr) + batch_id * params.x_v_batch_stride + dim_id * params.x_v_d_stride;

    float D_val = 0; // attention!
    if (params.D_ptr != nullptr) {
        D_val = reinterpret_cast<float *>(params.D_ptr)[dim_id];
    }
    float delta_bias = 0;
    if (params.delta_bias_ptr != nullptr) {
        delta_bias = reinterpret_cast<float *>(params.delta_bias_ptr)[delta_group_id];
    }

    // Load A_decay
    float A_decay = WEIGHT_DELTA_A_EXP;
    if (params.A_decay_ptr != nullptr) {
        A_decay = reinterpret_cast<float *>(params.A_decay_ptr)[0];
    } 

    constexpr int kChunkSizeX = kNThreads * kNItemsX;
    constexpr int kChunkSizeY = kNItemsY;
    const int nChunksX = params.nChunksX;
    const int nChunksY = params.nChunksY;
    // todo @longlong.yu 根据不同的 group_id 变化扫描顺序
    for (int chunk_y = 0; chunk_y < nChunksY; ++chunk_y) {
        for (int chunk_x = 0; chunk_x < nChunksX; ++chunk_x) {
            const int last_row = (kNItemsY < params.nrows - chunk_y * kChunkSizeY) ? kChunkSizeY : params.nrows - chunk_y * kChunkSizeY;
            // Load u, delta
            input_t u_vals[kNItemsY][kNItemsX], delta_vals_load[kNItemsY][kNItemsX];
            for (int nrow = 0; nrow < last_row; ++nrow) {
                load_input<Ktraits>(
                    u + (chunk_y * kChunkSizeY + nrow) * params.u_h_stride + chunk_x * kChunkSizeX,
                    u_vals[nrow], smem_load, params.ncols - chunk_x * kChunkSizeX
                );
                __syncthreads();
                load_input<Ktraits>(
                    delta + (chunk_y * kChunkSizeY + nrow) * params.delta_h_stride + chunk_x * kChunkSizeX, 
                    delta_vals_load[nrow], smem_load, params.ncols - chunk_x * kChunkSizeX
                );
                __syncthreads();
            } // end for kNItemsY

            // Load last row delta
            input_t last_delta_vals_[kNItemsX] = {0};
            float last_delta_vals[kNItemsX] = {0};
            if (chunk_y > 0) {
                load_input<Ktraits>(
                    delta + (chunk_y * kChunkSizeY - 1) * params.delta_h_stride + chunk_x * kChunkSizeX,
                    last_delta_vals_, smem_load, params.ncols - chunk_x * kChunkSizeX
                );
                __syncthreads();
                #pragma unroll
                for (int i = 0; i < kNItemsX; ++i) {
                    last_delta_vals[i] = float(last_delta_vals_[i]) + delta_bias;
                    if (params.delta_softplus) {
                        last_delta_vals[i] = last_delta_vals[i] <= 20.f ? log1pf(expf(last_delta_vals[i])) : last_delta_vals[i];
                    }
                }
            }

            // Load delta for previous column
            float delta_x[kNItemsY] = {0};
            if (threadIdx.x == 0 &&  chunk_x > 0) {
                for (int nrow = 0; nrow < last_row; ++nrow) {
                    delta_x[nrow] = float(delta[(chunk_y * kChunkSizeY + nrow) * params.delta_h_stride + chunk_x * kChunkSizeX - 1]);
                    delta_x[nrow] = float(delta_x[nrow]) + delta_bias;
                    if (params.delta_softplus) {
                        delta_x[nrow] = delta_x[nrow] <= 20.f ? log1pf(expf(delta_x[nrow])) : delta_x[nrow];
                    }
                }
            }

            float out_vals[kNItemsY][kNItemsX] = {0};
            for (int state_idx = 0; state_idx < params.dstate; ++state_idx) {
                constexpr float kLog2e = M_LOG2E;
                
                // Load A
                weight_t A_val = A[state_idx * params.A_dstate_stride];
                A_val *= kLog2e;

                // Init for first row
                float last_delta_A_exp_vals[kNItemsX];
                #pragma unroll
                for (int i = 0; i < kNItemsX; ++i) {
                    last_delta_A_exp_vals[i] = exp2f(last_delta_vals[i] * A_val) * A_decay;
                }
                scan_t last_running_prefix = chunk_x > 0 && chunk_y > 0 && threadIdx.x == 0 ? x[state_idx * nChunksX * params.nrows + (chunk_x - 1) * params.nrows + chunk_y * kChunkSizeY - 1] : make_float2(1.f, 0.f);
                weight_t last_row_data[kNItemsX] = {0};
                if (chunk_y > 0)  {
                    load_h<Ktraits>(
                        x_v + state_idx * params.x_v_dstate_stride + (chunk_y - 1) * params.x_v_chunk_stride + chunk_x * kChunkSizeX,
                        last_row_data, smem_load_h, params.ncols - chunk_x * kChunkSizeX
                    );
                    __syncthreads();
                }

                scan_t thread_data[kNItemsX] = {0};
                for (int nrow = 0; nrow < last_row; ++nrow) {
                    const int nrow_idx = chunk_y * kChunkSizeY + nrow;

                    float delta_vals[kNItemsX], delta_u_vals[kNItemsX];
                    #pragma unroll
                    for (int i = 0; i < kNItemsX; ++i) {
                        float u_val = float(u_vals[nrow][i]);
                        delta_vals[i] = float(delta_vals_load[nrow][i]) + delta_bias;
                        if (params.delta_softplus) {
                            delta_vals[i] = delta_vals[i] <= 20.f ? log1pf(expf(delta_vals[i])) : delta_vals[i];
                        }
                        delta_u_vals[i] = delta_vals[i] * u_val;
                        if (state_idx == 0) {
                            out_vals[nrow][i] += D_val * u_val;
                        }
                    }

                    // Load B, C
                    weight_t B_vals[kNItemsX], C_vals[kNItemsX];
                    load_weight<Ktraits>(
                        Bvar + state_idx * params.B_dstate_stride + nrow_idx * params.B_h_stride + chunk_x * kChunkSizeX, 
                        B_vals, smem_load_weight, (params.ncols - chunk_x * kChunkSizeX)
                    );
                    __syncthreads();
                    load_weight<Ktraits>(
                        Cvar + state_idx * params.C_dstate_stride + nrow_idx * params.C_h_stride + chunk_x * kChunkSizeX, 
                        C_vals, smem_load_weight1, (params.ncols - chunk_x * kChunkSizeX)
                    );
                    __syncthreads();
                    #pragma unroll
                    for (int i = 0; i < kNItemsX; ++i) {
                        const float delta_a_exp = exp2f(delta_vals[i] * A_val) * A_decay;

                        if (i == 0) {
                            float tmp_delta_a_exp = exp2f(delta_vals[kNItemsX - 1] * A_val) * A_decay;
                            Ktraits::BlockShuffleT(smem_shuffle).Offset(
                                tmp_delta_a_exp, tmp_delta_a_exp, -1
                            );
                            __syncthreads();
                            float last_prefix = 0.f;
                            Ktraits::BlockShuffleT(smem_shuffle).Offset(
                                last_row_data[kNItemsX - 1], last_prefix, -1
                            );
                            __syncthreads();

                            if (threadIdx.x == 0) {
                                tmp_delta_a_exp = exp2f(delta_x[nrow] * A_val) * A_decay;
                                last_prefix = last_running_prefix.y;
                            }
                            thread_data[i] = make_float2(
                                delta_a_exp, 
                                B_vals[i] * delta_u_vals[i] + delta_a_exp * (last_row_data[i]  - WEIGHT_LT_H * (last_delta_A_exp_vals[i] + tmp_delta_a_exp) * last_prefix)
                            ); 
                        } else {
                            thread_data[i] = make_float2(
                                delta_a_exp, 
                                B_vals[i] * delta_u_vals[i] + delta_a_exp * (last_row_data[i]  - WEIGHT_LT_H * (last_delta_A_exp_vals[i] + last_delta_A_exp_vals[i - 1]) * last_row_data[i-1])
                            );
                        }
                        last_delta_A_exp_vals[i] = delta_a_exp;

                        if constexpr (!Ktraits::kIsEvenLen) {  // So that the last state is correct
                            if (threadIdx.x * kNItemsX + i >= params.ncols - chunk_x * kChunkSizeX) {
                                thread_data[i] = make_float2(1.f, 0.f);
                            }
                        }
                    }

                    // Initialize running total
                    scan_t running_prefix;
                    // If we use WARP_SCAN then all lane 0 of all warps (not just thread 0) needs to read
                    // running_prefix = chunk_x > 0 && threadIdx.x % 32 == 0 ? smem_running_prefix[state_idx] : make_float2(1.f, 0.f);
                    // running_prefix = chunk_x > 0 && threadIdx.x % 32 == 0 ? x[(nrow * nChunksX + chunk_x - 1) * params.dstate + state_idx] : make_float2(1.f, 0.f);
                    running_prefix = chunk_x > 0 && threadIdx.x % 32 == 0 ? x[state_idx * nChunksX * params.nrows + (chunk_x - 1) * params.nrows + nrow_idx] : make_float2(1.f, 0.f);
                    last_running_prefix = running_prefix;
                    SSMScanPrefixCallbackOp<weight_t> prefix_op(running_prefix);
                    Ktraits::BlockScanT(smem_scan).InclusiveScan(
                        thread_data, thread_data, SSMScanOp<weight_t>(), prefix_op
                    );
                    // There's a syncthreads in the scan op, so we don't need to sync here.
                    // Unless there's only 1 warp, but then it's the same thread (0) reading and writing.
                    if (threadIdx.x == 0) {
                        x[state_idx * nChunksX * params.nrows + chunk_x * params.nrows + nrow_idx] = prefix_op.running_prefix;
                    }
                    
                    #pragma unroll
                    for (int i = 0; i < kNItemsX; ++i) {
                        out_vals[nrow][i] += thread_data[i].y * C_vals[i];
                        last_row_data[i] = thread_data[i].y;
                    }
                    __syncthreads();

                    if (x_sd != nullptr) {
                        #pragma unroll
                        for (int i = 0; i < kNItemsX; ++i) {
                            const float delta_a_exp = exp2f(delta_vals[i] * A_val) * A_decay;
                            thread_data[i] = make_float2(
                                delta_a_exp, 
                                B_vals[i] * delta_u_vals[i]
                            );
                            if constexpr (!Ktraits::kIsEvenLen) {  // So that the last state is correct
                                if (threadIdx.x * kNItemsX + i >= params.ncols - chunk_x * kChunkSizeX) {
                                    thread_data[i] = make_float2(1.f, 0.f);
                                }
                            }
                        }

                        // Initialize running total
                        scan_t running_prefix_sd;
                        running_prefix_sd = chunk_x > 0 && threadIdx.x % 32 == 0 ? x_sd[state_idx * nChunksX * params.nrows + (chunk_x - 1) * params.nrows + nrow_idx] : make_float2(1.f, 0.f);
                        SSMScanPrefixCallbackOp<weight_t> prefix_op_sd(running_prefix_sd);
                        Ktraits::BlockScanT(smem_scan).InclusiveScan(
                            thread_data, thread_data, SSMScanOp<weight_t>(), prefix_op_sd
                        );
                        if (threadIdx.x == 0) {
                            x_sd[state_idx * nChunksX * params.nrows + chunk_x * params.nrows + nrow_idx] = prefix_op_sd.running_prefix;
                        }
                        
                        #pragma unroll
                        for (int i = 0; i < kNItemsX; ++i) {
                            out_vals[nrow][i] += (0.25 * B_vals[i] * delta_u_vals[i] - thread_data[i].y) * C_vals[i];
                        }
                        __syncthreads();
                    }
                } // end for kNItemsY

                // Store last row
                store_h<Ktraits>(
                    x_v + state_idx * nChunksY * params.ncols + chunk_y * params.ncols + chunk_x * kChunkSizeX,
                    last_row_data, smem_store_h, params.ncols - chunk_x * kChunkSizeX
                );
                __syncthreads();
            } // end for dstate

            // Store out
            for (int nrow = 0; nrow < last_row; ++nrow) {
                output_t *out = reinterpret_cast<output_t *>(params.out_ptr) + batch_id * params.out_batch_stride
                    + dim_id * params.out_d_stride + (chunk_y * kChunkSizeY + nrow) * params.u_h_stride + chunk_x * kChunkSizeX;
                store_output1<Ktraits>(out, out_vals[nrow], smem_store1, params.ncols - chunk_x * kChunkSizeX);
                __syncthreads();
            }
        } // end for nChunkX
    } // end for nChunkY
}

template<int kNThreads, int kNItemsX, int kNItemsY, typename input_t, typename weight_t, typename output_t>
void selective_scan_fwd_launch(SSMParamsBase &params, cudaStream_t stream) {
    BOOL_SWITCH(params.ncols % (kNThreads * kNItemsX) == 0, kIsEvenLen, [&] {
        using Ktraits = Selective_Scan_fwd_kernel_traits<
            kNThreads, kNItemsX, kNItemsY, kIsEvenLen, input_t, weight_t, output_t
        >;
        constexpr int kSmemSize = Ktraits::kSmemSize;
        dim3 grid(params.batch, params.dim);
        auto kernel = &selective_scan_fwd_kernel<Ktraits>;
        if (kSmemSize >= 48 * 1024) {
            C10_CUDA_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, kSmemSize));
        }
        kernel<<<grid, Ktraits::kNThreads, kSmemSize, stream>>>(params);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    });
}

template<typename input_t, typename weight_t, typename output_t>
void selective_scan_fwd_cuda(SSMParamsBase &params, cudaStream_t stream) {
    constexpr int kMin = sizeof(input_t) == 2 ? 2 : 1;
    if (params.ncols <= 128) {
        selective_scan_fwd_launch<
            32, THREAD_ITEMS_X * kMin, THREAD_ITEMS_Y, input_t, weight_t, output_t
        >(params, stream);
    } else if (params.ncols <= 256) {
        selective_scan_fwd_launch<
            64, THREAD_ITEMS_X * kMin, THREAD_ITEMS_Y, input_t, weight_t, output_t
        >(params, stream);
    } else {
        selective_scan_fwd_launch<
            128, THREAD_ITEMS_X * kMin, THREAD_ITEMS_Y, input_t, weight_t, output_t
        >(params, stream);
    }
}

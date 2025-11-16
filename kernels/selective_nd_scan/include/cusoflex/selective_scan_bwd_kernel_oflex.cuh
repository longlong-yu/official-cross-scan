#pragma once

#include <algorithm>

#include <c10/util/BFloat16.h>
#include <c10/util/Half.h>
#include <c10/cuda/CUDAException.h>  // For C10_CUDA_CHECK and C10_CUDA_KERNEL_LAUNCH_CHECK
#include <ATen/cuda/Atomic.cuh>  // For atomicAdd on complex

#include <cub/block/block_load.cuh>
#include <cub/block/block_reduce.cuh>
#include <cub/block/block_scan.cuh>
#include <cub/block/block_shuffle.cuh>
#include <cub/block/block_store.cuh>

#include "selective_scan_oflex.h"
#include "selective_scan_common.h"
#include "reverse_scan.cuh"
#include "static_switch.h"

template<int kNThreads_, int kNItemsX_, int kNItemsY_, bool kIsEvenLen_, bool kDeltaSoftplus_, typename input_t_, typename weight_t_, typename output_t_>
struct Selective_Scan_bwd_kernel_traits {
    using input_t = input_t_;
    using weight_t = weight_t_;
    using output_t = output_t_;

    static constexpr int kNThreads = kNThreads_;
    static constexpr int kNItemsX = kNItemsX_;
    static constexpr int kNItemsY = kNItemsY_;
    static constexpr int MaxDState = MAX_DSTATE;
    static constexpr int kNBytes = sizeof(input_t);
    static_assert((kNBytes == 2 && kNItemsX % 8 == 0) || (kNBytes == 4 && kNItemsX % 4 == 0));
    static constexpr int kNElts = kNBytes == 2 ? 8 : 4; // float4
    static constexpr int kNLoads = kNItemsX / kNElts;
    static constexpr bool kIsEvenLen = kIsEvenLen_;
    static constexpr bool kDeltaSoftplus = kDeltaSoftplus_;
    // Setting MinBlocksPerMP to be 3 (instead of 2) for 128 threads with float improves occupancy.
    // For complex this would lead to massive register spilling, so we keep it at 2.
    static constexpr int kMinBlocks = kNThreads == 128 && 3;
    static constexpr int kNLoadsOutput = sizeof(output_t) * kNLoads / kNBytes;
    static constexpr int kNLoadsH = sizeof(weight_t) * kNLoads / kNBytes;
    static constexpr bool kDirectIO = kIsEvenLen && kNLoads == 1;
    static constexpr bool kDirectIOH = kDirectIO && (kNLoadsH == 1);

    using vec_t = typename BytesToType<kNBytes * kNElts>::Type;
    using scan_t = float2;
    using BlockLoadT = cub::BlockLoad<input_t, kNThreads, kNItemsX, cub::BLOCK_LOAD_WARP_TRANSPOSE>;
    using BlockLoadVecT = cub::BlockLoad<vec_t, kNThreads, kNLoads, cub::BLOCK_LOAD_WARP_TRANSPOSE>;
    using BlockLoadWeightT = cub::BlockLoad<input_t, kNThreads, kNItemsX, cub::BLOCK_LOAD_WARP_TRANSPOSE>;
    using BlockLoadWeightVecT = cub::BlockLoad<vec_t, kNThreads, kNLoads, cub::BLOCK_LOAD_WARP_TRANSPOSE>;
    using BlockLoadOutputT = cub::BlockLoad<output_t, kNThreads, kNItemsX, cub::BLOCK_LOAD_WARP_TRANSPOSE>;
    using BlockLoadOutputVecT = cub::BlockLoad<vec_t, kNThreads, kNLoadsOutput, cub::BLOCK_LOAD_WARP_TRANSPOSE>;
    using BlockStoreT = cub::BlockStore<input_t, kNThreads, kNItemsX, cub::BLOCK_STORE_WARP_TRANSPOSE>;
    using BlockStoreVecT = cub::BlockStore<vec_t, kNThreads, kNLoads, cub::BLOCK_STORE_WARP_TRANSPOSE>;
    
    using BlockLoadHT = cub::BlockLoad<weight_t, kNThreads, kNItemsX, cub::BLOCK_LOAD_WARP_TRANSPOSE>;
    using BlockLoadHVecT = cub::BlockLoad<vec_t, kNThreads, kNLoadsH, cub::BLOCK_LOAD_WARP_TRANSPOSE>;
    using BlockStoreHT = cub::BlockStore<weight_t, kNThreads, kNItemsX, cub::BLOCK_STORE_WARP_TRANSPOSE>;
    using BlockStoreHVecT = cub::BlockStore<vec_t, kNThreads, kNLoadsH,
        !kDirectIOH ? cub::BLOCK_STORE_WARP_TRANSPOSE  : cub::BLOCK_STORE_DIRECT>;
    
    // using BlockScanT = cub::BlockScan<scan_t, kNThreads, cub::BLOCK_SCAN_RAKING_MEMOIZE>;
    using BlockScanT = cub::BlockScan<scan_t, kNThreads, cub::BLOCK_SCAN_RAKING>;
    // using BlockScanT = cub::BlockScan<scan_t, kNThreads, cub::BLOCK_SCAN_WARP_SCANS>;
    using BlockReverseScanT = BlockReverseScan<scan_t, kNThreads>;
    using BlockReduceT = cub::BlockReduce<scan_t, kNThreads>;
    using BlockReduceFloatT = cub::BlockReduce<float, kNThreads>;
    using BlockExchangeT = cub::BlockExchange<float, kNThreads, kNItemsX>;
    using BlockShuffleT = cub::BlockShuffle<weight_t, kNThreads>;
    static constexpr int kSmemIOSize = std::max({
        sizeof(typename BlockLoadT::TempStorage),
        sizeof(typename BlockLoadVecT::TempStorage),
        2 * sizeof(typename BlockLoadWeightT::TempStorage),
        2 * sizeof(typename BlockLoadWeightVecT::TempStorage),
        sizeof(typename BlockLoadOutputT::TempStorage),
        sizeof(typename BlockLoadOutputVecT::TempStorage),
        sizeof(typename BlockStoreT::TempStorage),
        sizeof(typename BlockStoreVecT::TempStorage),

        sizeof(typename BlockLoadHT::TempStorage),
        sizeof(typename BlockLoadHVecT::TempStorage),
        sizeof(typename BlockStoreHT::TempStorage),
        sizeof(typename BlockStoreHVecT::TempStorage)
    });
    static constexpr int kSmemExchangeSize = 2 * sizeof(typename BlockExchangeT::TempStorage);
    static constexpr int kSmemReduceSize = sizeof(typename BlockReduceT::TempStorage);
    static constexpr int kSmemShuffleSize = sizeof(typename BlockShuffleT::TempStorage);
    static constexpr int kSmemSize = kSmemIOSize + kSmemExchangeSize + kSmemReduceSize + kSmemShuffleSize + sizeof(typename BlockScanT::TempStorage) + sizeof(typename BlockReverseScanT::TempStorage);
};

template<typename Ktraits>
__global__ __launch_bounds__(Ktraits::kNThreads, Ktraits::kMinBlocks)
void selective_scan_bwd_kernel(SSMParamsBwd params) {
    constexpr bool kDeltaSoftplus = Ktraits::kDeltaSoftplus;
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
    auto& smem_load1 = reinterpret_cast<typename Ktraits::BlockLoadOutputT::TempStorage&>(smem_);
    auto& smem_load_weight = reinterpret_cast<typename Ktraits::BlockLoadWeightT::TempStorage&>(smem_);
    auto& smem_load_weight1 = *reinterpret_cast<typename Ktraits::BlockLoadWeightT::TempStorage*>(smem_ + sizeof(typename Ktraits::BlockLoadWeightT::TempStorage));
    auto& smem_store = reinterpret_cast<typename Ktraits::BlockStoreT::TempStorage&>(smem_);
    auto& smem_load_h = reinterpret_cast<typename Ktraits::BlockLoadHT::TempStorage&>(smem_);
    auto& smem_store_h = reinterpret_cast<typename Ktraits::BlockStoreHT::TempStorage&>(smem_);
    auto& smem_exchange = *reinterpret_cast<typename Ktraits::BlockExchangeT::TempStorage*>(smem_ + Ktraits::kSmemIOSize);
    auto& smem_exchange1 = *reinterpret_cast<typename Ktraits::BlockExchangeT::TempStorage*>(smem_ + Ktraits::kSmemIOSize + sizeof(typename Ktraits::BlockExchangeT::TempStorage));
    auto& smem_reduce = *reinterpret_cast<typename Ktraits::BlockReduceT::TempStorage*>(reinterpret_cast<char *>(&smem_exchange) + Ktraits::kSmemExchangeSize);
    auto& smem_reduce_float = *reinterpret_cast<typename Ktraits::BlockReduceFloatT::TempStorage*>(&smem_reduce);
    auto& smem_scan = *reinterpret_cast<typename Ktraits::BlockScanT::TempStorage*>(reinterpret_cast<char *>(&smem_reduce) + Ktraits::kSmemReduceSize);
    auto& smem_reverse_scan = *reinterpret_cast<typename Ktraits::BlockReverseScanT::TempStorage*>(reinterpret_cast<char *>(&smem_scan) + sizeof(typename Ktraits::BlockScanT::TempStorage));
    auto& smem_shuffle = *reinterpret_cast<typename Ktraits::BlockShuffleT::TempStorage*>(reinterpret_cast<char *>(&smem_reverse_scan) + sizeof(typename Ktraits::BlockReverseScanT::TempStorage));
    weight_t *smem_da = reinterpret_cast<weight_t *>(smem_ + Ktraits::kSmemSize + Ktraits::MaxDState);

    const int batch_id = blockIdx.x;
    const int dim_id = blockIdx.y;
    const int group_id = dim_id / (params.dim_ngroups_ratio);
    const int delta_group_id = dim_id / (params.dim_deltagroups_ratio);
    input_t *u = reinterpret_cast<input_t *>(params.u_ptr) + batch_id * params.u_batch_stride
        + dim_id * params.u_d_stride;
    input_t *delta = reinterpret_cast<input_t *>(params.delta_ptr) + batch_id * params.delta_batch_stride
        + delta_group_id * params.delta_d_stride;

    weight_t *A = reinterpret_cast<weight_t *>(params.A_ptr) + dim_id * params.A_d_stride;
    float *dA_decay = params.dA_decay_ptr == nullptr ? nullptr : reinterpret_cast<float *>(params.dA_decay_ptr);
    input_t *Bvar = reinterpret_cast<input_t *>(params.B_ptr) + batch_id * params.B_batch_stride + group_id * params.B_group_stride;
    input_t *Cvar = reinterpret_cast<input_t *>(params.C_ptr) + batch_id * params.C_batch_stride + group_id * params.C_group_stride;
    weight_t *dA = reinterpret_cast<weight_t *>(params.dA_ptr) + dim_id * params.dA_d_stride;
    weight_t *dB = reinterpret_cast<weight_t *>(params.dB_ptr)
        + (batch_id * params.dB_batch_stride + group_id * params.dB_group_stride);
    weight_t *dC = reinterpret_cast<weight_t *>(params.dC_ptr)
        + (batch_id * params.dC_batch_stride + group_id * params.dC_group_stride);
    float *dD = params.dD_ptr == nullptr ? nullptr : reinterpret_cast<float *>(params.dD_ptr) + dim_id;
    float D_val = params.D_ptr == nullptr ? 0 : reinterpret_cast<float *>(params.D_ptr)[dim_id];
    float *ddelta_bias = params.ddelta_bias_ptr == nullptr ? nullptr : reinterpret_cast<float *>(params.ddelta_bias_ptr) + dim_id;
    float delta_bias = params.delta_bias_ptr == nullptr ? 0 : reinterpret_cast<float *>(params.delta_bias_ptr)[delta_group_id];
    scan_t *x = reinterpret_cast<scan_t *>(params.x_ptr) + (batch_id * params.dim + dim_id) * params.dstate * params.nChunksX * params.nrows;
    scan_t *x_sd = nullptr;
    if (params.x_sd_ptr != nullptr) {
        x_sd = reinterpret_cast<scan_t *>(params.x_sd_ptr) + (batch_id * params.dim + dim_id) * params.dstate * params.nChunksX * params.nrows;
    }
    weight_t *x_v = reinterpret_cast<weight_t *>(params.x_v_ptr) + batch_id * params.x_v_batch_stride + dim_id * params.x_v_d_stride;

    float dD_val = 0;
    float ddelta_bias_val = 0;

    output_t *dout = reinterpret_cast<output_t *>(params.dout_ptr) + batch_id * params.dout_batch_stride + dim_id * params.dout_d_stride;

    // Load A_decay
    float A_decay = WEIGHT_DELTA_A_EXP;
    if (params.A_decay_ptr != nullptr) {
        A_decay = reinterpret_cast<float *>(params.A_decay_ptr)[0];
    } 
    float dA_decay_val = 0;

    constexpr int kChunkSizeX = kNThreads * kNItemsX;
    constexpr int kChunkSizeY = kNItemsY;
    const int nChunksX = params.nChunksX;
    const int nChunksY = params.nChunksY;
    for (int chunk_y = nChunksY - 1; chunk_y >= 0; --chunk_y) {
        for (int chunk_x = nChunksX - 1; chunk_x >= 0; --chunk_x) {
            const int last_row = (kNItemsY < params.nrows - chunk_y * kChunkSizeY) ? kNItemsY : (params.nrows - chunk_y * kChunkSizeY);
            const int last_thread_x = (kNItemsX < params.ncols - chunk_x * kChunkSizeX) ? kNThreads - 1 : (params.ncols - chunk_x * kChunkSizeX + kNItemsX - 1) / kNItemsX - 1;
            const int last_col = (kNItemsX < params.ncols - chunk_x * kChunkSizeX - threadIdx.x * kNItemsX) ? kNItemsX : (params.ncols - chunk_x * kChunkSizeX - threadIdx.x * kNItemsX);
            // Load u, delta, dout
            input_t u_vals[kNItemsY][kNItemsX], delta_vals_load[kNItemsY][kNItemsX];
            float dout_vals[kNItemsY][kNItemsX];
            for (int nrow = 0; nrow < last_row; ++nrow) {
                const int nrow_idx = chunk_y * kChunkSizeY + nrow;
                load_input<Ktraits>(
                    u + nrow_idx * params.u_h_stride + chunk_x * kChunkSizeX,
                    u_vals[nrow], smem_load, params.ncols - chunk_x * kChunkSizeX
                );
                __syncthreads();
                load_input<Ktraits>(
                    delta + nrow_idx * params.delta_h_stride + chunk_x * kChunkSizeX, 
                    delta_vals_load[nrow], smem_load, params.ncols - chunk_x * kChunkSizeX
                );
                __syncthreads();

                if constexpr (std::is_same_v<output_t, input_t>) {
                    input_t dout_vals_load[kNItemsX];
                    load_input<Ktraits>(
                        reinterpret_cast<input_t *>(dout) + nrow_idx * params.dout_h_stride + chunk_x * kChunkSizeX, 
                        dout_vals_load, smem_load, params.ncols - chunk_x * kChunkSizeX
                    );
                    Converter<typename Ktraits::input_t, kNItemsX>::to_float(dout_vals_load, dout_vals[nrow]);
                } else {
                    static_assert(std::is_same_v<output_t, float>);
                    load_output<Ktraits>(
                        dout + nrow_idx * params.dout_h_stride + chunk_x * kChunkSizeX, 
                        dout_vals[nrow], smem_load1, params.ncols - chunk_x * kChunkSizeX
                    );
                }
                __syncthreads();
            }

            float ddelta_vals[kNItemsY][kNItemsX] = {0}, du_vals[kNItemsY][kNItemsX] = {0};
            float ddelta_vals_x[kNItemsX] = {0}, ddelta_vals_y[kNItemsY] = {0};

            for (int nrow = 0; nrow < last_row; ++nrow) { 
                #pragma unroll
                for (int i = 0; i < kNItemsX; ++i) {
                    // Calculate du
                    du_vals[nrow][i] += D_val * dout_vals[nrow][i];
                    dD_val += dout_vals[nrow][i] * float(u_vals[nrow][i]);
                }
            }

            // Load last chunk row delta for forward scan
            input_t last_delta_vals_[kNItemsX] = {0}, next_delta_vals_[kNItemsX] = {0};
            float last_delta_vals[kNItemsX] = {0}, next_delta_vals[kNItemsX] = {0};
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
            
            // Load next chunk row delta for backward scan
            if (chunk_y < nChunksY - 1) {
                load_input<Ktraits>(
                    delta + ((chunk_y + 1) * kChunkSizeY) * params.delta_h_stride + chunk_x * kChunkSizeX,
                    next_delta_vals_, smem_load, params.ncols - chunk_x * kChunkSizeX
                );
                __syncthreads();
                #pragma unroll
                for (int i = 0; i < kNItemsX; ++i) {
                    next_delta_vals[i] = float(next_delta_vals_[i]) + delta_bias;
                    if (params.delta_softplus) {
                        next_delta_vals[i] = next_delta_vals[i] <= 20.f ? log1pf(expf(next_delta_vals[i])) : next_delta_vals[i];
                    }
                }
            }

            // Load delta for previous column
            float last_delta_x_[kNItemsY] = {0}, last_delta_x[kNItemsY] = {0};
            if (threadIdx.x == 0 && chunk_x > 0) {
                for (int nrow = 0; nrow < last_row; ++nrow) {
                    last_delta_x_[nrow] = float(delta[(chunk_y * kChunkSizeY + nrow) * params.delta_h_stride + chunk_x * kChunkSizeX - 1]);
                    last_delta_x[nrow] = float(last_delta_x_[nrow]) + delta_bias;
                    if (params.delta_softplus) {
                        last_delta_x[nrow] = last_delta_x[nrow] <= 20.f ? log1pf(expf(last_delta_x[nrow])) : last_delta_x[nrow];
                    }
                }
            }
            // Load delta for next column
            float delta_x[kNItemsY + 1] = {0};
            if (threadIdx.x == kNThreads - 1 && chunk_x < nChunksX - 1) {
                for (int nrow = 0; nrow < (chunk_y < nChunksY - 1 ? last_row + 1 : last_row); ++nrow) { 
                    delta_x[nrow] = float(delta[(chunk_y * kChunkSizeY + nrow) * params.delta_h_stride + (chunk_x + 1) * kChunkSizeX]);
                    delta_x[nrow] = float(delta_x[nrow]) + delta_bias;
                    if (params.delta_softplus) {
                        delta_x[nrow] = delta_x[nrow] <= 20.f ? log1pf(expf(delta_x[nrow])) : delta_x[nrow];
                    }
                }
            }

            for (int state_idx = 0; state_idx < params.dstate; ++state_idx) {
                constexpr float kLog2e = M_LOG2E;
                
                // Load A
                weight_t A_val = A[state_idx * params.A_dstate_stride];
                weight_t A_scaled = A_val * kLog2e;

                // Load B
                weight_t B_vals[kNItemsY][kNItemsX];
                for (int nrow = 0; nrow < last_row; ++nrow) {
                    load_weight<Ktraits>(
                        Bvar + state_idx * params.B_dstate_stride + (chunk_y * kChunkSizeY + nrow) * params.B_h_stride + chunk_x * kChunkSizeX, 
                        B_vals[nrow], smem_load_weight, (params.ncols - chunk_x * kChunkSizeX)
                    );
                    __syncthreads();
                }

                weight_t dA_val = 0;

                // Init for first row
                float last_delta_A_exp_vals[kNItemsX];
                #pragma unroll
                for (int i = 0; i < kNItemsX; ++i) {
                    last_delta_A_exp_vals[i] = exp2f(last_delta_vals[i] * A_scaled) * A_decay;
                }
                // Load last running prefix
                scan_t last_running_prefix[kNItemsY + 1];
                for (int nrow = 0; nrow < last_row + 1; ++nrow) {
                    if (chunk_x > 0 && threadIdx.x % 32 == 0 && nrow + chunk_y > 0) {
                        last_running_prefix[nrow] = x[state_idx * nChunksX * params.nrows + (chunk_x - 1) * params.nrows + chunk_y * kChunkSizeY + nrow - 1];
                    } else {
                        last_running_prefix[nrow] = make_float2(1.f, 0.f);
                    }
                }
                weight_t last_row_data[kNItemsX] = {0};
                if (chunk_y > 0)  {
                    load_h<Ktraits>(
                        x_v + state_idx * params.x_v_dstate_stride + (chunk_y - 1) * params.x_v_chunk_stride+ chunk_x * kChunkSizeX,
                        last_row_data, smem_load_h, params.ncols - chunk_x * kChunkSizeX
                    );
                    __syncthreads();
                }

                // Forward scan
                scan_t thread_data[kNItemsY][kNItemsX] = {0};
                for (int nrow = 0; nrow < last_row; ++nrow) {
                    float delta_vals[kNItemsX], delta_u_vals[kNItemsX];
                    #pragma unroll
                    for (int i = 0; i < kNItemsX; ++i) {
                        float u_val = float(u_vals[nrow][i]);
                        delta_vals[i] = float(delta_vals_load[nrow][i]) + delta_bias;
                        if (params.delta_softplus) {
                            delta_vals[i] = delta_vals[i] <= 20.f ? log1pf(expf(delta_vals[i])) : delta_vals[i];
                        }
                        delta_u_vals[i] = delta_vals[i] * u_val;
                    }

                    #pragma unroll
                    for (int i = 0; i < kNItemsX; ++i) {
                        const float delta_a_exp = exp2f(delta_vals[i] * A_scaled) * A_decay;
                        const float tmp_1 = nrow == 0 ? last_row_data[i] : thread_data[nrow - 1][i].y;

                        if (i == 0) {
                            float tmp_delta_a_exp = exp2f(delta_vals[kNItemsX - 1] * A_scaled) * A_decay;
                            Ktraits::BlockShuffleT(smem_shuffle).Offset(
                                tmp_delta_a_exp, tmp_delta_a_exp, -1
                            );
                            __syncthreads();
                            float last_prefix = 0.f;
                            const float tmp_3 = nrow == 0 ? last_row_data[kNItemsX - 1] : thread_data[nrow - 1][kNItemsX - 1].y;
                            Ktraits::BlockShuffleT(smem_shuffle).Offset(
                                tmp_3, last_prefix, -1
                            );
                            __syncthreads();

                            if (threadIdx.x == 0) {
                                tmp_delta_a_exp = exp2f(last_delta_x[nrow] * A_scaled) * A_decay;
                                last_prefix = last_running_prefix[nrow].y;
                            }
                            thread_data[nrow][i] = make_float2(
                                delta_a_exp, 
                                B_vals[nrow][i] * delta_u_vals[i] + delta_a_exp * (tmp_1  - WEIGHT_LT_H * (last_delta_A_exp_vals[i] + tmp_delta_a_exp) * last_prefix)
                            );
                        } else {
                            const float tmp_2 = nrow == 0 ? last_row_data[i - 1] : thread_data[nrow - 1][i - 1].y;
                            thread_data[nrow][i] = make_float2(
                                delta_a_exp, 
                                B_vals[nrow][i] * delta_u_vals[i] + delta_a_exp * (tmp_1  - WEIGHT_LT_H * (last_delta_A_exp_vals[i] + last_delta_A_exp_vals[i - 1]) * tmp_2)
                            );
                        }
                        last_delta_A_exp_vals[i] = delta_a_exp;

                        if constexpr (!Ktraits::kIsEvenLen) {  // So that the last state is correct
                            if (threadIdx.x * kNItemsX + i >= params.ncols - chunk_x * kChunkSizeX) {
                                thread_data[nrow][i] = make_float2(1.f, 0.f);
                            }
                        }
                    }

                    // Initialize running total
                    scan_t running_prefix = last_running_prefix[nrow + 1];

                    SSMScanPrefixCallbackOp<weight_t> prefix_op(running_prefix);
                    Ktraits::BlockScanT(smem_scan).InclusiveScan(
                        thread_data[nrow], thread_data[nrow], SSMScanOp<weight_t>(), prefix_op
                    );
                    __syncthreads();
                } // end for kNItemsY

                // Backward scan
                // Init for first row
                float next_delta_A_exp_vals[kNItemsX];
                #pragma unroll
                for (int i = 0; i < kNItemsX; ++i) {
                    next_delta_A_exp_vals[i] = exp2f(next_delta_vals[i] * A_scaled) * A_decay;
                }
                scan_t next_running_postfix = chunk_x < nChunksX - 1 && chunk_y < nChunksY - 1 && threadIdx.x == last_thread_x ? x[state_idx * nChunksX * params.nrows + (chunk_x + 1) * params.nrows + (chunk_y + 1) * kChunkSizeY] : make_float2(1.f, 0.f);
                weight_t next_row_data[kNItemsX] = {0};
                if (chunk_y < nChunksY - 1)  {
                    load_h<Ktraits>(
                        x_v + state_idx * nChunksY * params.ncols + (chunk_y + 1) * params.ncols + chunk_x * kChunkSizeX,
                        next_row_data, smem_load_h, params.ncols - chunk_x * kChunkSizeX
                    );
                    __syncthreads();
                }

                scan_t thread_reverse_data[kNItemsX] = {0};
                for (int nrow = last_row - 1; nrow >= 0; --nrow) {
                    const int nrow_idx = chunk_y * kChunkSizeY + nrow;

                    float delta_vals[kNItemsX];
                    #pragma unroll
                    for (int i = 0; i < kNItemsX; ++i) {
                        delta_vals[i] = float(delta_vals_load[nrow][i]) + delta_bias;
                        if constexpr (kDeltaSoftplus) {
                            delta_vals[i] = delta_vals[i] <= 20.f ? log1pf(expf(delta_vals[i])) : delta_vals[i];
                        }
                    }
                    
                    // Load C
                    weight_t C_vals[kNItemsX];
                    load_weight<Ktraits>(
                        Cvar + state_idx * params.C_dstate_stride + nrow_idx * params.C_h_stride + chunk_x * kChunkSizeX, 
                        C_vals, smem_load_weight1, (params.ncols - chunk_x * kChunkSizeX)
                    );
                    __syncthreads();

                    #pragma unroll
                    for (int i = 0; i < kNItemsX; ++i) {
                        float delta_a_exp = exp2f(delta_vals[i] * A_scaled) * A_decay;

                        if (i == 0) {
                            Ktraits::BlockShuffleT(smem_shuffle).Offset(
                                delta_a_exp, thread_reverse_data[kNItemsX - 1].x, 1
                            );
                            __syncthreads();
                        } else {
                            thread_reverse_data[i - 1].x = delta_a_exp;
                        }
                    }

                    if (threadIdx.x == last_thread_x) {
                        thread_reverse_data[last_col - 1].x = exp2f(delta_x[nrow] * A_scaled) * A_decay;
                    }

                    #pragma unroll
                    for (int i = 0; i < kNItemsX; ++i) {
                        const float delta_a_exp = exp2f(delta_vals[i] * A_scaled) * A_decay;
                        
                        float h3_x, h3_y;
                        if (i == kNItemsX - 1) {
                            float tmp_delta_a_exp = next_delta_A_exp_vals[0];
                            Ktraits::BlockShuffleT(smem_shuffle).Offset(
                                tmp_delta_a_exp, tmp_delta_a_exp, 1
                            );
                            if (threadIdx.x == last_thread_x) {
                                tmp_delta_a_exp = exp2f(delta_x[nrow + 1] * A_scaled) * A_decay;
                            }
                            __syncthreads();

                            float next_prefix = 0.f;
                            Ktraits::BlockShuffleT(smem_shuffle).Offset(
                                next_row_data[0], next_prefix, 1
                            );
                            __syncthreads();
                            if (threadIdx.x == last_thread_x) {
                                next_prefix = next_running_postfix.y;
                            }
                            __syncthreads();
                            h3_x = tmp_delta_a_exp * WEIGHT_LT_H * thread_reverse_data[i].x * next_prefix;
                            h3_y = tmp_delta_a_exp * WEIGHT_LT_H * next_delta_A_exp_vals[i] * next_prefix;
                        } else {
                            h3_x = next_delta_A_exp_vals[i + 1] * WEIGHT_LT_H * thread_reverse_data[i].x * next_row_data[i + 1];
                            h3_y = next_delta_A_exp_vals[i + 1] * WEIGHT_LT_H * next_delta_A_exp_vals[i] * next_row_data[i + 1];
                        }
                        thread_reverse_data[i].y = dout_vals[nrow][i] * C_vals[i] + next_row_data[i] * next_delta_A_exp_vals[i] - h3_x - h3_y;

                        if constexpr (!Ktraits::kIsEvenLen) {  // So that the last state is correct
                            if (threadIdx.x * kNItemsX + i >= params.ncols - chunk_x * kChunkSizeX) {
                                thread_reverse_data[i] = make_float2(1.f, 0.f);
                            }
                        }
                    }

                    // Initialize running total
                    scan_t running_postfix = chunk_x < nChunksX - 1 && threadIdx.x % 32 == 0 ? x[state_idx * nChunksX * params.nrows + (chunk_x + 1) * params.nrows + nrow_idx] : make_float2(1.f, 0.f);
                    next_running_postfix = running_postfix;
                    Ktraits::BlockShuffleT(smem_shuffle).Offset(
                        running_postfix.y, next_running_postfix.y, - last_thread_x
                    );
                    __syncthreads();
                    SSMScanPrefixCallbackOp<weight_t> postfix_op(running_postfix);
                    Ktraits::BlockReverseScanT(smem_reverse_scan).InclusiveReverseScan(
                        thread_reverse_data, thread_reverse_data, SSMScanOp<weight_t>(), postfix_op
                    );
                    if (threadIdx.x == 0) { 
                        x[state_idx * nChunksX * params.nrows + chunk_x * params.nrows + nrow_idx] =  postfix_op.running_prefix;
                    }
                    __syncthreads();

                    // Calculate du, ddelta, dB, dC
                    weight_t dB_vals[kNItemsX], dC_vals[kNItemsX];
                    {
                        weight_t tmp = 0;
                        Ktraits::BlockShuffleT(smem_shuffle).Offset(
                            thread_data[nrow][kNItemsX - 1].y, tmp, -1
                        );
                        if (threadIdx.x == 0 && chunk_x > 0) {
                            tmp = x[state_idx * nChunksX * params.nrows + (chunk_x - 1) * params.nrows + nrow_idx].y;
                        }
                        __syncthreads();
                        float delta_a_exps[kNItemsX];
                        #pragma unroll
                        for (int i = kNItemsX - 1; i >= 0; --i) {
                            delta_a_exps[i] = exp2f(delta_vals[i] * A_scaled) * A_decay;
                        }
                        #pragma unroll
                        for (int i = kNItemsX - 1; i >= 0; --i) {
                            const float dx = thread_reverse_data[i].y;
                            const float ddelta_u = dx * B_vals[nrow][i];
                            du_vals[nrow][i] += ddelta_u * delta_vals[i];
                            const float a = thread_data[nrow][i].y - (delta_vals[i] * float(u_vals[nrow][i]) * B_vals[nrow][i]);
                            ddelta_vals[nrow][i] += ddelta_u * float(u_vals[nrow][i]) + dx * A_val * a;

                            dA_val += dx * delta_vals[i] * a;
                            dA_decay_val += dx * a / A_decay;

                            float h3 = 0;
                            if (nrow == 0 && i == 0) {
                                Ktraits::BlockShuffleT(smem_shuffle).Offset(
                                    last_row_data[kNItemsX - 1], h3, -1
                                );
                                if (threadIdx.x == 0) {
                                    h3 = last_running_prefix[nrow].y;
                                }
                                __syncthreads();
                            } else if (nrow == 0) {
                                h3 = last_row_data[i - 1];
                            } else if (i == 0) {
                                Ktraits::BlockShuffleT(smem_shuffle).Offset(
                                    thread_data[nrow - 1][kNItemsX - 1].y, h3, -1
                                );
                                if (threadIdx.x == 0) {
                                    h3 = last_running_prefix[nrow].y;
                                }
                                    __syncthreads();
                            } else {
                                h3 = thread_data[nrow - 1][i - 1].y;
                            }

                            if (nrow == 0 && i < last_col) {
                                float tmp_delta = last_delta_vals[i];

                                dA_val += -dx * WEIGHT_LT_H * h3 * delta_a_exps[i] * tmp_delta * exp2f(tmp_delta * A_scaled) * A_decay;
                                dA_decay_val += -dx * WEIGHT_LT_H * h3 * delta_a_exps[i] * exp2f(tmp_delta * A_scaled);
                                ddelta_vals_x[i] += -dx * WEIGHT_LT_H * h3 * A_val * delta_a_exps[i] * exp2f(tmp_delta * A_scaled) * A_decay;
                            } else if (i < last_col) {
                                float tmp_delta = float(delta_vals_load[nrow - 1][i]) + delta_bias;
                                if constexpr (kDeltaSoftplus) {
                                   tmp_delta = tmp_delta <= 20.f ? log1pf(expf(tmp_delta)) : tmp_delta;
                                }
                                dA_val += -dx * WEIGHT_LT_H * h3 * delta_a_exps[i] * tmp_delta * exp2f(tmp_delta * A_scaled) * A_decay;
                                dA_decay_val += -dx * WEIGHT_LT_H * h3 * delta_a_exps[i] * exp2f(tmp_delta * A_scaled);
                                ddelta_vals[nrow - 1][i] += -dx * WEIGHT_LT_H * h3 * A_val * delta_a_exps[i] * exp2f(tmp_delta * A_scaled) * A_decay;
                            }
                            if (i == 0) {
                                float tmp_delta;
                                Ktraits::BlockShuffleT(smem_shuffle).Offset(
                                    delta_vals[kNItemsX - 1], tmp_delta, -1
                                );
                                __syncthreads();

                                if (threadIdx.x == 0) {
                                    tmp_delta = last_delta_x[nrow];
                                    ddelta_vals_y[nrow] += -dx * WEIGHT_LT_H * h3 * A_val * delta_a_exps[i] * exp2f(tmp_delta * A_scaled) * A_decay;
                                }
                                
                                dA_val += -dx * WEIGHT_LT_H * h3 * delta_a_exps[i] * tmp_delta * exp2f(tmp_delta * A_scaled) * A_decay;
                                dA_decay_val += -dx * WEIGHT_LT_H * h3 * delta_a_exps[i] * exp2f(tmp_delta * A_scaled);

                                tmp_delta = -dx * WEIGHT_LT_H * h3 * A_val * delta_a_exps[i] * exp2f(tmp_delta * A_scaled) * A_decay;
                                Ktraits::BlockShuffleT(smem_shuffle).Offset(
                                    tmp_delta, tmp_delta, 1
                                );
                                __syncthreads();

                                if (threadIdx.x < last_thread_x) {
                                    ddelta_vals[nrow][kNItemsX - 1] += tmp_delta;
                                }
                            } else if (i < last_col) {
                                dA_val += -dx * WEIGHT_LT_H * h3 * delta_a_exps[i] * delta_vals[i-1] * delta_a_exps[i-1];
                                dA_decay_val += -dx * WEIGHT_LT_H * h3 * delta_a_exps[i] * delta_a_exps[i-1] / A_decay;
                                ddelta_vals[nrow][i - 1] += -dx * WEIGHT_LT_H * h3 * A_val * delta_a_exps[i] * delta_a_exps[i-1];
                            }

                            dB_vals[i] = dx * delta_vals[i] * float(u_vals[nrow][i]);
                            dC_vals[i] = dout_vals[nrow][i] * thread_data[nrow][i].y;
                            
                            next_row_data[i] = thread_reverse_data[i].y;
                            next_delta_A_exp_vals[i] = delta_a_exps[i];
                        }
                    }

                    // Block-exchange to make the atomicAdd's coalesced, otherwise they're much slower
                    Ktraits::BlockExchangeT(smem_exchange).BlockedToStriped(dB_vals, dB_vals);
                    __syncthreads();
                    Ktraits::BlockExchangeT(smem_exchange1).BlockedToStriped(dC_vals, dC_vals);
                    __syncthreads();
                    const int seqlen_remaining = params.ncols - chunk_x * kChunkSizeX - threadIdx.x;
                    weight_t *dB_cur = dB + state_idx * params.dB_dstate_stride + nrow_idx * params.dB_h_stride + chunk_x * kChunkSizeX + threadIdx.x;
                    weight_t *dC_cur = dC + state_idx * params.dC_dstate_stride + nrow_idx * params.dC_h_stride + chunk_x * kChunkSizeX + threadIdx.x;
                    #pragma unroll
                    for (int i = 0; i < kNItemsX; ++i) {
                        if (i * kNThreads < seqlen_remaining) {
                            { gpuAtomicAdd(dB_cur + i * kNThreads, dB_vals[i]); }
                            { gpuAtomicAdd(dC_cur + i * kNThreads, dC_vals[i]); }
                        }
                    }
                } // end for kNItemsY

                // Store last row
                store_h<Ktraits>(
                    x_v + state_idx * nChunksY * params.ncols + chunk_y * params.ncols + chunk_x * kChunkSizeX,
                    next_row_data, smem_store_h, params.ncols - chunk_x * kChunkSizeX
                );
                __syncthreads();

                if (x_sd != nullptr) {
                    // Load last running prefix
                    for (int nrow = 0; nrow < last_row + 1; ++nrow) {
                        if (chunk_x > 0 && threadIdx.x % 32 == 0 && nrow + chunk_y > 0) {
                            last_running_prefix[nrow] = x_sd[state_idx * nChunksX * params.nrows + (chunk_x - 1) * params.nrows + chunk_y * kChunkSizeY + nrow - 1];
                        } else {
                            last_running_prefix[nrow] = make_float2(1.f, 0.f);
                        }
                    }
                    // Forward scan
                    for (int nrow = 0; nrow < last_row; ++nrow) {
                        float delta_vals[kNItemsX], delta_u_vals[kNItemsX];
                        #pragma unroll
                        for (int i = 0; i < kNItemsX; ++i) {
                            float u_val = float(u_vals[nrow][i]);
                            delta_vals[i] = float(delta_vals_load[nrow][i]) + delta_bias;
                            if (params.delta_softplus) {
                                delta_vals[i] = delta_vals[i] <= 20.f ? log1pf(expf(delta_vals[i])) : delta_vals[i];
                            }
                            delta_u_vals[i] = delta_vals[i] * u_val;
                        }

                        #pragma unroll
                        for (int i = 0; i < kNItemsX; ++i) {
                            const float delta_a_exp = exp2f(delta_vals[i] * A_scaled) * A_decay;
                            thread_data[nrow][i] = make_float2(
                                delta_a_exp, 
                                B_vals[nrow][i] * delta_u_vals[i]
                            );

                            if constexpr (!Ktraits::kIsEvenLen) {  // So that the last state is correct
                                if (threadIdx.x * kNItemsX + i >= params.ncols - chunk_x * kChunkSizeX) {
                                    thread_data[nrow][i] = make_float2(1.f, 0.f);
                                }
                            }
                        }

                        // Initialize running total
                        scan_t running_prefix = last_running_prefix[nrow + 1];

                        SSMScanPrefixCallbackOp<weight_t> prefix_op(running_prefix);
                        Ktraits::BlockScanT(smem_scan).InclusiveScan(
                            thread_data[nrow], thread_data[nrow], SSMScanOp<weight_t>(), prefix_op
                        );
                        __syncthreads();
                    } // end for kNItemsY

                    // Backward scan
                    for (int nrow = last_row - 1; nrow >= 0; --nrow) {
                        const int nrow_idx = chunk_y * kChunkSizeY + nrow;

                        float delta_vals[kNItemsX];
                        #pragma unroll
                        for (int i = 0; i < kNItemsX; ++i) {
                            delta_vals[i] = float(delta_vals_load[nrow][i]) + delta_bias;
                            if constexpr (kDeltaSoftplus) {
                                delta_vals[i] = delta_vals[i] <= 20.f ? log1pf(expf(delta_vals[i])) : delta_vals[i];
                            }
                        }
                        
                        // Load C
                        weight_t C_vals[kNItemsX];
                        load_weight<Ktraits>(
                            Cvar + state_idx * params.C_dstate_stride + nrow_idx * params.C_h_stride + chunk_x * kChunkSizeX, 
                            C_vals, smem_load_weight1, (params.ncols - chunk_x * kChunkSizeX)
                        );
                        __syncthreads();

                        #pragma unroll
                        for (int i = 0; i < kNItemsX; ++i) {
                            float delta_a_exp = exp2f(delta_vals[i] * A_scaled) * A_decay;

                            if (i == 0) {
                                Ktraits::BlockShuffleT(smem_shuffle).Offset(
                                    delta_a_exp, thread_reverse_data[kNItemsX - 1].x, 1
                                );
                                __syncthreads();
                            } else {
                                thread_reverse_data[i - 1].x = delta_a_exp;
                            }
                        }

                        if (threadIdx.x == last_thread_x) {
                            thread_reverse_data[last_col - 1].x = exp2f(delta_x[nrow] * A_scaled) * A_decay;
                        }

                        #pragma unroll
                        for (int i = 0; i < kNItemsX; ++i) {
                            const float delta_a_exp = exp2f(delta_vals[i] * A_scaled) * A_decay;
                            thread_reverse_data[i].y = -dout_vals[nrow][i] * C_vals[i];

                            if constexpr (!Ktraits::kIsEvenLen) {  // So that the last state is correct
                                if (threadIdx.x * kNItemsX + i >= params.ncols - chunk_x * kChunkSizeX) {
                                    thread_reverse_data[i] = make_float2(1.f, 0.f);
                                }
                            }
                        }

                        // Initialize running total
                        scan_t running_postfix = chunk_x < nChunksX - 1 && threadIdx.x % 32 == 0 ? x_sd[state_idx * nChunksX * params.nrows + (chunk_x + 1) * params.nrows + nrow_idx] : make_float2(1.f, 0.f);
                        SSMScanPrefixCallbackOp<weight_t> postfix_op(running_postfix);
                        Ktraits::BlockReverseScanT(smem_reverse_scan).InclusiveReverseScan(
                            thread_reverse_data, thread_reverse_data, SSMScanOp<weight_t>(), postfix_op
                        );
                        if (threadIdx.x == 0) { 
                            x_sd[state_idx * nChunksX * params.nrows + chunk_x * params.nrows + nrow_idx] =  postfix_op.running_prefix;
                        }
                        __syncthreads();

                        // Calculate du, ddelta, dB, dC
                        weight_t dB_vals[kNItemsX], dC_vals[kNItemsX];
                        {
                            weight_t tmp = 0;
                            Ktraits::BlockShuffleT(smem_shuffle).Offset(
                                thread_data[nrow][kNItemsX - 1].y, tmp, -1
                            );
                            if (threadIdx.x == 0 && chunk_x > 0) {
                                tmp = x[state_idx * nChunksX * params.nrows + (chunk_x - 1) * params.nrows + nrow_idx].y;
                            }
                            __syncthreads();
                            float delta_a_exps[kNItemsX];
                            #pragma unroll
                            for (int i = kNItemsX - 1; i >= 0; --i) {
                                delta_a_exps[i] = exp2f(delta_vals[i] * A_scaled) * A_decay;
                            }
                            #pragma unroll
                            for (int i = kNItemsX - 1; i >= 0; --i) {
                                const float dx = thread_reverse_data[i].y;
                                const float ddelta_u = dx * B_vals[nrow][i];
                                du_vals[nrow][i] += ddelta_u * delta_vals[i] + dout_vals[nrow][i] * 0.25 * delta_vals[i] * B_vals[nrow][i] * C_vals[i];
                                const float a = thread_data[nrow][i].y - (delta_vals[i] * float(u_vals[nrow][i]) * B_vals[nrow][i]);
                                ddelta_vals[nrow][i] += ddelta_u * float(u_vals[nrow][i]) + dx * A_val * a + dout_vals[nrow][i] * 0.25 * float(u_vals[nrow][i]) * B_vals[nrow][i] * C_vals[i];
                                dA_val += dx * delta_vals[i] * a;
                                dB_vals[i] = dx * delta_vals[i] * float(u_vals[nrow][i]) + dout_vals[nrow][i] * 0.25 * float(u_vals[nrow][i]) * delta_vals[i] * C_vals[i];
                                dC_vals[i] = dout_vals[nrow][i] * (0.25 * B_vals[nrow][i] * float(u_vals[nrow][i]) * delta_vals[i]  - thread_data[nrow][i].y);
                            }
                        }

                        // Block-exchange to make the atomicAdd's coalesced, otherwise they're much slower
                        Ktraits::BlockExchangeT(smem_exchange).BlockedToStriped(dB_vals, dB_vals);
                        __syncthreads();
                        Ktraits::BlockExchangeT(smem_exchange1).BlockedToStriped(dC_vals, dC_vals);
                        __syncthreads();
                        const int seqlen_remaining = params.ncols - chunk_x * kChunkSizeX - threadIdx.x;
                        weight_t *dB_cur = dB + state_idx * params.dB_dstate_stride + nrow_idx * params.dB_h_stride + chunk_x * kChunkSizeX + threadIdx.x;
                        weight_t *dC_cur = dC + state_idx * params.dC_dstate_stride + nrow_idx * params.dC_h_stride + chunk_x * kChunkSizeX + threadIdx.x;
                        #pragma unroll
                        for (int i = 0; i < kNItemsX; ++i) {
                            if (i * kNThreads < seqlen_remaining) {
                                { gpuAtomicAdd(dB_cur + i * kNThreads, dB_vals[i]); }
                                { gpuAtomicAdd(dC_cur + i * kNThreads, dC_vals[i]); }
                            }
                        }
                    } // end for kNItemsY
                } 

                dA_val = Ktraits::BlockReduceFloatT(smem_reduce_float).Sum(dA_val);
                if (threadIdx.x == 0) {
                    smem_da[state_idx] = chunk_x == nChunksX - 1 && chunk_y == nChunksY - 1 ? dA_val : dA_val + smem_da[state_idx];
                }
                __syncthreads();

            } // end for dstate
            
            if constexpr (kDeltaSoftplus) {
                float delta_val, delta_val_neg_exp;
                for (int nrow = 0; nrow < last_row; ++nrow) { 
                    #pragma unroll
                    for (int i = 0; i < kNItemsX; ++i) {
                        delta_val = float(delta_vals_load[nrow][i]) + delta_bias;
                        delta_val_neg_exp = expf(-delta_val);
                        ddelta_vals[nrow][i] = delta_val <= 20.f ? ddelta_vals[nrow][i] / (1.f + delta_val_neg_exp) : ddelta_vals[nrow][i];
                    }

                    delta_val = float(last_delta_x_[nrow]) + delta_bias;
                    delta_val_neg_exp = expf(-delta_val);
                    ddelta_vals_y[nrow] = delta_val <= 20.f ? ddelta_vals_y[nrow] / (1.f + delta_val_neg_exp) : ddelta_vals_y[nrow];
                }

                #pragma unroll
                for (int i = 0; i < kNItemsX; ++i) {
                    delta_val = float(last_delta_vals_[i]) + delta_bias;
                    delta_val_neg_exp = expf(-delta_val);
                    ddelta_vals_x[i] = delta_val <= 20.f ? ddelta_vals_x[i] / (1.f + delta_val_neg_exp) : ddelta_vals_x[i];
                }
            }

            // ddelta_bias
            for (int nrow = 0; nrow < last_row; ++nrow) { 
                #pragma unroll
                for (int i = 0; i < kNItemsX; ++i) { 
                    ddelta_bias_val += ddelta_vals[nrow][i]; 
                }
                ddelta_bias_val += ddelta_vals_y[nrow];
            }
            #pragma unroll
            for (int i = 0; i < kNItemsX; ++i) {
                ddelta_bias_val += ddelta_vals_x[i];
            }

            input_t *ddelta = reinterpret_cast<input_t *>(params.ddelta_ptr) + batch_id * params.ddelta_batch_stride
                    + dim_id * params.ddelta_d_stride + chunk_x * kChunkSizeX;
            for (int nrow = 0; nrow < last_row; ++nrow) {
                const int nrow_idx = chunk_y * kChunkSizeY + nrow;
                
                input_t *du = reinterpret_cast<input_t *>(params.du_ptr) + batch_id * params.du_batch_stride
                    + dim_id * params.du_d_stride + nrow_idx * params.du_h_stride +  chunk_x * kChunkSizeX;
                store_output<Ktraits>(du, du_vals[nrow], smem_store, params.ncols - chunk_x * kChunkSizeX);
                __syncthreads();
                
                if (chunk_x < nChunksX - 1 && threadIdx.x == kNThreads - 1) {
                    ddelta_vals[nrow][kNItemsX - 1] += ddelta[nrow_idx * params.ddelta_h_stride + kChunkSizeX - 1]; 
                }
                if (nrow == last_row - 1 && chunk_y < nChunksY - 1) {
                    float tmp_ddelta_vals[kNItemsX];
                    input_t tmp_ddelta_loads[kNItemsX];
                    load_input<Ktraits>(
                        ddelta + nrow_idx * params.ddelta_h_stride, 
                        tmp_ddelta_loads, smem_load, params.ncols - chunk_x * kChunkSizeX
                    );
                    Converter<typename Ktraits::input_t, kNItemsX>::to_float(tmp_ddelta_loads, tmp_ddelta_vals);
                    __syncthreads();

                    if (chunk_x < nChunksX - 1 && threadIdx.x == kNThreads - 1) {
                        #pragma unroll
                        for (int i = 0; i < kNItemsX - 1; ++i) {
                            ddelta_vals[nrow][i] += tmp_ddelta_vals[i];    
                        }
                    } else {
                        #pragma unroll
                        for (int i = 0; i < kNItemsX; ++i) {
                            ddelta_vals[nrow][i] += tmp_ddelta_vals[i];    
                        }
                    }
                }
                store_output<Ktraits>(
                    ddelta + nrow_idx * params.ddelta_h_stride, 
                    ddelta_vals[nrow], smem_store, params.ncols - chunk_x * kChunkSizeX
                );
                
                if (chunk_x > 0 && threadIdx.x == 0) {
                    input_t *ddelta1 = reinterpret_cast<input_t *>(params.ddelta_ptr) + batch_id * params.ddelta_batch_stride
                    + dim_id * params.ddelta_d_stride + chunk_x * kChunkSizeX + nrow_idx * params.ddelta_h_stride - 1;
                    if (chunk_y < nChunksY - 1 && nrow == last_row - 1) {
                        ddelta1[0] += ddelta_vals_y[nrow];
                    } else {
                        ddelta1[0] = ddelta_vals_y[nrow];
                    }
                }
                __syncthreads();
            }

            if (chunk_y > 0) {
                store_output<Ktraits>(
                    ddelta + (chunk_y * kChunkSizeY - 1) * params.ddelta_h_stride, 
                    ddelta_vals_x, smem_store, params.ncols - chunk_x * kChunkSizeX
                );
                __syncthreads();
            }
        } // end for nChunkX
    } // end for nChunkY

    if (params.dD_ptr != nullptr) {
        dD_val = Ktraits::BlockReduceFloatT(smem_reduce_float).Sum(dD_val);
        __syncthreads();
        if (threadIdx.x == 0) { gpuAtomicAdd(dD, dD_val); }
    }
    if (params.ddelta_bias_ptr != nullptr) {
        ddelta_bias_val = Ktraits::BlockReduceFloatT(smem_reduce_float).Sum(ddelta_bias_val);
        __syncthreads();
        if (threadIdx.x == 0) { gpuAtomicAdd(ddelta_bias, ddelta_bias_val); }
    }
    __syncthreads();
    // !Note: It requires that dstate be smaller than blockDim.x
    for (int state_idx = threadIdx.x; state_idx < params.dstate; state_idx += blockDim.x) {
        gpuAtomicAdd(&(dA[state_idx * params.dA_dstate_stride]), smem_da[state_idx]);
    }

    if (params.dA_decay_ptr != nullptr) {
        gpuAtomicAdd(dA_decay, dA_decay_val);
    }
}

template<int kNThreads, int kNItemsX, int kNItemsY, typename input_t, typename weight_t, typename output_t>
void selective_scan_bwd_launch(SSMParamsBwd &params, cudaStream_t stream) {
    BOOL_SWITCH(params.ncols % (kNThreads * kNItemsX) == 0, kIsEvenLen, [&] {
        BOOL_SWITCH(params.delta_softplus, kDeltaSoftplus, [&] {
            using Ktraits = Selective_Scan_bwd_kernel_traits<kNThreads, kNItemsX, kNItemsY, kIsEvenLen, kDeltaSoftplus, input_t, weight_t, output_t>;
            constexpr int kSmemSize = Ktraits::kSmemSize + Ktraits::MaxDState * sizeof(typename Ktraits::weight_t);
            // printf("smem_size = %d\n", kSmemSize);
            dim3 grid(params.batch, params.dim);
            auto kernel = &selective_scan_bwd_kernel<Ktraits>;
            if (kSmemSize >= 48 * 1024) {
                C10_CUDA_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, kSmemSize));
            }
            kernel<<<grid, Ktraits::kNThreads, kSmemSize, stream>>>(params);
            C10_CUDA_KERNEL_LAUNCH_CHECK();
        });
    });
}

template<typename input_t, typename weight_t, typename output_t>
void selective_scan_bwd_cuda(SSMParamsBwd &params, cudaStream_t stream) {
    constexpr int kMin = sizeof(input_t) == 2 ? 2 : 1;
    if (params.ncols <= 128) {
        selective_scan_bwd_launch<
            32, THREAD_ITEMS_X * kMin, THREAD_ITEMS_Y, input_t, weight_t, output_t
        >(params, stream);
    } else if (params.ncols <= 256) {
        selective_scan_bwd_launch<
            64, THREAD_ITEMS_X * kMin, THREAD_ITEMS_Y, input_t, weight_t, output_t
        >(params, stream);
    } else {
        selective_scan_bwd_launch<
            128, THREAD_ITEMS_X * kMin, THREAD_ITEMS_Y, input_t, weight_t, output_t
        >(params, stream);
    }
}

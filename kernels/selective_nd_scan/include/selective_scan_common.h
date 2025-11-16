/******************************************************************************
 * Copyright (c) 2023, Tri Dao.
 ******************************************************************************/

#pragma once

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>


// Used to determine the size of the sem
#define MAX_DSTATE 16

#define THREAD_ITEMS_X 4

#define THREAD_ITEMS_Y 16
#define WEIGHT_DELTA_A_EXP 1.0
#define MAX_THREADS_X 128
#define WEIGHT_LT_H 0.5
#define SD_SWITCH_ON 1


inline __device__ float2 operator+(const float2 & a, const float2 & b){
    return {a.x + b.x, a.y + b.y};
}

inline __device__ float3 operator+(const float3 &a, const float3 &b) {
  return {a.x + b.x, a.y + b.y, a.z + b.z};
}

inline __device__ float4 operator+(const float4 & a, const float4 & b){
    return {a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w};
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template<int BYTES> struct BytesToType {};

template<> struct BytesToType<16> {
    using Type = uint4;
    static_assert(sizeof(Type) == 16);
};

template<> struct BytesToType<8> {
    using Type = uint64_t;
    static_assert(sizeof(Type) == 8);
};

template<> struct BytesToType<4> {
    using Type = uint32_t;
    static_assert(sizeof(Type) == 4);
};

template<> struct BytesToType<2> {
    using Type = uint16_t;
    static_assert(sizeof(Type) == 2);
};

template<> struct BytesToType<1> {
    using Type = uint8_t;
    static_assert(sizeof(Type) == 1);
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template<typename scalar_t, int N>
struct Converter{
    static inline __device__ void to_float(const scalar_t (&src)[N], float (&dst)[N]) {
        #pragma unroll
        for (int i = 0; i < N; ++i) { dst[i] = src[i]; }
    }
};

template<int N>
struct Converter<at::Half, N>{
    static inline __device__ void to_float(const at::Half (&src)[N], float (&dst)[N]) {
        static_assert(N % 2 == 0);
        auto &src2 = reinterpret_cast<const half2 (&)[N / 2]>(src);
        auto &dst2 = reinterpret_cast<float2 (&)[N / 2]>(dst);
        #pragma unroll
        for (int i = 0; i < N / 2; ++i) { dst2[i] = __half22float2(src2[i]); }
    }
};

#if __CUDA_ARCH__ >= 800
template<int N>
struct Converter<at::BFloat16, N>{
    static inline __device__ void to_float(const at::BFloat16 (&src)[N], float (&dst)[N]) {
        static_assert(N % 2 == 0);
        auto &src2 = reinterpret_cast<const nv_bfloat162 (&)[N / 2]>(src);
        auto &dst2 = reinterpret_cast<float2 (&)[N / 2]>(dst);
        #pragma unroll
        for (int i = 0; i < N / 2; ++i) { dst2[i] = __bfloat1622float2(src2[i]); }
    }
};
#endif

////////////////////////////////////////////////////////////////////////////////////////////////////
template<typename scalar_t> struct SSMScanOp;

template<>
struct SSMScanOp<float> {
    __device__ __forceinline__ float2 operator()(const float2 &ab0, const float2 &ab1) const {
        return make_float2(ab1.x * ab0.x, ab1.x * ab0.y + ab1.y);
    }
};

// A stateful callback functor that maintains a running prefix to be applied
// during consecutive scan operations.
template <typename scalar_t> struct SSMScanPrefixCallbackOp {
    using scan_t = std::conditional_t<std::is_same_v<scalar_t, float>, float2, float4>;
    scan_t running_prefix;
    // Constructor
    __device__ SSMScanPrefixCallbackOp(scan_t running_prefix_) : running_prefix(running_prefix_) {}
    // Callback operator to be entered by the first warp of threads in the block.
    // Thread-0 is responsible for returning a value for seeding the block-wide scan.
    __device__ scan_t operator()(scan_t block_aggregate) {
        scan_t old_prefix = running_prefix;
        running_prefix = SSMScanOp<scalar_t>()(running_prefix, block_aggregate);
        return old_prefix;
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template<typename Ktraits>
inline __device__ void load_input(typename Ktraits::input_t *u,
                                  typename Ktraits::input_t (&u_vals)[Ktraits::kNItemsX],
                                  typename Ktraits::BlockLoadT::TempStorage &smem_load,
                                  int seqlen) {
    if constexpr (Ktraits::kIsEvenLen) {
        auto& smem_load_vec = reinterpret_cast<typename Ktraits::BlockLoadVecT::TempStorage&>(smem_load);
        using vec_t = typename Ktraits::vec_t;
        Ktraits::BlockLoadVecT(smem_load_vec).Load(
            reinterpret_cast<vec_t*>(u),
            reinterpret_cast<vec_t(&)[Ktraits::kNLoads]>(u_vals)
       );
    } else {
        Ktraits::BlockLoadT(smem_load).Load(u, u_vals, seqlen, 0.f);
    }
}

template<typename Ktraits>
inline __device__ void load_weight(typename Ktraits::input_t *Bvar,
                                   typename Ktraits::weight_t (&B_vals)[Ktraits::kNItemsX],
                                   typename Ktraits::BlockLoadWeightT::TempStorage &smem_load_weight,
                                   int seqlen) {
    constexpr int kNItemsX = Ktraits::kNItemsX;
    typename Ktraits::input_t B_vals_load[kNItemsX];
    if constexpr (Ktraits::kIsEvenLen) {
        auto& smem_load_weight_vec = reinterpret_cast<typename Ktraits::BlockLoadWeightVecT::TempStorage&>(smem_load_weight);
        using vec_t = typename Ktraits::vec_t;
        Ktraits::BlockLoadWeightVecT(smem_load_weight_vec).Load(
            reinterpret_cast<vec_t*>(Bvar),
            reinterpret_cast<vec_t(&)[Ktraits::kNLoads]>(B_vals_load)
      );
    } else {
        Ktraits::BlockLoadWeightT(smem_load_weight).Load(Bvar, B_vals_load, seqlen, 0.f);
    }
    // #pragma unroll
    // for (int i = 0; i < kNItemsX; ++i) { B_vals[i] = B_vals_load[i]; }
    Converter<typename Ktraits::input_t, kNItemsX>::to_float(B_vals_load, B_vals);
}

template<typename Ktraits>
inline __device__ void store_output(typename Ktraits::input_t *out,
                                    const float (&out_vals)[Ktraits::kNItemsX],
                                    typename Ktraits::BlockStoreT::TempStorage &smem_store,
                                    int seqlen) {
    typename Ktraits::input_t write_vals[Ktraits::kNItemsX];
    #pragma unroll
    for (int i = 0; i < Ktraits::kNItemsX; ++i) { write_vals[i] = out_vals[i]; }
    if constexpr (Ktraits::kIsEvenLen) {
        auto& smem_store_vec = reinterpret_cast<typename Ktraits::BlockStoreVecT::TempStorage&>(smem_store);
        using vec_t = typename Ktraits::vec_t;
        Ktraits::BlockStoreVecT(smem_store_vec).Store(
            reinterpret_cast<vec_t*>(out),
            reinterpret_cast<vec_t(&)[Ktraits::kNLoads]>(write_vals)
       );
    } else {
        Ktraits::BlockStoreT(smem_store).Store(out, write_vals, seqlen);
    }
}

template<typename Ktraits>
inline __device__ void store_output1(typename Ktraits::output_t *out,
                                    const float (&out_vals)[Ktraits::kNItemsX],
                                    typename Ktraits::BlockStoreOutputT::TempStorage &smem_store,
                                    int seqlen) {
    typename Ktraits::output_t write_vals[Ktraits::kNItemsX];
    #pragma unroll
    for (int i = 0; i < Ktraits::kNItemsX; ++i) { write_vals[i] = out_vals[i]; }
    if constexpr (Ktraits::kIsEvenLen) {
        auto& smem_store_vec = reinterpret_cast<typename Ktraits::BlockStoreOutputVecT::TempStorage&>(smem_store);
        using vec_t = typename Ktraits::vec_t;
        Ktraits::BlockStoreOutputVecT(smem_store_vec).Store(
            reinterpret_cast<vec_t*>(out),
            reinterpret_cast<vec_t(&)[Ktraits::kNLoadsOutput]>(write_vals)
       );
    } else {
        Ktraits::BlockStoreOutputT(smem_store).Store(out, write_vals, seqlen);
    }
}

template<typename Ktraits>
inline __device__ void load_output(typename Ktraits::output_t *u,
                                  typename Ktraits::output_t (&u_vals)[Ktraits::kNItemsX],
                                  typename Ktraits::BlockLoadOutputT::TempStorage &smem_load,
                                  int seqlen) {
    if constexpr (Ktraits::kIsEvenLen) {
        auto& smem_load_vec = reinterpret_cast<typename Ktraits::BlockLoadOutputVecT::TempStorage&>(smem_load);
        using vec_t = typename Ktraits::vec_t;
        Ktraits::BlockLoadOutputVecT(smem_load_vec).Load(
            reinterpret_cast<vec_t*>(u),
            reinterpret_cast<vec_t(&)[Ktraits::kNLoadsOutput]>(u_vals)
       );
    } else {
        Ktraits::BlockLoadOutputT(smem_load).Load(u, u_vals, seqlen, 0.f);
    }
}

template<typename Ktraits>
inline __device__ void store_h(typename Ktraits::weight_t *h,
                                typename Ktraits::weight_t (&h_vals)[Ktraits::kNItemsX],
                                typename Ktraits::BlockStoreHT::TempStorage &smem_store,
                                int seqlen) {
    typename Ktraits::weight_t write_vals[Ktraits::kNItemsX];
    #pragma unroll
    for (int i = 0; i < Ktraits::kNItemsX; ++i) { write_vals[i] = h_vals[i]; }
    if constexpr (Ktraits::kIsEvenLen) {
        auto& smem_store_vec = reinterpret_cast<typename Ktraits::BlockStoreHVecT::TempStorage&>(smem_store);
        using vec_t = typename Ktraits::vec_t;
        Ktraits::BlockStoreHVecT(smem_store_vec).Store(
            reinterpret_cast<vec_t*>(h),
            reinterpret_cast<vec_t(&)[Ktraits::kNLoadsH]>(write_vals)
       );
    } else {
        Ktraits::BlockStoreHT(smem_store).Store(h, write_vals, seqlen);
    }
}

template<typename Ktraits>
inline __device__ void load_h(typename Ktraits::weight_t *h,
                            typename Ktraits::weight_t (&h_vals)[Ktraits::kNItemsX],
                            typename Ktraits::BlockLoadHT::TempStorage &smem_load,
                            int seqlen) {
    if constexpr (Ktraits::kIsEvenLen) {
        auto& smem_load_vec = reinterpret_cast<typename Ktraits::BlockLoadHVecT::TempStorage&>(smem_load);
        using vec_t = typename Ktraits::vec_t;
        Ktraits::BlockLoadHVecT(smem_load_vec).Load(
            reinterpret_cast<vec_t*>(h),
            reinterpret_cast<vec_t(&)[Ktraits::kNLoadsH]>(h_vals)
       );
    } else {
        Ktraits::BlockLoadHT(smem_load).Load(h, h_vals, seqlen, 0.f);
    }
}


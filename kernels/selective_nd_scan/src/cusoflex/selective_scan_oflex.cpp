/******************************************************************************
 * Copyright (c) 2023, Tri Dao.
 ******************************************************************************/

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>
#include <vector>

#include "selective_scan_oflex.h"
#include "selective_scan_common.h"


#define CHECK_SHAPE(x, ...) TORCH_CHECK(x.sizes() == torch::IntArrayRef({__VA_ARGS__}), #x " must have shape (" #__VA_ARGS__ ")")
using weight_t = float;

#define DISPATCH_ITYPE_FLOAT_AND_HALF_AND_BF16(ITYPE, NAME, ...)                    \
    if (ITYPE == at::ScalarType::Half) {                                            \
        using input_t = at::Half;                                                   \
        __VA_ARGS__();                                                              \
    } else if (ITYPE == at::ScalarType::BFloat16) {                                 \
        using input_t = at::BFloat16;                                               \
        __VA_ARGS__();                                                              \
    } else if (ITYPE == at::ScalarType::Float)  {                                   \
        using input_t = float;                                                      \
        __VA_ARGS__();                                                              \
    } else {                                                                        \
        AT_ERROR(#NAME, " not implemented for input type '", toString(ITYPE), "'"); \
    }

template<typename input_t, typename weight_t, typename output_t>
void selective_scan_fwd_cuda(SSMParamsBase &params, cudaStream_t stream);

template <typename input_t, typename weight_t, typename output_t>
void selective_scan_bwd_cuda(SSMParamsBwd &params, cudaStream_t stream);

void set_ssm_params_fwd(
    SSMParamsBase &params,
    const size_t nChunksY,
    const size_t nChunksX,
    const at::Tensor &u,
    const at::Tensor &delta,
    const at::Tensor &A,
    const at::Tensor &B,
    const at::Tensor &C,
    const at::Tensor &out,
    void* D_ptr,
    void* delta_bias_ptr,
    void* A_decay_ptr,
    void* x_ptr,
    void* x_sd_ptr,
    const at::Tensor &x_v,
    bool delta_softplus
) {

    // Reset the parameters
    memset(&params, 0, sizeof(params));

    params.batch = u.size(0);
    params.dim = u.size(1);
    params.nrows = u.size(2);
    params.ncols = u.size(3);
    params.dstate = B.size(2);
    params.n_groups = B.size(1);
    params.nChunksY = nChunksY;
    params.nChunksX = nChunksX;
    params.dim_ngroups_ratio = params.dim / params.n_groups;
    params.dim_deltagroups_ratio = params.dim / delta.size(1);

    params.delta_softplus = delta_softplus;

    // Set the pointers and strides.
    params.u_ptr = u.data_ptr();
    params.delta_ptr = delta.data_ptr();
    params.A_ptr = A.data_ptr();
    params.A_decay_ptr = A_decay_ptr;
    params.B_ptr = B.data_ptr();
    params.C_ptr = C.data_ptr();
    params.D_ptr = D_ptr;
    params.delta_bias_ptr = delta_bias_ptr;
    params.out_ptr = out.data_ptr();
    params.x_ptr = x_ptr;
    params.x_sd_ptr = x_sd_ptr;
    params.x_v_ptr = x_v.data_ptr();

    // All stride are in elements, not bytes.
    params.A_d_stride = A.stride(0);
    params.A_dstate_stride = A.stride(1);
    params.B_batch_stride = B.stride(0);
    params.B_group_stride = B.stride(1);
    params.B_dstate_stride = B.stride(2);
    params.B_h_stride = B.stride(3);
    params.C_batch_stride = C.stride(0);
    params.C_group_stride = C.stride(1);
    params.C_dstate_stride = C.stride(2);
    params.C_h_stride = C.stride(3);
    params.u_batch_stride = u.stride(0);
    params.u_d_stride = u.stride(1);
    params.u_h_stride = u.stride(2);
    params.delta_batch_stride = delta.stride(0);
    params.delta_d_stride = delta.stride(1);
    params.delta_h_stride = delta.stride(2);

    params.out_batch_stride = out.stride(0);
    params.out_d_stride = out.stride(1);
    params.out_h_stride = out.stride(2);

    params.x_v_batch_stride = x_v.stride(0);
    params.x_v_d_stride = x_v.stride(1);
    params.x_v_dstate_stride = x_v.stride(2);
    params.x_v_chunk_stride = x_v.stride(3);
}

void set_ssm_params_bwd(
    SSMParamsBwd &params,
    const size_t nChunksY,
    const size_t nChunksX,
    const at::Tensor &u,
    const at::Tensor &delta,
    const at::Tensor &A,
    const at::Tensor &B,
    const at::Tensor &C,
    const at::Tensor &out,
    void* D_ptr,
    void* delta_bias_ptr,
    void* A_decay_ptr,
    void* x_ptr,
    void* x_sd_ptr,
    const at::Tensor &x_v,
    const at::Tensor &dout,
    const at::Tensor &du,
    const at::Tensor &ddelta,
    const at::Tensor &dA,
    const at::Tensor &dB,
    const at::Tensor &dC,
    void* dD_ptr,
    void* ddelta_bias_ptr,
    void* dA_decay_ptr,
    bool delta_softplus
) {
    set_ssm_params_fwd(
        params, nChunksY, nChunksX,
        u, delta, A, B, C, dout,
        D_ptr, delta_bias_ptr, A_decay_ptr, x_ptr, x_sd_ptr, x_v, delta_softplus
    );

    // Set the pointers and strides.
    params.dout_ptr = dout.data_ptr();
    params.du_ptr = du.data_ptr();
    params.dA_ptr = dA.data_ptr();
    params.dA_decay_ptr = dA_decay_ptr;
    params.dB_ptr = dB.data_ptr();
    params.dC_ptr = dC.data_ptr();
    params.dD_ptr = dD_ptr;
    params.ddelta_ptr = ddelta.data_ptr();
    params.ddelta_bias_ptr = ddelta_bias_ptr;
    // All stride are in elements, not bytes.
    params.dout_batch_stride = dout.stride(0);
    params.dout_d_stride = dout.stride(1);
    params.dout_h_stride = dout.stride(2);
    params.dA_d_stride = dA.stride(0);
    params.dA_dstate_stride = dA.stride(1);
    params.dB_batch_stride = dB.stride(0);
    params.dB_group_stride = dB.stride(1);
    params.dB_dstate_stride = dB.stride(2);
    params.dB_h_stride = dB.stride(3);
    params.dC_batch_stride = dC.stride(0);
    params.dC_group_stride = dC.stride(1);
    params.dC_dstate_stride = dC.stride(2);
    params.dC_h_stride = dC.stride(3);
    params.du_batch_stride = du.stride(0);
    params.du_d_stride = du.stride(1);
    params.du_h_stride = du.stride(2);
    params.ddelta_batch_stride = ddelta.stride(0);
    params.ddelta_d_stride = ddelta.stride(1);
    params.ddelta_h_stride = ddelta.stride(2);
}

std::vector<at::Tensor>
selective_scan_fwd(const at::Tensor &u, const at::Tensor &delta,
                  const at::Tensor &A, const at::Tensor &B, const at::Tensor &C,
                  const c10::optional<at::Tensor> &D_,
                  const c10::optional<at::Tensor> &delta_bias_,
                  bool delta_softplus,
                  bool out_float,
                  const c10::optional<at::Tensor> &A_decay_
                  ) {
    auto input_type = u.scalar_type();
    auto weight_type = A.scalar_type();
    TORCH_CHECK(input_type == at::ScalarType::Float || input_type == at::ScalarType::Half || input_type == at::ScalarType::BFloat16);
    TORCH_CHECK(weight_type == at::ScalarType::Float);

    TORCH_CHECK(delta.scalar_type() == input_type);
    TORCH_CHECK(B.scalar_type() == input_type);
    TORCH_CHECK(C.scalar_type() == input_type);

    TORCH_CHECK(u.is_cuda());
    TORCH_CHECK(delta.is_cuda());
    TORCH_CHECK(A.is_cuda());
    TORCH_CHECK(B.is_cuda());
    TORCH_CHECK(C.is_cuda());

    TORCH_CHECK(u.stride(-1) == 1 || u.size(-1) == 1);
    TORCH_CHECK(delta.stride(-1) == 1 || delta.size(-1) == 1);

    const auto sizes = u.sizes();
    const int batch_size = sizes[0];
    const int dim = sizes[1];
    const int nrows = sizes[2];
    const int ncols = sizes[3];
    const int dstate = A.size(1);
    const int n_groups = B.size(1);
    const int delta_dim = delta.size(1);

    TORCH_CHECK(dim % n_groups == 0, "dims should be dividable by n_groups");
    TORCH_CHECK(dim % delta_dim == 0, "dims should be dividable by delta_dim");
    TORCH_CHECK(dstate <= MAX_DSTATE, "selective_scan only supports state dimension <= 256");

    CHECK_SHAPE(u, batch_size, dim, nrows, ncols);
    CHECK_SHAPE(delta, batch_size, delta_dim, nrows, ncols);
    CHECK_SHAPE(A, dim, dstate);
    if (A_decay_.has_value()) {
        auto A_decay = A_decay_.value();
        CHECK_SHAPE(A_decay, 1);
    }
    CHECK_SHAPE(B, batch_size, n_groups, dstate, nrows, ncols);
    TORCH_CHECK(B.stride(-1) == 1 || B.size(-1) == 1);
    CHECK_SHAPE(C, batch_size, n_groups, dstate, nrows, ncols);
    TORCH_CHECK(C.stride(-1) == 1 || C.size(-1) == 1);

    if (D_.has_value()) {
        auto D = D_.value();
        TORCH_CHECK(D.scalar_type() == at::ScalarType::Float);
        TORCH_CHECK(D.is_cuda());
        TORCH_CHECK(D.stride(-1) == 1 || D.size(-1) == 1);
        CHECK_SHAPE(D, dim);
    }

    if (delta_bias_.has_value()) {
        auto delta_bias = delta_bias_.value();
        TORCH_CHECK(delta_bias.scalar_type() == at::ScalarType::Float);
        TORCH_CHECK(delta_bias.is_cuda());
        TORCH_CHECK(delta_bias.stride(-1) == 1 || delta_bias.size(-1) == 1);
        CHECK_SHAPE(delta_bias, delta_dim);
    }

    const int kMin = c10::elementSize(u.scalar_type()) == 2 ? 2 : 1;
    const int nChunksY = (nrows + THREAD_ITEMS_Y - 1) / THREAD_ITEMS_Y;
    const int nChunksX = (ncols + MAX_THREADS_X * THREAD_ITEMS_X * kMin - 1) / (MAX_THREADS_X * THREAD_ITEMS_X * kMin); // max is 128 * THREAD_WIDTH in fwd_kernel
    at::Tensor out = torch::empty({batch_size, dim, nrows, ncols}, u.options().dtype(out_float? (at::ScalarType::Float): input_type));
    at::Tensor x = torch::empty({batch_size, dim, dstate, nChunksX, nrows * 2}, u.options().dtype(weight_type));
    at::Tensor x_v = torch::empty({batch_size, dim, dstate, nChunksY, ncols}, u.options().dtype(weight_type));

    #if SD_SWITCH_ON
    at::Tensor x_sd = torch::empty({batch_size, dim, dstate, nChunksX, nrows * 2}, u.options().dtype(weight_type));
    #else
    at::Tensor x_sd = at::Tensor();
    #endif

    SSMParamsBase params;
    set_ssm_params_fwd(
        params, nChunksY, nChunksX,
        u, delta, A, B, C, out,
        D_.has_value() ? D_.value().data_ptr() : nullptr,
        delta_bias_.has_value() ? delta_bias_.value().data_ptr() : nullptr,
        A_decay_.has_value() ? A_decay_.value().data_ptr() : nullptr,
        x.data_ptr(), x_sd.defined() ? x_sd.data_ptr(): nullptr, x_v,
        delta_softplus
    );

    // Otherwise the kernel will be launched from cuda:0 device
    // Cast to char to avoid compiler warning about narrowing
    at::cuda::CUDAGuard device_guard{(char)u.get_device()};
    auto stream = at::cuda::getCurrentCUDAStream().stream();
    DISPATCH_ITYPE_FLOAT_AND_HALF_AND_BF16(u.scalar_type(), "selective_scan_fwd", [&] {
        if (!out_float) {
            selective_scan_fwd_cuda<input_t, weight_t, input_t>(params, stream);
        } else {
            selective_scan_fwd_cuda<input_t, weight_t, float>(params, stream);
        }
    });
    std::vector<at::Tensor> result = {out, x, x_v, x_sd};
    return result;
}

std::vector<at::Tensor>
selective_scan_bwd(const at::Tensor &u, const at::Tensor &delta,
                  const at::Tensor &A, const at::Tensor &B, const at::Tensor &C,
                  const c10::optional<at::Tensor> &D_,
                  const c10::optional<at::Tensor> &delta_bias_,
                  const at::Tensor &dout,
                  const c10::optional<at::Tensor> &x_,
                  const c10::optional<at::Tensor> &x_v_,
                  bool delta_softplus,
                  const c10::optional<at::Tensor> &A_decay_,
                  const c10::optional<at::Tensor> &x_sd_
                  ) {
    auto input_type = u.scalar_type();
    auto weight_type = A.scalar_type();
    auto output_type = dout.scalar_type();
    TORCH_CHECK(input_type == at::ScalarType::Float || input_type == at::ScalarType::Half || input_type == at::ScalarType::BFloat16);
    TORCH_CHECK(weight_type == at::ScalarType::Float);
    TORCH_CHECK(output_type == input_type || output_type == at::ScalarType::Float);

    TORCH_CHECK(delta.scalar_type() == input_type);
    TORCH_CHECK(B.scalar_type() == input_type);
    TORCH_CHECK(C.scalar_type() == input_type);

    TORCH_CHECK(u.is_cuda());
    TORCH_CHECK(delta.is_cuda());
    TORCH_CHECK(A.is_cuda());
    TORCH_CHECK(B.is_cuda());
    TORCH_CHECK(C.is_cuda());
    TORCH_CHECK(dout.is_cuda());

    TORCH_CHECK(u.stride(-1) == 1 || u.size(-1) == 1);
    TORCH_CHECK(delta.stride(-1) == 1 || delta.size(-1) == 1);
    TORCH_CHECK(dout.stride(-1) == 1 || dout.size(-1) == 1);

    const auto sizes = u.sizes();
    const int batch_size = sizes[0];
    const int dim = sizes[1];
    const int nrows = sizes[2];
    const int ncols = sizes[3];
    const int dstate = A.size(1);
    const int n_groups = B.size(1);
    const int delta_dim = delta.size(1);
    const int delta_dim_repeat = dim / delta_dim;

    TORCH_CHECK(dim % n_groups == 0, "dims should be dividable by n_groups");
    TORCH_CHECK(dim % delta_dim == 0, "dims should be dividable by delta_dim");
    TORCH_CHECK(dstate <= MAX_DSTATE, "selective_scan only supports state dimension <= 256");
    
    CHECK_SHAPE(u, batch_size, dim, nrows, ncols);
    CHECK_SHAPE(delta, batch_size, delta_dim, nrows, ncols);
    CHECK_SHAPE(A, dim, dstate);
    if (A_decay_.has_value()) {
        auto A_decay = A_decay_.value();
        CHECK_SHAPE(A_decay, 1);
    }
    CHECK_SHAPE(B, batch_size, n_groups, dstate, nrows, ncols);
    TORCH_CHECK(B.stride(-1) == 1 || B.size(-1) == 1);
    CHECK_SHAPE(C, batch_size, n_groups, dstate, nrows, ncols);
    TORCH_CHECK(C.stride(-1) == 1 || C.size(-1) == 1);
    CHECK_SHAPE(dout, batch_size, dim, nrows, ncols);

    if (D_.has_value()) {
        auto D = D_.value();
        TORCH_CHECK(D.scalar_type() == at::ScalarType::Float);
        TORCH_CHECK(D.is_cuda());
        TORCH_CHECK(D.stride(-1) == 1 || D.size(-1) == 1);
        CHECK_SHAPE(D, dim);
    }

    if (delta_bias_.has_value()) {
        auto delta_bias = delta_bias_.value();
        TORCH_CHECK(delta_bias.scalar_type() == at::ScalarType::Float);
        TORCH_CHECK(delta_bias.is_cuda());
        TORCH_CHECK(delta_bias.stride(-1) == 1 || delta_bias.size(-1) == 1);
        CHECK_SHAPE(delta_bias, delta_dim);
    }

    at::Tensor out;

    const int kMin = c10::elementSize(u.scalar_type()) == 2 ? 2 : 1;
    const int nChunksY = (nrows + THREAD_ITEMS_Y - 1) / THREAD_ITEMS_Y;
    const int nChunksX = (ncols + MAX_THREADS_X * THREAD_ITEMS_X * kMin - 1) / (MAX_THREADS_X * THREAD_ITEMS_X * kMin); // max is 128 * THREAD_WIDTH in fwd_kernel

    
    if (nChunksX > 1 || nChunksY > 1) { TORCH_CHECK(x_.has_value()); }
    if (x_.has_value()) {
        auto x = x_.value();
        TORCH_CHECK(x.scalar_type() == weight_type);
        TORCH_CHECK(x.is_cuda());
        TORCH_CHECK(x.is_contiguous());
        CHECK_SHAPE(x, batch_size, dim, dstate, nChunksX, nrows * 2);
    }
    if (x_sd_.has_value()) {
        auto x_sd = x_sd_.value();
        TORCH_CHECK(x_sd.scalar_type() == weight_type);
        TORCH_CHECK(x_sd.is_cuda());
        TORCH_CHECK(x_sd.is_contiguous());
        CHECK_SHAPE(x_sd, batch_size, dim, dstate, nChunksX, nrows * 2);
    }
    if (x_v_.has_value()) {
        auto x_v = x_v_.value();
        TORCH_CHECK(x_v.scalar_type() == weight_type);
        TORCH_CHECK(x_v.is_cuda());
        TORCH_CHECK(x_v.is_contiguous());
        CHECK_SHAPE(x_v, batch_size, dim, dstate, nChunksY, ncols);  
    }

    at::Tensor du = torch::empty_like(u);
    at::Tensor ddelta = torch::empty_like(u);
    at::Tensor dA = torch::zeros_like(A);
    at::Tensor dA_decay;
    if (A_decay_.has_value()) { dA_decay = torch::zeros_like(A_decay_.value()); }
    at::Tensor dB = torch::zeros_like(B, B.options().dtype(torch::kFloat32));
    at::Tensor dC = torch::zeros_like(C, C.options().dtype(torch::kFloat32));
    at::Tensor dD;
    if (D_.has_value()) { dD = torch::zeros_like(D_.value()); }
    at::Tensor ddelta_bias;
    if (delta_bias_.has_value()) { ddelta_bias = torch::zeros({dim}, delta_bias_.value().options()); }

    SSMParamsBwd params;
    set_ssm_params_bwd(params, nChunksY, nChunksX,
                       u, delta, A, B, C, out,
                       D_.has_value() ? D_.value().data_ptr() : nullptr,
                       delta_bias_.has_value() ? delta_bias_.value().data_ptr() : nullptr,
                       A_decay_.has_value() ? A_decay_.value().data_ptr() : nullptr,
                       x_.has_value() ? x_.value().data_ptr() : nullptr,
                       x_sd_.has_value() ? x_sd_.value().data_ptr() : nullptr,
                       x_v_.value(),
                       dout, du, ddelta, dA, dB, dC,
                       D_.has_value() ? dD.data_ptr() : nullptr,
                       delta_bias_.has_value() ? ddelta_bias.data_ptr() : nullptr,
                       A_decay_.has_value() ? dA_decay.data_ptr() : nullptr,
                       delta_softplus);

    // Otherwise the kernel will be launched from cuda:0 device
    // Cast to char to avoid compiler warning about narrowing
    at::cuda::CUDAGuard device_guard{(char)u.get_device()};
    auto stream = at::cuda::getCurrentCUDAStream().stream();
    DISPATCH_ITYPE_FLOAT_AND_HALF_AND_BF16(u.scalar_type(), "selective_scan_bwd", [&] {
        if (output_type == input_type) {
            selective_scan_bwd_cuda<input_t, weight_t, input_t>(params, stream);
        } else {
            selective_scan_bwd_cuda<input_t, weight_t, float>(params, stream);
        }
    });

    std::vector<at::Tensor> result = {du, ddelta, dA, dB.to(B.dtype()), dC.to(C.dtype()), dD, ddelta_bias, dA_decay};
    if (dim != delta_dim) {
        result[1] = ddelta.view({batch_size, delta_dim, delta_dim_repeat, nrows, ncols}).sum(2);
        if (delta_bias_.has_value()) {
            result[6] = ddelta_bias.view({delta_dim, delta_dim_repeat}).sum(1);
        }
    }
    return result;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fwd", &selective_scan_fwd, "Selective nd scan forward");
    m.def("bwd", &selective_scan_bwd, "Selective nd scan backward");
}

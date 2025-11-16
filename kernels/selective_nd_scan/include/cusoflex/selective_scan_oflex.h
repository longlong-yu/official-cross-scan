#pragma once


struct SSMParamsBase {
    using index_t = uint32_t;

    int batch, dim, dstate, n_groups, nrows, ncols, nChunksY, nChunksX;
    int dim_ngroups_ratio, dim_deltagroups_ratio;

    bool delta_softplus;

    index_t A_d_stride;
    index_t A_dstate_stride;
    index_t B_batch_stride;
    index_t B_d_stride;
    index_t B_dstate_stride;
    index_t B_group_stride;
    index_t B_h_stride;
    index_t C_batch_stride;
    index_t C_d_stride;
    index_t C_dstate_stride;
    index_t C_group_stride;
    index_t C_h_stride;
    index_t u_batch_stride;
    index_t u_d_stride;
    index_t u_h_stride;
    index_t delta_batch_stride;
    index_t delta_d_stride;
    index_t delta_h_stride;
    index_t out_batch_stride;
    index_t out_d_stride;
    index_t out_h_stride;
    index_t x_v_batch_stride;
    index_t x_v_d_stride;
    index_t x_v_dstate_stride;
    index_t x_v_chunk_stride;

    // Common data pointers.
    void *__restrict__ A_ptr;
    void *__restrict__ A_decay_ptr;
    void *__restrict__ B_ptr;
    void *__restrict__ C_ptr;
    void *__restrict__ D_ptr;
    void *__restrict__ u_ptr;
    void *__restrict__ delta_ptr;
    void *__restrict__ delta_bias_ptr;
    void *__restrict__ out_ptr;
    void *__restrict__ x_ptr;
    void *__restrict__ x_sd_ptr;
    void *__restrict__ x_v_ptr;
};

struct SSMParamsBwd: public SSMParamsBase {
    index_t dout_batch_stride;
    index_t dout_d_stride;
    index_t dout_h_stride;
    index_t dA_d_stride;
    index_t dA_dstate_stride;
    index_t dB_batch_stride;
    index_t dB_group_stride;
    index_t dB_d_stride;
    index_t dB_dstate_stride;
    index_t dB_h_stride;
    index_t dC_batch_stride;
    index_t dC_group_stride;
    index_t dC_d_stride;
    index_t dC_dstate_stride;
    index_t dC_h_stride;
    index_t du_batch_stride;
    index_t du_d_stride;
    index_t du_h_stride;
    index_t ddelta_batch_stride;
    index_t ddelta_d_stride;
    index_t ddelta_h_stride;

    // Common data pointers.
    void *__restrict__ dout_ptr;
    void *__restrict__ dA_ptr;
    void *__restrict__ dA_decay_ptr;
    void *__restrict__ dB_ptr;
    void *__restrict__ dC_ptr;
    void *__restrict__ dD_ptr;
    void *__restrict__ du_ptr;
    void *__restrict__ ddelta_ptr;
    void *__restrict__ ddelta_bias_ptr;
};

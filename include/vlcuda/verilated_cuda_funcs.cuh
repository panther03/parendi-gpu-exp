// -*- mode: C++; c-file-style: "cc-mode" -*-
//*************************************************************************
//
// Code available from: https://verilator.org
//
// Copyright 2003-2023 by Wilson Snyder. This program is free software; you can
// redistribute it and/or modify it under the terms of either the GNU
// Lesser General Public License Version 3 or the Perl Artistic License
// Version 2.0.
// SPDX-License-Identifier: LGPL-3.0-only OR Artistic-2.0
//
//*************************************************************************
///
/// \file
/// \brief Verilated common functions
///
/// verilated.h should be included instead of this file.
///
/// Those macro/function/variable starting or ending in _ are internal,
/// however many of the other function/macros here are also internal.
///
//*************************************************************************

#ifndef VERILATOR_VERILATED_CUDA_FUNCS_H_
#define VERILATOR_VERILATED_CUDA_FUNCS_H_

#ifndef VERILATOR_VERILATED_H_INTERNAL_
#error "verilated_cuda_funcs.h should only be included by verilated.h"
#endif

#include <cooperative_groups.h>

namespace cg = cooperative_groups;

using state_update_fn_t = void(*)(void*,void*);

#define ITERS 500 // placeholder until dpi is working

__global__ void gpuSimulate(void *topState, state_update_fn_t *compLut, state_update_fn_t *xchgLoadLut) {
    extern __shared__ void* localState;
    cg::grid_group g = cg::this_grid();
    for (int i = 0; i < ITERS; i++) {
        compLut[blockIdx.x](topState, localState);
        g.sync();
        xchgLoadLut[blockIdx.x](topState, localState);
        g.sync();
    }
}

template <typename Derived, typename TopStateTy>
class VBspCudaTileCls {
    private:
    __device__ inline void compute(TopStateTy *topState);
    __device__ inline void exchangeLoad(TopStateTy *topState);
    // __device__ inline void exchangeStore(TopStateTy *topState);
    public:
    static __device__ __noinline__ computeWrap(void* topState, void *localState) {
        ((Derived*)localState)->compute((TopStateTy*)topState);
    }

    static __device__ __noinline__ exchangeLoadWrap(void* topState, void *localState) {
        ((Derived*)localState)->exchangeLoad((TopStateTy*)topState);
    }

};

template <typename Derived, typename TopStateTy, typename ...Tiles>
class VBspCudaRootCls {
    private:
    static const std::size_t numTiles = sizeof...(Tiles);
    static const state_update_fn_t *h_compLut = {&Tiles::computeWrap...};
    static const state_update_fn_t *h_xchgLoadLut = {&Tiles::xchgLoadLut...};
    __device__ state_update_fn_t *d_compLut = nullptr;
    __device__ state_update_fn_t *d_xchgLoadLut = nullptr;

    public:
    TopStateTy *h_topState; // add to constructor ??
    __device__ TopStateTy *d_topState = nullptr;

    private: 

    void initCompute(); 
    // implemented by verilator code 
    // it could be implemented at the TopStateTy level... hmm...

    void gpuInitialize() {
        // TODO
        //d_compLut = cudaMalloc(...);
        //d_xchgLoadLut = cudaMalloc(...);
        // ....

        // copy it over
        // cudaMemcpy(...)
    }

    public:

    void initialize() {
        ((Derived*)this)->initCompute();
        gpuInitialize();
    }

    void simulate() {
        gpuSimulate((void*)d_topState, d_compLut, d_xchgLoadLut);
    }
};

#endif // VERILATOR_VERILATED_CUDA_FUNCS_H_

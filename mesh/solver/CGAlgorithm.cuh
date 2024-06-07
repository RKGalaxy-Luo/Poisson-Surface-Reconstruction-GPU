/*****************************************************************//**
 * \file   CGAlgorithm.cuh
 * \brief  ÌÝ¶ÈÏÂ½µ·¨
 * 
 * \author LUOJIAXUAN
 * \date   May 27th 2024
 *********************************************************************/
#pragma once
#ifndef CG_ALGORITHM_CUH
#define CG_ALGORITHM_CUH

#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <base/DeviceAPI/safe_call.hpp>

#include <base/GlobalConfigs.h>

#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>

#define ENABLE_CPU_DEBUG_CODE 0
#define THREADS_PER_BLOCK 512

#ifndef EXIT_WAIVED
#define EXIT_WAIVED 2
#endif

namespace cg = cooperative_groups;

extern "C" __global__ void gpuConjugateGradient(int* I, int* J, float* val, float* x, float* Ax, float* p, float* r, double* dot_result, int nnz, int N, float tol);

namespace SparseSurfelFusion {
    namespace device {
        __device__ void gpuSpMV(int* I, int* J, float* val, int nnz, int num_rows, float alpha, float* inputVecX, float* outputVecY, cg::thread_block& cta, const cg::grid_group& grid);

        __device__ void gpuSaxpy(float* x, float* y, float a, int size, const cg::grid_group& grid);

        __device__ void gpuDotProduct(float* vecA, float* vecB, double* result, int size, const cg::thread_block& cta, const cg::grid_group& grid);

        __device__ void gpuCopyVector(float* srcA, float* destB, int size, const cg::grid_group& grid);

        __device__ void gpuScaleVectorAndSaxpy(const float* x, float* y, float a, float scale, int size, const cg::grid_group& grid);
    
        __global__ void gpuGetTestSummary(int* I, int* J, float* val, float* x, float* rhs, int* err, int num);
    }

    /* genTridiag: generate a random tridiagonal symmetric matrix */
    void genTridiag(int* I, int* J, float* val, int N, int nz);

    // I - contains location of the given non-zero element in the row of the matrix
    // J - contains location of the given non-zero element in the column of the
    // matrix val - contains values of the given non-zero elements of the matrix
    // inputVecX - input vector to be multiplied
    // outputVecY - resultant vector
    void cpuSpMV(int* I, int* J, float* val, int nnz, int num_rows, float alpha, float* inputVecX, float* outputVecY);

    double dotProduct(float* vecA, float* vecB, int size);

    void scaleVector(float* vec, float alpha, int size);

    void saxpy(float* x, float* y, float a, int size);

    void cpuConjugateGrad(int* I, int* J, float* val, float* x, float* Ax, float* p, float* r, int nnz, int N, float tol);

    bool areAlmostEqual(float a, float b, float maxRelDiff);

    void solverCG_DeviceToDevice(const int& N, const int& nz, int* I, int* J, float* val, float* rhs, float* x, cudaStream_t stream);

}

#endif // !CG_ALGORITHM_CUH


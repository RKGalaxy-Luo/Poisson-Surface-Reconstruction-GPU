/*
 * Software License Agreement (BSD License)
 *
 *  Copyright (c) 2011, Willow Garage, Inc.
 *  All rights reserved.
 *
 *  Redistribution and use in source and binary forms, with or without
 *  modification, are permitted provided that the following conditions
 *  are met:
 *
 *   * Redistributions of source code must retain the above copyright
 *     notice, this list of conditions and the following disclaimer.
 *   * Redistributions in binary form must reproduce the above
 *     copyright notice, this list of conditions and the following
 *     disclaimer in the documentation and/or other materials provided
 *     with the distribution.
 *   * Neither the name of Willow Garage, Inc. nor the names of its
 *     contributors may be used to endorse or promote products derived
 *     from this software without specific prior written permission.
 *
 *  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 *  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 *  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
 *  FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
 *  COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
 *  INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 *  BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 *  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 *  CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 *  LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
 *  ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 *  POSSIBILITY OF SUCH DAMAGE.
 *
 *  Author: Anatoly Baskeheev, Itseez Ltd, (myname.mysurname@mycompany.com)
 */

#ifndef __PCL_CUDA_SAFE_CALL_HPP__
#define __PCL_CUDA_SAFE_CALL_HPP__

#include <cstdio>
#include <exception>
#include <stdexcept>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cublas_v2.h>
#include <cublas_api.h>
#include <iostream>
#include <stdlib.h>
#include <stdio.h>

#if defined(__GNUC__)
    #define cudaSafeCall(expr)  pcl::gpu::___cudaSafeCall(expr, __FILE__, __LINE__, __func__)
#else /* defined(__CUDACC__) || defined(__MSVC__) */
    #define cudaSafeCall(expr)  pcl::gpu::___cudaSafeCall(expr, __FILE__, __LINE__)    
#endif

// 获得CUDA常用操作对应的报错
#define CHECKCUDA(call)                                 \
do                                                      \
{                                                       \
    const cudaError_t error_code = call;                \
    if (error_code != cudaSuccess)                      \
    {                                                   \
        printf("CUDA Error:\n");                        \
        printf("    错误文件:       %s\n", __FILE__);   \
        printf("    错误行数:       %d\n", __LINE__);   \
        printf("    错误内容:       %s\n",              \
            cudaGetErrorString(error_code));            \
        exit(1);                                        \
    }                                                   \
} while (0)

//检查CUDA驱动的API是否报错
#define CHECKCUDADRIVER(call)                           \
do                                                      \
{                                                       \
    CUresult error = call;                              \
    const char* error_name;                             \
    const char* error_string;                           \
    cuGetErrorName(error, &error_name);                 \
    cuGetErrorString(error, &error_string);             \
    if (error != CUDA_SUCCESS)                          \
    {                                                   \
        printf("CUDA Driver Error:\n");                 \
        printf("    错误文件:       %s\n", __FILE__);   \
        printf("    错误行数:       %d\n", __LINE__);   \
        printf("    错误内容:       %s：%s\n",          \
            error_name, error_string);                  \
        exit(1);                                        \
    }                                                   \
} while (0)                                             \

//获得CUBLAS的报错
#define CHECKCUBLAS(call)                               \
do                                                      \
{                                                       \
    const cublasStatus_t error = call;                  \
    if(CUBLAS_STATUS_SUCCESS != error) {                \
        printf("CUBLAS Error:\n");                      \
        printf("    错误文件:       %s\n", __FILE__);   \
        printf("    错误行数:       %d\n", __LINE__);   \
        printf("    错误内容:       %d\n", error);      \
        cudaDeviceReset();                              \
        exit(1);                                        \
    }                                                   \
} while(0)                                              \

namespace pcl
{
    namespace gpu
    {
        static inline void ___cudaSafeCall(cudaError_t err, const char *file, const int line, const char *func = "")
        {
            if (cudaSuccess != err){
                std::cout << "Error: " << "\t" << file << ":" << line << std::endl;
                exit(0);
            }
        }

        static inline int divUp(int total, int grain) { return (total + grain - 1) / grain; }
    }

    namespace device
    {
        using pcl::gpu::divUp;        
    }
}


#endif /* __PCL_CUDA_SAFE_CALL_HPP__ */

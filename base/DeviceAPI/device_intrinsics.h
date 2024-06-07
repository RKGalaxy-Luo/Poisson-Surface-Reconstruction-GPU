/*****************************************************************//**
 * \file   device_intrinsics.h
 * \brief  用来处理一个线程束中的快速归约加法，这个头文件只能在.cu文件中包含，因为这里是在PTX编译的时候告诉编译器不要优化这段汇编代码
 * 
 * \author LUO
 * \date   February 4th 2024
 *********************************************************************/
#pragma once
#include <vector_functions.h>

namespace SparseSurfelFusion {
        /**
         * \brief 扫描相加(float类型).
         * 
         * \param x 当前线程束中第一个线程的数据
         * \param offset 线程束中的偏移量(bit位右移)
         * \return 同一个线程束数据相加的结果
         */
        __device__ __forceinline__ float shfl_add(float x, int offset) {
            float result = 0;
            //__asm__ 或asm用来声明一个内联汇编表达式，任何内联汇编表达式都是以它开头，必不可少
            //volatile 是可选的，假如用了它，则是向GCC 声明不答应对该内联汇编优化
            asm volatile (
                "{.reg .f32 r0;"
                ".reg .pred p;"
                "shfl.sync.up.b32 r0|p, %1, %2, 0, 0;"
                "@p add.f32 r0, r0, %1;"
                "mov.f32 %0, r0;}"
                : "=f"(result) : "f"(x), "r"(offset));

            return result;
        }

    /**
     * \brief 线程束内扫描相加(float类型).
     * 
     * \param data 线程束第一个数据(bit位右移相加)
     * \return 同一个线程束数据相加的结果
     */
     __device__ __forceinline__ float warp_scan(float data) {
         data = shfl_add(data, 1);
         data = shfl_add(data, 2);
         data = shfl_add(data, 4);
         data = shfl_add(data, 8);
         data = shfl_add(data, 16);
         return data;
    }

    /**
     * \brief 扫描相加(int 类型).
     * 
     * \param x 当前线程束中第一个线程的数据
     * \param offset 线程束中的偏移量(bit位右移)
     * \return 同一个线程束数据相加的结果
     */
    __device__ __forceinline__ int shfl_add(int x, int offset)
    {
        int result = 0;
        asm volatile (
            "{.reg .s32 r0;"
            ".reg .pred p;"
            "shfl.sync.up.b32 r0|p, %1, %2, 0, 0;"
            "@p add.s32 r0, r0, %1;"
            "mov.s32 %0, r0;}"
            : "=r"(result) : "r"(x), "r"(offset));

        return result;
    }

    /**
     * \brief 线程束内扫描相加(int类型).
     * 
     * \param data 线程束第一个数据(bit位右移相加)
     * \return 同一个线程束数据相加的结果
     */
    
    __device__ __forceinline__ int warp_scan(int data)
    {
        data = shfl_add(data, 1);
        data = shfl_add(data, 2);
        data = shfl_add(data, 4);
        data = shfl_add(data, 8);
        data = shfl_add(data, 16);
        return data;
    }

    /**
     * \brief 扫描相加(unsigned int 类型).
     * 
     * \param x 当前线程束中第一个线程的数据
     * \param offset 线程束中的偏移量(bit位右移)
     * \return 同一个线程束数据相加的结果
     */
    __device__ __forceinline__ unsigned int shfl_add(unsigned int x, unsigned int offset)
    {
        unsigned int result = 0;
        asm volatile (
            "{.reg .u32 r0;"
            ".reg .pred p;"
            "shfl.sync.up.b32 r0|p, %1, %2, 0, 0;"
            "@p add.u32 r0, r0, %1;"
            "mov.u32 %0, r0;}"
            : "=r"(result) : "r"(x), "r"(offset));

        return result;
    }

    /**
     * \brief 线程束内扫描相加(unsigned int类型).
     * 
     * \param data 线程束第一个数据(bit位右移相加)
     * \return 同一个线程束数据相加的结果
     */
    __device__ __forceinline__ unsigned int warp_scan(unsigned int data)
    {
        data = shfl_add(data, 1u);
        data = shfl_add(data, 2u);
        data = shfl_add(data, 4u);
        data = shfl_add(data, 8u);
        data = shfl_add(data, 16u);
        return data;
    }
}

/*****************************************************************//**
 * \file   CommonUtils.h
 * \brief  主要是记录一些常用函数，纹理内存分配，开辟等
 * 
 * \author LUO
 * \date   January 9th 2024
 *********************************************************************/
#pragma once
#include "CommonTypes.h"
#include <cuda_texture_types.h>
#include <cuda_runtime_api.h>
#include <cuda.h>
#include <driver_types.h>
#include <exception>
#include <cuda.h>


 //  此类型不能隐式拷贝/赋值
#define NO_COPY_ASSIGN(TypeName)                        \
    TypeName(const TypeName&) = delete;                 \
    TypeName& operator=(const TypeName&) = delete

// 此类型不可以隐式拷贝/赋值/移动
#define NO_COPY_ASSIGN_MOVE(TypeName)                   \
    TypeName(const TypeName&) = delete;                 \
    TypeName& operator=(const TypeName&) = delete;      \
    TypeName(TypeName&&) = delete;                      \
    TypeName& operator=(TypeName&&) = delete

// 此类默认移动
#define DEFAULT_MOVE(TypeName)                          \
	TypeName(TypeName&&) noexcept = default;            \
	TypeName& operator=(TypeName&&) noexcept = default

// 此类型构造和析构函数均为默认
#define DEFAULT_CONSTRUCT_DESTRUCT(TypeName)            \
    TypeName() = default;                               \
    ~TypeName() = default

namespace SparseSurfelFusion {
	
    /**
     * \brief 将a，b交换，通用模板函数
     * \param a 交换参数a
     * \param b 交换参数b
     * \return 
     */
    template <typename T> // __forceinline__在编译的时候就把函数放到对应调用的位置，加速了算法过程，增加了编译文件的大小
    __host__ __device__ __forceinline__ void swap(T& a, T& b) noexcept
    {
        T c(a); a = b; b = c;
    }


    /**
     * \brief 内参的倒数
     * \param intrinsic 相机的内参矩阵
     * \return 
     */
    __host__ __forceinline__ IntrinsicInverse inverse(const Intrinsic& intrinsic) {
        IntrinsicInverse intr_inv;
        intr_inv.principal_x = intrinsic.principal_x;
        intr_inv.principal_y = intrinsic.principal_y;
        intr_inv.inv_focal_x = 1.0f / intrinsic.focal_x;
        intr_inv.inv_focal_y = 1.0f / intrinsic.focal_y;
        return intr_inv;
    }


    /**
     * \brief 初始化cuda操作的上下文和驱动程序的API
     * \param selected_device 选择使用的GPU的ID.
     */
    CUcontext initCudaContext(int selected_device = 0);



    /**
     * \brief 在程序结束时清除cuda上下文
     * \param context cuda上下文
     */
    void destroyCudaContext(CUcontext context);

/*************************************** 为GPU分配1维，2维纹理内存 ***************************************/


    /**
     * \brief 创建默认的2D纹理描述子，用来指明cudaArray_t应该是多少通道，声明数据类型的纹理数据.
     *
     * \param descriptor cuda纹理描述子.
     */
    void createDefault2DTextureDescriptor(cudaTextureDesc& descriptor);

    /**
     * \brief 创建默认的2D资源描述子，用来指明cudaArray_t应该是多少通道，声明数据类型的纹理数据.
     *
     * \param descriptor cuda资源描述子
     * \param cudaArray 传入的资源数据
     */
    void createDefault2DResourceDescriptor(cudaResourceDesc& descriptor, cudaArray_t& cudaArray);

    /**
     * \brief 创建分配深度纹理内存，数据类型为uint16(16位无符号整型).
     *
     * \param rows 图像的高
     * \param cols 图像的宽
     * \param texture 需要分配创建的图像纹理(输入cudaTextureObject_t类型)
     * \param cudaArray 需要传入的深度纹理数据(输入cudaArray_t类型)
     */
    void createDepthTexture(const unsigned int rows, const unsigned int cols, cudaTextureObject_t& texture, cudaArray_t& cudaArray);
    
    /**
      * \brief 创建分配深度纹理及表面内存，数据类型为uint16(16位无符号整型).
      *
      * \param rows 图像的高
      * \param cols 图像的宽
      * \param texture 需要分配创建的图像纹理(输入cudaTextureObject_t类型)
      * \param surface 需要分配创建的图像表面(输入cudaSurfaceObject_t类型)
      * \param cudaArray 需要传入的深度纹理(表面)数据(输入cudaArray_t类型)
      */
    void createDepthTextureSurface(const unsigned int rows, const unsigned int cols, cudaTextureObject_t& texture, cudaSurfaceObject_t& surface, cudaArray_t& cudaArray);
    
    /**
     * \brief 创建分配深度纹理及表面内存，数据类型为uint16(16位无符号整型).
     *
     * \param rows 图像的高
     * \param cols 图像的宽
     * \param collect 需要传入的纹理及表面数据(输入CudaTextureSurface类型)
     */
    void createDepthTextureSurface(const unsigned int rows, const unsigned int cols, CudaTextureSurface& collect);



    /**
     * \brief 创建分配2维Float1类型的纹理及表面内存(为平均场推理创建2D float1纹理(和表面)).
     *
     * \param rows 图像的高
     * \param cols 图像的宽
     * \param texture 需要分配的纹理内存(以cudaTextureObject_t形式输入)
     * \param surface 需要分配的表面内存(以cudaSurfaceObject_t形式输入)
     * \param cudaArray 数据的地址
     */
    void createFloat1TextureSurface(const unsigned int rows, const unsigned int cols, cudaTextureObject_t& texture, cudaSurfaceObject_t& surface, cudaArray_t& cudaArray);

    /**
     * \brief 创建分配2维Float1类型的纹理及表面内存(为平均场推理创建2D float1纹理(和表面)).
     *
     * \param rows 图像的高
     * \param cols 图像的宽
     * \param textureCollect 需要分配的纹理及表面数据地址(以CudaTextureSurface形式输入)
     */
    void createFloat1TextureSurface(const unsigned int rows, const unsigned int cols, CudaTextureSurface& textureCollect);

    /**
     * \brief 创建分配2维Float2类型的纹理及表面内存(为梯度图创建2D float2纹理(和表面)).
     *
     * \param rows 图像的高
     * \param cols 图像的宽
     * \param texture 需要分配的纹理内存(以cudaTextureObject_t形式输入)
     * \param surface 需要分配的表面内存(以cudaSurfaceObject_t形式输入)
     * \param cudaArray 数据的地址
     */
    void createFloat2TextureSurface(const unsigned int rows, const unsigned int cols, cudaTextureObject_t& texture, cudaSurfaceObject_t& surface, cudaArray_t& cudaArray);

    /**
     * \brief 创建分配2维Float2类型的纹理及表面内存(为梯度图创建2D float2纹理(和表面)).
     *
     * \param rows 图像的高
     * \param cols 图像的宽
     * \param textureCollect 需要分配的纹理及表面数据地址(以CudaTextureSurface形式输入)
     */
    void createFloat2TextureSurface(const unsigned int rows, const unsigned int cols, CudaTextureSurface& textureCollect);

    /**
     * \brief 创建分配2维UChar1类型的纹理及表面数据(为二进制Mask图像创建2D uchar1纹理(和表面)).
     *
     * \param rows 图像的高
     * \param cols 图像的宽
     * \param texture 需要分配的纹理内存(以cudaTextureObject_t形式输入)
     * \param surface 需要分配的表面内存(以cudaSurfaceObject_t形式输入)
     * \param cudaArray 数据的地址
     */
    void createUChar1TextureSurface(const unsigned rows, const unsigned cols, cudaTextureObject_t& texture, cudaSurfaceObject_t& surface, cudaArray_t& cudaArray);

    /**
     * \brief 创建分配2维UChar1类型的纹理及表面数据(为二进制Mask图像创建2D uchar1纹理(和表面)).
     *
     * \param rows 图像的高
     * \param cols 图像的宽
     * \param textureCollect 需要分配的纹理及表面数据地址(以CudaTextureSurface形式输入)
     */
    void createUChar1TextureSurface(const unsigned rows, const unsigned cols, CudaTextureSurface& textureCollect);

    /**
     * \brief 分配二维图像的float4类型纹理cudaTextureObject_t和表面cudaSurfaceObject_t内存.
     *
     * \param rows 图像的高
     * \param cols 图像的宽
     * \param texture 需要开辟的纹理内存
     * \param surface 需要开辟的表面内存
     * \param cudaArray 存入的数据纹理(表面)数据
     */
    void createFloat4TextureSurface(const unsigned int rows, const unsigned int cols, cudaTextureObject_t& texture, cudaSurfaceObject_t& surface, cudaArray_t& cudaArray);

    /**
     * \brief 分配二维图像的float4类型纹理CudaTextureSurface内存.
     *
     * \param rows 图像的高
     * \param cols 图像的宽
     * \param textureCollect 需要开辟的纹理表面内存
     */
    void createFloat4TextureSurface(const unsigned int rows, const unsigned int cols, CudaTextureSurface& textureCollect);

    /**
     * \brief 释放2D纹理
     *
     * \param textureCollect 需要释放的纹理(传入CudaTextureSurface类型)
     */
    void releaseTextureCollect(CudaTextureSurface& textureCollect);

    /**
     * \brief 2D纹理查询函数，查询2D Array和2D线性内存在2个维度下的尺寸(CUDA API中使用结构体cudaExtent描述3D Array和3D线性内存在三个维度上的尺寸)
     *
     * \param texture 需要查询的2D纹理
     * \param width 纹理在2维下的尺寸(此处填宽)
     * \param height 纹理在2维下的尺寸(此处填高)
     */
    void query2DTextureExtent(cudaTextureObject_t texture, unsigned int& width, unsigned int& height);

}
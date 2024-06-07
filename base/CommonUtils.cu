/*****************************************************************//**
 * \file   CommonUtils.cu
 * \brief  文件主要包含一些常用的cuda工具函数
 * 
 * \author LUO
 * \date   January 12th 2024
 *********************************************************************/
#include "CommonUtils.h"


CUcontext SparseSurfelFusion::initCudaContext(int selected_device) {
    //初始化Cuda驱动的API
    CHECKCUDADRIVER(cuInit(0));

    //Query the device
    int device_count = 0;
    CHECKCUDADRIVER(cuDeviceGetCount(&device_count));
    for (auto dev_idx = 0; dev_idx < device_count; dev_idx++) {
        char dev_name[256] = { 0 };
        CHECKCUDADRIVER(cuDeviceGetName(dev_name, 256, dev_idx));
        printf("device %d: %s\n", dev_idx, dev_name);
    }

    //选择GPU
    printf("设备 %d 被用作并行处理器.\n", selected_device);
    CUdevice cuda_device;
    CHECKCUDADRIVER(cuDeviceGet(&cuda_device, selected_device));

    //创建cuda上下文
    CUcontext cuda_context;
    CHECKCUDADRIVER(cuCtxCreate(&cuda_context, CU_CTX_SCHED_AUTO, cuda_device));
    return cuda_context;
}

void SparseSurfelFusion::destroyCudaContext(CUcontext context)
{
    cudaDeviceSynchronize();
    CHECKCUDADRIVER(cuCtxDestroy(context));
}



void SparseSurfelFusion::createDefault2DTextureDescriptor(cudaTextureDesc& descriptor)
{
    memset(&descriptor, 0, sizeof(descriptor));
    // 纹理寻址模式，使用3个维度(实际上只使用2维)
    descriptor.addressMode[0] = cudaAddressModeBorder;  // 在边界之外返回0
    descriptor.addressMode[1] = cudaAddressModeBorder;
    descriptor.addressMode[2] = cudaAddressModeBorder;
    // 从纹理获取时要使用的过滤模式
    descriptor.filterMode = cudaFilterModePoint;        // 最邻近插值--cudaFilterModePoint       双线性插值--cudaFilterModeLinear
    // 指定是否应将整数数据转换为浮点数
    descriptor.readMode = cudaReadModeElementType;      // 读数据以指定的数据类型读，不全部转化成float
    // 是否将纹理坐标标准化
    descriptor.normalizedCoords = 0;                    // 不使用归一化纹理内存

}

void SparseSurfelFusion::createDefault2DResourceDescriptor(cudaResourceDesc& descriptor, cudaArray_t& cudaArray)
{
    memset(&descriptor, 0, sizeof(cudaResourceDesc));   // 资源描述子初值为0
    // 使用CUDA数组--cudaResourceTypeArray      
    // 使用CUDA映射数组--cudaResourceTypeMipmappedArray      
    // 使用设备上一段线性内存--cudaResourceTypeLinear
    // 使用设备上一个2D块资源
    descriptor.resType = cudaResourceTypeArray;         
    descriptor.res.array.array = cudaArray;             // 将值的内存段赋入
}

void SparseSurfelFusion::createDepthTexture(const unsigned int rows, const unsigned int cols, cudaTextureObject_t& texture, cudaArray_t& cudaArray)
{
    // 声明纹理描述
    cudaTextureDesc depth_texture_desc;
    createDefault2DTextureDescriptor(depth_texture_desc);
    // 声明通道描述(只有一个通道的数据，数据类型是uint16)
    cudaChannelFormatDesc depth_channel_desc = cudaCreateChannelDesc(16, 0, 0, 0, cudaChannelFormatKindUnsigned); // 16bit的无符号整型
    // 分配cuda数组
    CHECKCUDA(cudaMallocArray(&cudaArray, &depth_channel_desc, cols, rows));
    // 声明资源描述
    cudaResourceDesc resource_desc;
    createDefault2DResourceDescriptor(resource_desc, cudaArray); // 初始化资源描述子并将资源数据cudaArray赋值进去
    // 分配纹理内存
    CHECKCUDA(cudaCreateTextureObject(&texture, &resource_desc, &depth_texture_desc, 0));
}

void SparseSurfelFusion::createDepthTextureSurface(const unsigned int rows, const unsigned int cols, cudaTextureObject_t& texture, cudaSurfaceObject_t& surface, cudaArray_t& cudaArray)
{
    //纹理描述
    cudaTextureDesc depth_texture_description;
    createDefault2DTextureDescriptor(depth_texture_description);
    //创建通道描述
    cudaChannelFormatDesc depth_channel_desc = cudaCreateChannelDesc(16, 0, 0, 0, cudaChannelFormatKindUnsigned);
    //分配cuda数组
    CHECKCUDA(cudaMallocArray(&cudaArray, &depth_channel_desc, cols, rows));
    //创建资源desc
    cudaResourceDesc resource_desc;
    createDefault2DResourceDescriptor(resource_desc, cudaArray); // 初始化资源描述子并将资源数据cudaArray赋值进去
    //分配纹理
    CHECKCUDA(cudaCreateTextureObject(&texture, &resource_desc, &depth_texture_description, 0));
    CHECKCUDA(cudaCreateSurfaceObject(&surface, &resource_desc));
}

void SparseSurfelFusion::createDepthTextureSurface(const unsigned int rows, const unsigned int cols, CudaTextureSurface& collect)
{
    createDepthTextureSurface(rows, cols,collect.texture, collect.surface, collect.cudaArray);
}

void SparseSurfelFusion::createFloat1TextureSurface(const unsigned int rows, const unsigned int cols, cudaTextureObject_t& texture, cudaSurfaceObject_t& surface, cudaArray_t& cudaArray)
{
    //纹理描述
    cudaTextureDesc float1_texture_desc;
    createDefault2DTextureDescriptor(float1_texture_desc);
    //创建通道描述，使用指定类型返回通道描述子，并填入每一个通道分量的bit数 (下述为1个通道，此通道数据位数为32bit)
    cudaChannelFormatDesc float1_channel_desc = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);
    //分配cuda数组
    CHECKCUDA(cudaMallocArray(&cudaArray, &float1_channel_desc, cols, rows));
    //创建资源desc
    cudaResourceDesc resource_desc;
    createDefault2DResourceDescriptor(resource_desc, cudaArray); // 初始化资源描述子并将资源数据cudaArray赋值进去
    //分配纹理
    CHECKCUDA(cudaCreateTextureObject(&texture, &resource_desc, &float1_texture_desc, 0));
    CHECKCUDA(cudaCreateSurfaceObject(&surface, &resource_desc));
}

void SparseSurfelFusion::createFloat1TextureSurface(const unsigned int rows, const unsigned int cols, CudaTextureSurface& textureCollect)
{
    createFloat1TextureSurface(rows, cols, textureCollect.texture, textureCollect.surface, textureCollect.cudaArray);
}

void SparseSurfelFusion::createFloat2TextureSurface(const unsigned int rows, const unsigned int cols, cudaTextureObject_t& texture, cudaSurfaceObject_t& surface, cudaArray_t& cudaArray)
{
    //纹理描述
    cudaTextureDesc float2_texture_desc;
    createDefault2DTextureDescriptor(float2_texture_desc);
    //创建通道描述，使用指定类型返回通道描述子，并填入每一个通道分量的bit数 (下述为2个通道，此通道数据位数为32bit)
    cudaChannelFormatDesc float2_channel_desc = cudaCreateChannelDesc(32, 32, 0, 0, cudaChannelFormatKindFloat);
    //分配cuda数组
    CHECKCUDA(cudaMallocArray(&cudaArray, &float2_channel_desc, cols, rows));
    //创建资源desc
    cudaResourceDesc resource_desc;
    createDefault2DResourceDescriptor(resource_desc, cudaArray); // 初始化资源描述子并将资源数据cudaArray赋值进去
    //分配纹理
    CHECKCUDA(cudaCreateTextureObject(&texture, &resource_desc, &float2_texture_desc, 0));
    CHECKCUDA(cudaCreateSurfaceObject(&surface, &resource_desc));
}

void SparseSurfelFusion::createFloat2TextureSurface(const unsigned int rows, const unsigned int cols, CudaTextureSurface& textureCollect)
{
    createFloat2TextureSurface(rows, cols, textureCollect.texture, textureCollect.surface, textureCollect.cudaArray);
}

void SparseSurfelFusion::createUChar1TextureSurface(const unsigned rows, const unsigned cols, cudaTextureObject_t& texture, cudaSurfaceObject_t& surface, cudaArray_t& cudaArray)
{
    //纹理描述
    cudaTextureDesc uchar1_texture_desc;
    createDefault2DTextureDescriptor(uchar1_texture_desc);
    //创建通道描述，使用指定类型返回通道描述子，并填入每一个通道分量的bit数 (下述为1个通道，此通道数据位数为8bit)
    cudaChannelFormatDesc uchar1_channel_desc = cudaCreateChannelDesc(8, 0, 0, 0, cudaChannelFormatKindUnsigned);
    //分配cuda数组
    CHECKCUDA(cudaMallocArray(&cudaArray, &uchar1_channel_desc, cols, rows));
    //创建资源desc
    cudaResourceDesc resource_desc;
    createDefault2DResourceDescriptor(resource_desc, cudaArray); // 初始化资源描述子并将资源数据cudaArray赋值进去
    //分配纹理
    CHECKCUDA(cudaCreateTextureObject(&texture, &resource_desc, &uchar1_texture_desc, 0));
    CHECKCUDA(cudaCreateSurfaceObject(&surface, &resource_desc));
}

void SparseSurfelFusion::createUChar1TextureSurface(const unsigned rows, const unsigned cols, CudaTextureSurface& textureCollect)
{
    createUChar1TextureSurface(rows, cols, textureCollect.texture, textureCollect.surface, textureCollect.cudaArray);
}


void SparseSurfelFusion::createFloat4TextureSurface(const unsigned int rows, const unsigned int cols, cudaTextureObject_t& texture, cudaSurfaceObject_t& surface, cudaArray_t& cudaArray)
{
    // 声明并初始化纹理描述子
    cudaTextureDesc float4_texture_desc; // 纹理描述子
    createDefault2DTextureDescriptor(float4_texture_desc); // 将纹理描述子初始化

    // 创造通道描述，使用指定类型返回通道描述子，并填入每一个通道分量的bit数
    // 使用float类型返回描述子，并且每个通道分量都是32bit位
    cudaChannelFormatDesc float4_channel_desc = cudaCreateChannelDesc(32, 32, 32, 32, cudaChannelFormatKindFloat); // 返回了一个float类型的通道描述子

    // 根据通道描述子float4_channel_desc分配cuda的内存
    CHECKCUDA(cudaMallocArray(&cudaArray, &float4_channel_desc, cols, rows));

    // 创建资源描述子
    cudaResourceDesc resource_desc;
    createDefault2DResourceDescriptor(resource_desc, cudaArray); // 初始化资源描述子并将资源数据cudaArray赋值进去

    // 分配纹理内存
    CHECKCUDA(cudaCreateTextureObject(&texture, &resource_desc, &float4_texture_desc, 0));
    // 分配表面内存
    CHECKCUDA(cudaCreateSurfaceObject(&surface, &resource_desc));
}

void SparseSurfelFusion::createFloat4TextureSurface(const unsigned int rows, const unsigned int cols, CudaTextureSurface& textureCollect)
{
    createFloat4TextureSurface(rows, cols, textureCollect.texture, textureCollect.surface, textureCollect.cudaArray);
}

void SparseSurfelFusion::releaseTextureCollect(CudaTextureSurface& textureCollect)
{
    CHECKCUDA(cudaDestroyTextureObject(textureCollect.texture));
    CHECKCUDA(cudaDestroySurfaceObject(textureCollect.surface));
    CHECKCUDA(cudaFreeArray(textureCollect.cudaArray));
}

void SparseSurfelFusion::query2DTextureExtent(cudaTextureObject_t texture, unsigned int& width, unsigned int& height)
{
    cudaResourceDesc texture_res;
    cudaSafeCall(cudaGetTextureObjectResourceDesc(&texture_res, texture));
    cudaArray_t cu_array = texture_res.res.array.array;
    cudaChannelFormatDesc channel_desc;
    cudaExtent extent;
    unsigned int flag;
    cudaSafeCall(cudaArrayGetInfo(&channel_desc, &extent, &flag, cu_array));
    width = extent.width;
    height = extent.height;
}

/*****************************************************************//**
 * \file   CommonUtils.cu
 * \brief  �ļ���Ҫ����һЩ���õ�cuda���ߺ���
 * 
 * \author LUO
 * \date   January 12th 2024
 *********************************************************************/
#include "CommonUtils.h"


CUcontext SparseSurfelFusion::initCudaContext(int selected_device) {
    //��ʼ��Cuda������API
    CHECKCUDADRIVER(cuInit(0));

    //Query the device
    int device_count = 0;
    CHECKCUDADRIVER(cuDeviceGetCount(&device_count));
    for (auto dev_idx = 0; dev_idx < device_count; dev_idx++) {
        char dev_name[256] = { 0 };
        CHECKCUDADRIVER(cuDeviceGetName(dev_name, 256, dev_idx));
        printf("device %d: %s\n", dev_idx, dev_name);
    }

    //ѡ��GPU
    printf("�豸 %d ���������д�����.\n", selected_device);
    CUdevice cuda_device;
    CHECKCUDADRIVER(cuDeviceGet(&cuda_device, selected_device));

    //����cuda������
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
    // ����Ѱַģʽ��ʹ��3��ά��(ʵ����ֻʹ��2ά)
    descriptor.addressMode[0] = cudaAddressModeBorder;  // �ڱ߽�֮�ⷵ��0
    descriptor.addressMode[1] = cudaAddressModeBorder;
    descriptor.addressMode[2] = cudaAddressModeBorder;
    // �������ȡʱҪʹ�õĹ���ģʽ
    descriptor.filterMode = cudaFilterModePoint;        // ���ڽ���ֵ--cudaFilterModePoint       ˫���Բ�ֵ--cudaFilterModeLinear
    // ָ���Ƿ�Ӧ����������ת��Ϊ������
    descriptor.readMode = cudaReadModeElementType;      // ��������ָ�����������Ͷ�����ȫ��ת����float
    // �Ƿ����������׼��
    descriptor.normalizedCoords = 0;                    // ��ʹ�ù�һ�������ڴ�

}

void SparseSurfelFusion::createDefault2DResourceDescriptor(cudaResourceDesc& descriptor, cudaArray_t& cudaArray)
{
    memset(&descriptor, 0, sizeof(cudaResourceDesc));   // ��Դ�����ӳ�ֵΪ0
    // ʹ��CUDA����--cudaResourceTypeArray      
    // ʹ��CUDAӳ������--cudaResourceTypeMipmappedArray      
    // ʹ���豸��һ�������ڴ�--cudaResourceTypeLinear
    // ʹ���豸��һ��2D����Դ
    descriptor.resType = cudaResourceTypeArray;         
    descriptor.res.array.array = cudaArray;             // ��ֵ���ڴ�θ���
}

void SparseSurfelFusion::createDepthTexture(const unsigned int rows, const unsigned int cols, cudaTextureObject_t& texture, cudaArray_t& cudaArray)
{
    // ������������
    cudaTextureDesc depth_texture_desc;
    createDefault2DTextureDescriptor(depth_texture_desc);
    // ����ͨ������(ֻ��һ��ͨ�������ݣ�����������uint16)
    cudaChannelFormatDesc depth_channel_desc = cudaCreateChannelDesc(16, 0, 0, 0, cudaChannelFormatKindUnsigned); // 16bit���޷�������
    // ����cuda����
    CHECKCUDA(cudaMallocArray(&cudaArray, &depth_channel_desc, cols, rows));
    // ������Դ����
    cudaResourceDesc resource_desc;
    createDefault2DResourceDescriptor(resource_desc, cudaArray); // ��ʼ����Դ�����Ӳ�����Դ����cudaArray��ֵ��ȥ
    // ���������ڴ�
    CHECKCUDA(cudaCreateTextureObject(&texture, &resource_desc, &depth_texture_desc, 0));
}

void SparseSurfelFusion::createDepthTextureSurface(const unsigned int rows, const unsigned int cols, cudaTextureObject_t& texture, cudaSurfaceObject_t& surface, cudaArray_t& cudaArray)
{
    //��������
    cudaTextureDesc depth_texture_description;
    createDefault2DTextureDescriptor(depth_texture_description);
    //����ͨ������
    cudaChannelFormatDesc depth_channel_desc = cudaCreateChannelDesc(16, 0, 0, 0, cudaChannelFormatKindUnsigned);
    //����cuda����
    CHECKCUDA(cudaMallocArray(&cudaArray, &depth_channel_desc, cols, rows));
    //������Դdesc
    cudaResourceDesc resource_desc;
    createDefault2DResourceDescriptor(resource_desc, cudaArray); // ��ʼ����Դ�����Ӳ�����Դ����cudaArray��ֵ��ȥ
    //��������
    CHECKCUDA(cudaCreateTextureObject(&texture, &resource_desc, &depth_texture_description, 0));
    CHECKCUDA(cudaCreateSurfaceObject(&surface, &resource_desc));
}

void SparseSurfelFusion::createDepthTextureSurface(const unsigned int rows, const unsigned int cols, CudaTextureSurface& collect)
{
    createDepthTextureSurface(rows, cols,collect.texture, collect.surface, collect.cudaArray);
}

void SparseSurfelFusion::createFloat1TextureSurface(const unsigned int rows, const unsigned int cols, cudaTextureObject_t& texture, cudaSurfaceObject_t& surface, cudaArray_t& cudaArray)
{
    //��������
    cudaTextureDesc float1_texture_desc;
    createDefault2DTextureDescriptor(float1_texture_desc);
    //����ͨ��������ʹ��ָ�����ͷ���ͨ�������ӣ�������ÿһ��ͨ��������bit�� (����Ϊ1��ͨ������ͨ������λ��Ϊ32bit)
    cudaChannelFormatDesc float1_channel_desc = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);
    //����cuda����
    CHECKCUDA(cudaMallocArray(&cudaArray, &float1_channel_desc, cols, rows));
    //������Դdesc
    cudaResourceDesc resource_desc;
    createDefault2DResourceDescriptor(resource_desc, cudaArray); // ��ʼ����Դ�����Ӳ�����Դ����cudaArray��ֵ��ȥ
    //��������
    CHECKCUDA(cudaCreateTextureObject(&texture, &resource_desc, &float1_texture_desc, 0));
    CHECKCUDA(cudaCreateSurfaceObject(&surface, &resource_desc));
}

void SparseSurfelFusion::createFloat1TextureSurface(const unsigned int rows, const unsigned int cols, CudaTextureSurface& textureCollect)
{
    createFloat1TextureSurface(rows, cols, textureCollect.texture, textureCollect.surface, textureCollect.cudaArray);
}

void SparseSurfelFusion::createFloat2TextureSurface(const unsigned int rows, const unsigned int cols, cudaTextureObject_t& texture, cudaSurfaceObject_t& surface, cudaArray_t& cudaArray)
{
    //��������
    cudaTextureDesc float2_texture_desc;
    createDefault2DTextureDescriptor(float2_texture_desc);
    //����ͨ��������ʹ��ָ�����ͷ���ͨ�������ӣ�������ÿһ��ͨ��������bit�� (����Ϊ2��ͨ������ͨ������λ��Ϊ32bit)
    cudaChannelFormatDesc float2_channel_desc = cudaCreateChannelDesc(32, 32, 0, 0, cudaChannelFormatKindFloat);
    //����cuda����
    CHECKCUDA(cudaMallocArray(&cudaArray, &float2_channel_desc, cols, rows));
    //������Դdesc
    cudaResourceDesc resource_desc;
    createDefault2DResourceDescriptor(resource_desc, cudaArray); // ��ʼ����Դ�����Ӳ�����Դ����cudaArray��ֵ��ȥ
    //��������
    CHECKCUDA(cudaCreateTextureObject(&texture, &resource_desc, &float2_texture_desc, 0));
    CHECKCUDA(cudaCreateSurfaceObject(&surface, &resource_desc));
}

void SparseSurfelFusion::createFloat2TextureSurface(const unsigned int rows, const unsigned int cols, CudaTextureSurface& textureCollect)
{
    createFloat2TextureSurface(rows, cols, textureCollect.texture, textureCollect.surface, textureCollect.cudaArray);
}

void SparseSurfelFusion::createUChar1TextureSurface(const unsigned rows, const unsigned cols, cudaTextureObject_t& texture, cudaSurfaceObject_t& surface, cudaArray_t& cudaArray)
{
    //��������
    cudaTextureDesc uchar1_texture_desc;
    createDefault2DTextureDescriptor(uchar1_texture_desc);
    //����ͨ��������ʹ��ָ�����ͷ���ͨ�������ӣ�������ÿһ��ͨ��������bit�� (����Ϊ1��ͨ������ͨ������λ��Ϊ8bit)
    cudaChannelFormatDesc uchar1_channel_desc = cudaCreateChannelDesc(8, 0, 0, 0, cudaChannelFormatKindUnsigned);
    //����cuda����
    CHECKCUDA(cudaMallocArray(&cudaArray, &uchar1_channel_desc, cols, rows));
    //������Դdesc
    cudaResourceDesc resource_desc;
    createDefault2DResourceDescriptor(resource_desc, cudaArray); // ��ʼ����Դ�����Ӳ�����Դ����cudaArray��ֵ��ȥ
    //��������
    CHECKCUDA(cudaCreateTextureObject(&texture, &resource_desc, &uchar1_texture_desc, 0));
    CHECKCUDA(cudaCreateSurfaceObject(&surface, &resource_desc));
}

void SparseSurfelFusion::createUChar1TextureSurface(const unsigned rows, const unsigned cols, CudaTextureSurface& textureCollect)
{
    createUChar1TextureSurface(rows, cols, textureCollect.texture, textureCollect.surface, textureCollect.cudaArray);
}


void SparseSurfelFusion::createFloat4TextureSurface(const unsigned int rows, const unsigned int cols, cudaTextureObject_t& texture, cudaSurfaceObject_t& surface, cudaArray_t& cudaArray)
{
    // ��������ʼ������������
    cudaTextureDesc float4_texture_desc; // ����������
    createDefault2DTextureDescriptor(float4_texture_desc); // �����������ӳ�ʼ��

    // ����ͨ��������ʹ��ָ�����ͷ���ͨ�������ӣ�������ÿһ��ͨ��������bit��
    // ʹ��float���ͷ��������ӣ�����ÿ��ͨ����������32bitλ
    cudaChannelFormatDesc float4_channel_desc = cudaCreateChannelDesc(32, 32, 32, 32, cudaChannelFormatKindFloat); // ������һ��float���͵�ͨ��������

    // ����ͨ��������float4_channel_desc����cuda���ڴ�
    CHECKCUDA(cudaMallocArray(&cudaArray, &float4_channel_desc, cols, rows));

    // ������Դ������
    cudaResourceDesc resource_desc;
    createDefault2DResourceDescriptor(resource_desc, cudaArray); // ��ʼ����Դ�����Ӳ�����Դ����cudaArray��ֵ��ȥ

    // ���������ڴ�
    CHECKCUDA(cudaCreateTextureObject(&texture, &resource_desc, &float4_texture_desc, 0));
    // ��������ڴ�
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

/*****************************************************************//**
 * \file   AlgorithmTypes.cu
 * \brief  对一些基础算法类中方法的简单实现，主要是调用cuh库的相关算法
 * 
 * \author LUO
 * \date   January 31st 2024
 *********************************************************************/

#include "AlgorithmTypes.h"
#if defined(__CUDACC__)		//如果由NVCC编译器编译
#include <cub/cub.cuh>
#endif

namespace SparseSurfelFusion {
    namespace device {

        /**
         * \brief 初始化seleted_input的核函数，将选择的输出作为原始数组中的索引.
         * 
         * \param selection_input_buffer 需要初始化的容器
         * \return 
         */
        __global__ void selectionIndexInitKernel(PtrSize<int> selection_input_buffer) {
            const auto idx = threadIdx.x + blockIdx.x * blockDim.x;
            if (idx < selection_input_buffer.size)
                selection_input_buffer.data[idx] = idx;     // PtrSize继承DevPtr类型，data是DevPtr类型中的公共成员变量，因此可以访问
        }
    }
}


void SparseSurfelFusion::FlagSelection::AllocateAndInit(size_t input_size, cudaStream_t stream)
{
    if (m_selection_input_buffer.size() >= input_size) return;

    // 在需要的时候清除内存
    if (m_selected_idx_buffer.size() > 0) {
        m_selected_idx_buffer.release();
        m_selection_input_buffer.release();
        m_temp_storage.release();
        selectIndicatorBuffer.release();
    }

    // 分配新的缓存
    size_t allocate_size = 3 * input_size / 2;

    // 分配:所选索引的最大大小与选择输入相同
    m_selected_idx_buffer.create(allocate_size);
    m_selection_input_buffer.create(allocate_size);
    selectIndicatorBuffer.create(allocate_size);

    // 查看所需要的暂存数据容器的大小
    size_t temp_storage_bytes = 0;  // 运行算法暂存容器的大小
    cub::DeviceSelect::Flagged(NULL, temp_storage_bytes, m_selection_input_buffer.ptr(), m_selected_idx_buffer.ptr(), validSelectedIndex.ptr(), m_device_num_selected, allocate_size, stream);

    m_temp_storage.create(temp_storage_bytes);
    // 初始化m_selection_input_buffer
    dim3 block(128);
    dim3 grid(divUp(m_selection_input_buffer.size(), block.x));
    device::selectionIndexInitKernel << <grid, block, 0, stream >> > (m_selection_input_buffer);
    return;
}

void SparseSurfelFusion::FlagSelection::Select(const DeviceArray<char>& flags, cudaStream_t stream)
{
    // 分配并初始化
    AllocateAndInit(flags.size(), stream);

    // 构建选择数组
    DeviceArray<int> selection_idx_input = DeviceArray<int>(m_selection_input_buffer.ptr(), flags.size());
    validSelectedIndex = DeviceArray<int>(m_selected_idx_buffer.ptr(), flags.size());

    // 筛选
    size_t temp_storage_bytes = m_temp_storage.sizeBytes();
    cub::DeviceSelect::Flagged(m_temp_storage.ptr(), temp_storage_bytes, selection_idx_input.ptr(), flags.ptr(), validSelectedIndex.ptr(), m_device_num_selected, (int)flags.size(), stream, false);

    // 将GPU数据拷贝到CPU中
    CHECKCUDA(cudaMemcpyAsync(m_host_num_selected, m_device_num_selected, sizeof(int), cudaMemcpyDeviceToHost, stream));
    CHECKCUDA(cudaStreamSynchronize(stream));   // 需要做流同步，以便host能正常的访问m_host_num_selected

    // 矫正并赋值输出选择有效Index的结果
    validSelectedIndex = DeviceArray<int>(m_selected_idx_buffer.ptr(), *m_host_num_selected);
    return;
}

void SparseSurfelFusion::FlagSelection::SelectUnsigned(const DeviceArray<char>& flags, const DeviceArray<unsigned int>& selectFrom, DeviceArray<unsigned int>& selectToBuffer, DeviceArray<unsigned int>& validSelectToArray, cudaStream_t stream)
{
    // 分配并初始化flags数组
    AllocateAndInit(flags.size(), stream);

    // 筛选
    size_t temp_storage_bytes = m_temp_storage.sizeBytes();
    cub::DeviceSelect::Flagged(m_temp_storage.ptr(), temp_storage_bytes, selectFrom.ptr(), flags.ptr(), selectToBuffer.ptr(), m_device_num_selected, (int)flags.size(), stream, false);

    // Host访问前需要同步
    CHECKCUDA(cudaMemcpyAsync(m_host_num_selected, m_device_num_selected, sizeof(int), cudaMemcpyDeviceToHost, stream));
    CHECKCUDA(cudaStreamSynchronize(stream));

    // 现在可以安全地访问内存输出了
    validSelectToArray = DeviceArray<unsigned>(selectToBuffer.ptr(), *m_host_num_selected);
    return;
}

void SparseSurfelFusion::UniqueSelection::Allocate(size_t input_size)
{
    if (m_selected_element_buffer.size() >= input_size) return;

    //Clear existing cache
    if (m_selected_element_buffer.size() > 0) {
        m_selected_element_buffer.release();
        m_temp_storage.release();
    }

    //Allocate new storages
    size_t allocate_size = 3 * input_size / 2;
    m_selected_element_buffer.create(allocate_size);

    //Query the required buffer
    size_t temp_storage_bytes = 0;
    cub::DeviceSelect::Unique(
        m_temp_storage.ptr(), temp_storage_bytes,
        m_selected_element_buffer.ptr(), //The input and output are not used in querying
        m_selected_element_buffer.ptr(), m_device_num_selected,
        (int)m_selected_element_buffer.size()
    );

    m_temp_storage.create(temp_storage_bytes);
}

void SparseSurfelFusion::UniqueSelection::Select(const DeviceArray<int>& key_in, cudaStream_t stream, bool debug_sync)
{
    //Check and allocate required buffer
    Allocate(key_in.size());

    //Do selection
    size_t temp_storage_bytes = m_temp_storage.sizeBytes();
    cub::DeviceSelect::Unique(
        m_temp_storage.ptr(), temp_storage_bytes,
        key_in.ptr(), //The input
        m_selected_element_buffer.ptr(), m_device_num_selected, //The output
        (int)key_in.size(), stream, debug_sync
    );

    //Need sync before host accessing
    CHECKCUDA(cudaMemcpyAsync(m_host_num_selected, m_device_num_selected, sizeof(int), cudaMemcpyDeviceToHost, stream));
    CHECKCUDA(cudaStreamSynchronize(stream));

    //Construct the size-correct result
    valid_selected_element = DeviceArray<int>(m_selected_element_buffer.ptr(), *m_host_num_selected);
}

__host__ void SparseSurfelFusion::PrefixSum::AllocateBuffer(size_t input_size)
{
    //不需要分配
    if (m_prefixsum_buffer.size() >= input_size) return;

    //如果存在缓冲区，清除它们
    if (m_prefixsum_buffer.size() > 0) {
        m_prefixsum_buffer.release();
        m_temp_storage.release();
    }

    //做分配
    m_prefixsum_buffer.create(input_size);
    //查询临时存储以获取输入大小
    size_t prefixsum_bytes = 0;
    cub::DeviceScan::InclusiveSum(NULL, prefixsum_bytes,
        m_prefixsum_buffer.ptr(), m_prefixsum_buffer.ptr(), (int)input_size, 0);
    m_temp_storage.create(prefixsum_bytes);
    return;
}

__host__ void SparseSurfelFusion::PrefixSum::InclusiveSum(const DeviceArray<unsigned>& array_in, cudaStream_t stream, bool debug_sync)
{
    //Allocate the buffer if not enough
    AllocateBuffer(array_in.size());

    //Construct the result array
    valid_prefixsum_array = DeviceArray<unsigned>(m_prefixsum_buffer.ptr(), array_in.size());

    //Do prefixsum
    size_t inclusive_sum_bytes = m_temp_storage.sizeBytes();
    cub::DeviceScan::InclusiveSum(m_temp_storage, inclusive_sum_bytes,
        array_in.ptr(), valid_prefixsum_array.ptr(), (int)array_in.size(), stream,
        debug_sync);
    return;
}

__host__ void SparseSurfelFusion::PrefixSum::InclusiveSum(const DeviceArrayView<unsigned>& array_in, cudaStream_t stream)
{
    //如果缓冲区不够，则分配缓冲区
    AllocateBuffer(array_in.Size()); //这个地方已经默认给m_temp_storage赋大小了

    //构造结果数组valid_prefixsum_array是GPU缓存，valid_prefixsum_array的地址就是m_prefixsum_buffer地址
    valid_prefixsum_array = DeviceArray<unsigned>(m_prefixsum_buffer.ptr(), array_in.Size());
    //做前置和
    size_t inclusive_sum_bytes = m_temp_storage.sizeBytes();//m_temp_storage 就是 m_prefixsum_buffer的大小
    //计算设备范围内的前缀和。前项和：InclusiveSum(a[0],a[1],a[2]) = (a[0], a[0]+a[1], a[0]+a[1]+a[2])
    cub::DeviceScan::InclusiveSum(m_temp_storage, inclusive_sum_bytes, array_in.RawPtr(), valid_prefixsum_array.ptr(), (int)array_in.Size(), stream, false);
    return;
}

__host__ void SparseSurfelFusion::PrefixSum::ExclusiveSum(const DeviceArray<unsigned>& array_in, cudaStream_t stream, bool debug_sync)
{
    //Allocate the buffer if not enough
    AllocateBuffer(array_in.size());

    //Construct the result array
    valid_prefixsum_array = DeviceArray<unsigned>(m_prefixsum_buffer.ptr(), array_in.size());

    //Do prefixsum
    size_t exclusive_sum_bytes = m_temp_storage.sizeBytes();
    cub::DeviceScan::ExclusiveSum(m_temp_storage, exclusive_sum_bytes,
        array_in.ptr(), valid_prefixsum_array.ptr(), (int)array_in.size(), stream,
        debug_sync);
    return;
}


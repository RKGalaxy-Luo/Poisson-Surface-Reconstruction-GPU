#include "AlgorithmTypes.h"
#if defined(__CUDACC__)		//如果由NVCC编译器编译
#include <cub/cub.cuh>
#endif

template<typename KeyT, typename ValueT>
void SparseSurfelFusion::KeyValueSort<KeyT, ValueT>::AllocateBuffer(size_t input_size)
{
    if (input_size <= m_sorted_key_buffer.size()) return;

    //如果存在缓冲区，则清除它们
    if (m_sorted_key_buffer.size() > 0) {
        m_sorted_key_buffer.release();
        m_sorted_value_buffer.release();
        m_temp_storage.release();
    }

    //分配缓存
    size_t allocate_size = 3 * input_size / 2;
    m_sorted_key_buffer.create(allocate_size);
    m_sorted_value_buffer.create(allocate_size);

    //查询所需的临时存储
    size_t required_temp_bytes = 0;
    cub::DeviceRadixSort::SortPairs < KeyT, ValueT >(NULL, required_temp_bytes, m_sorted_key_buffer.ptr(), m_sorted_key_buffer.ptr(), m_sorted_value_buffer.ptr(), m_sorted_value_buffer.ptr(), (int)m_sorted_key_buffer.size());

    //分配所需的存储
    m_temp_storage.create(required_temp_bytes);
}

template<typename KeyT, typename ValueT>
void SparseSurfelFusion::KeyValueSort<KeyT, ValueT>::Sort(const DeviceArray<KeyT>& key_in, const DeviceArray<ValueT>& value_in, cudaStream_t stream, int end_bit, bool debug_sync)
{
    //如果缓冲区不够，则分配缓冲区
    AllocateBuffer(key_in.size());

    //构造大小正确的结果
    valid_sorted_key = DeviceArray<KeyT>(m_sorted_key_buffer.ptr(), key_in.size());
    valid_sorted_value = DeviceArray<ValueT>(m_sorted_value_buffer.ptr(), value_in.size());

    //做排序
    size_t required_temp_bytes = m_temp_storage.sizeBytes();
    cub::DeviceRadixSort::SortPairs<KeyT, ValueT>(
        m_temp_storage.ptr(), required_temp_bytes, 
        key_in.ptr(), valid_sorted_key.ptr(), 
        value_in.ptr(), valid_sorted_value.ptr(),
        (int)key_in.size(), 
        0, end_bit, 
        stream, debug_sync);
}

template<typename KeyT, typename ValueT>
void SparseSurfelFusion::KeyValueSort<KeyT, ValueT>::Sort(const DeviceArrayView<KeyT>& key_in, const DeviceArrayView<ValueT>& value_in, cudaStream_t stream)
{
    //如果缓冲区不够，则分配缓冲区
    AllocateBuffer(key_in.Size());

    //构造大小正确的结果
    valid_sorted_key = DeviceArray<KeyT>(m_sorted_key_buffer.ptr(), key_in.Size());
    valid_sorted_value = DeviceArray<ValueT>(m_sorted_value_buffer.ptr(), value_in.Size());

    //做排序
    size_t required_temp_bytes = m_temp_storage.sizeBytes();

    //就是根据键来排列这个键值对，即便排列完成，Key和Value也是一一对应的
    cub::DeviceRadixSort::SortPairs<KeyT, ValueT>(
        m_temp_storage.ptr(), required_temp_bytes,
        key_in.RawPtr(), valid_sorted_key.ptr(),		//key_in.RawPtr()排列好给valid_sorted_key.ptr() （地址传递）
        value_in.RawPtr(), valid_sorted_value.ptr(),	//value_in.RawPtr()排列好了给valid_sorted_value.ptr  (地址传递)
        (int)key_in.Size(),
        0, 8 * sizeof(KeyT),
        stream, false
        );
}

template<typename KeyT, typename ValueT>
void SparseSurfelFusion::KeyValueSort<KeyT, ValueT>::Sort(const DeviceArrayView<KeyT>& key_in, const DeviceArrayView<ValueT>& value_in, int end_bit, cudaStream_t stream)
{
    //如果缓冲区不够，则分配缓冲区
    AllocateBuffer(key_in.Size());

    //构造大小正确的结果
    valid_sorted_key = DeviceArray<KeyT>(m_sorted_key_buffer.ptr(), key_in.Size());
    valid_sorted_value = DeviceArray<ValueT>(m_sorted_value_buffer.ptr(), value_in.Size());

    //做排序
    size_t required_temp_bytes = m_temp_storage.sizeBytes();
    cub::DeviceRadixSort::SortPairs<KeyT, ValueT>(
        m_temp_storage.ptr(), required_temp_bytes,
        key_in.RawPtr(), valid_sorted_key.ptr(),
        value_in.RawPtr(), valid_sorted_value.ptr(),
        (int)key_in.Size(),
        0, end_bit,
        stream, false
        );
}

template<typename KeyT, typename ValueT>
void SparseSurfelFusion::KeyValueSort<KeyT, ValueT>::Sort(const DeviceArray<KeyT>& key_in, cudaStream_t stream, int end_bit, bool debug_sync)
{
    //如果缓冲区不够，则分配缓冲区
    AllocateBuffer(key_in.size());

    //构造大小正确的结果
    valid_sorted_key = DeviceArray<KeyT>(m_sorted_key_buffer.ptr(), key_in.size());

    //调用排序器
    size_t required_temp_bytes = m_temp_storage.sizeBytes();
    cub::DeviceRadixSort::SortKeys(
        m_temp_storage.ptr(), required_temp_bytes,
        key_in.ptr(), valid_sorted_key.ptr(), key_in.size(),
        0, end_bit,
        stream,
        debug_sync
    );
}


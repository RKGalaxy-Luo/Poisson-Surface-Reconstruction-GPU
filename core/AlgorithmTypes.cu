/*****************************************************************//**
 * \file   AlgorithmTypes.cu
 * \brief  ��һЩ�����㷨���з����ļ�ʵ�֣���Ҫ�ǵ���cuh�������㷨
 * 
 * \author LUO
 * \date   January 31st 2024
 *********************************************************************/

#include "AlgorithmTypes.h"
#if defined(__CUDACC__)		//�����NVCC����������
#include <cub/cub.cuh>
#endif

namespace SparseSurfelFusion {
    namespace device {

        /**
         * \brief ��ʼ��seleted_input�ĺ˺�������ѡ��������Ϊԭʼ�����е�����.
         * 
         * \param selection_input_buffer ��Ҫ��ʼ��������
         * \return 
         */
        __global__ void selectionIndexInitKernel(PtrSize<int> selection_input_buffer) {
            const auto idx = threadIdx.x + blockIdx.x * blockDim.x;
            if (idx < selection_input_buffer.size)
                selection_input_buffer.data[idx] = idx;     // PtrSize�̳�DevPtr���ͣ�data��DevPtr�����еĹ�����Ա��������˿��Է���
        }
    }
}


void SparseSurfelFusion::FlagSelection::AllocateAndInit(size_t input_size, cudaStream_t stream)
{
    if (m_selection_input_buffer.size() >= input_size) return;

    // ����Ҫ��ʱ������ڴ�
    if (m_selected_idx_buffer.size() > 0) {
        m_selected_idx_buffer.release();
        m_selection_input_buffer.release();
        m_temp_storage.release();
        selectIndicatorBuffer.release();
    }

    // �����µĻ���
    size_t allocate_size = 3 * input_size / 2;

    // ����:��ѡ����������С��ѡ��������ͬ
    m_selected_idx_buffer.create(allocate_size);
    m_selection_input_buffer.create(allocate_size);
    selectIndicatorBuffer.create(allocate_size);

    // �鿴����Ҫ���ݴ����������Ĵ�С
    size_t temp_storage_bytes = 0;  // �����㷨�ݴ������Ĵ�С
    cub::DeviceSelect::Flagged(NULL, temp_storage_bytes, m_selection_input_buffer.ptr(), m_selected_idx_buffer.ptr(), validSelectedIndex.ptr(), m_device_num_selected, allocate_size, stream);

    m_temp_storage.create(temp_storage_bytes);
    // ��ʼ��m_selection_input_buffer
    dim3 block(128);
    dim3 grid(divUp(m_selection_input_buffer.size(), block.x));
    device::selectionIndexInitKernel << <grid, block, 0, stream >> > (m_selection_input_buffer);
    return;
}

void SparseSurfelFusion::FlagSelection::Select(const DeviceArray<char>& flags, cudaStream_t stream)
{
    // ���䲢��ʼ��
    AllocateAndInit(flags.size(), stream);

    // ����ѡ������
    DeviceArray<int> selection_idx_input = DeviceArray<int>(m_selection_input_buffer.ptr(), flags.size());
    validSelectedIndex = DeviceArray<int>(m_selected_idx_buffer.ptr(), flags.size());

    // ɸѡ
    size_t temp_storage_bytes = m_temp_storage.sizeBytes();
    cub::DeviceSelect::Flagged(m_temp_storage.ptr(), temp_storage_bytes, selection_idx_input.ptr(), flags.ptr(), validSelectedIndex.ptr(), m_device_num_selected, (int)flags.size(), stream, false);

    // ��GPU���ݿ�����CPU��
    CHECKCUDA(cudaMemcpyAsync(m_host_num_selected, m_device_num_selected, sizeof(int), cudaMemcpyDeviceToHost, stream));
    CHECKCUDA(cudaStreamSynchronize(stream));   // ��Ҫ����ͬ�����Ա�host�������ķ���m_host_num_selected

    // ��������ֵ���ѡ����ЧIndex�Ľ��
    validSelectedIndex = DeviceArray<int>(m_selected_idx_buffer.ptr(), *m_host_num_selected);
    return;
}

void SparseSurfelFusion::FlagSelection::SelectUnsigned(const DeviceArray<char>& flags, const DeviceArray<unsigned int>& selectFrom, DeviceArray<unsigned int>& selectToBuffer, DeviceArray<unsigned int>& validSelectToArray, cudaStream_t stream)
{
    // ���䲢��ʼ��flags����
    AllocateAndInit(flags.size(), stream);

    // ɸѡ
    size_t temp_storage_bytes = m_temp_storage.sizeBytes();
    cub::DeviceSelect::Flagged(m_temp_storage.ptr(), temp_storage_bytes, selectFrom.ptr(), flags.ptr(), selectToBuffer.ptr(), m_device_num_selected, (int)flags.size(), stream, false);

    // Host����ǰ��Ҫͬ��
    CHECKCUDA(cudaMemcpyAsync(m_host_num_selected, m_device_num_selected, sizeof(int), cudaMemcpyDeviceToHost, stream));
    CHECKCUDA(cudaStreamSynchronize(stream));

    // ���ڿ��԰�ȫ�ط����ڴ������
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
    //����Ҫ����
    if (m_prefixsum_buffer.size() >= input_size) return;

    //������ڻ��������������
    if (m_prefixsum_buffer.size() > 0) {
        m_prefixsum_buffer.release();
        m_temp_storage.release();
    }

    //������
    m_prefixsum_buffer.create(input_size);
    //��ѯ��ʱ�洢�Ի�ȡ�����С
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
    //�������������������仺����
    AllocateBuffer(array_in.Size()); //����ط��Ѿ�Ĭ�ϸ�m_temp_storage����С��

    //����������valid_prefixsum_array��GPU���棬valid_prefixsum_array�ĵ�ַ����m_prefixsum_buffer��ַ
    valid_prefixsum_array = DeviceArray<unsigned>(m_prefixsum_buffer.ptr(), array_in.Size());
    //��ǰ�ú�
    size_t inclusive_sum_bytes = m_temp_storage.sizeBytes();//m_temp_storage ���� m_prefixsum_buffer�Ĵ�С
    //�����豸��Χ�ڵ�ǰ׺�͡�ǰ��ͣ�InclusiveSum(a[0],a[1],a[2]) = (a[0], a[0]+a[1], a[0]+a[1]+a[2])
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


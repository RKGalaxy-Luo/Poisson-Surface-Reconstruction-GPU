/*****************************************************************//**
 * \file   AlgorithmTypes.h
 * \brief  ������������ķ�������Ҫ�ǵ���cuh�������㷨
 * 
 * \author LUO
 * \date   January 31st 2024
 *********************************************************************/
#pragma once
#include <device_launch_parameters.h>
#include <cuda_runtime_api.h>
#include <base/DeviceReadWrite/DeviceBufferArray.h>


namespace SparseSurfelFusion {

	/**
	 * \brief ���ݱ�־λ���飬ɸѡֵ�����е����ݣ������������С�ṹ��.
	 */
	struct FlagSelection {
	private:
		DeviceArray<int> m_selection_input_buffer;
		DeviceArray<int> m_selected_idx_buffer;
		DeviceArray<unsigned char> m_temp_storage;


		//�洢����ѡ�е�������������
		int* m_device_num_selected;	// GPU�ϴ洢����ѡ�е�������������
		int* m_host_num_selected;	// CPU�ϴ洢����ѡ�е�������������

	public:
		FlagSelection() {
			CHECKCUDA(cudaMalloc((void**)(&m_device_num_selected), sizeof(int)));
			CHECKCUDA(cudaMallocHost((void**)(&m_host_num_selected), sizeof(int)));
		}

		~FlagSelection() {
			CHECKCUDA(cudaFree(m_device_num_selected));
			CHECKCUDA(cudaFreeHost(m_host_num_selected));
		}

		/**
		 * \brief ���䲢��ʼ���ڴ棬��Selection�����Buffer��С��input_size����������䣻��Selection�����Buffer��С�������input_sizeС��ʱ��
		 *		  �ͷŵ�ǰm_selection_input_buffer��m_selected_idx_buffer��m_temp_storage�Ļ��棬���������·��仺�棬��input_size��1.5�����仺��,
		 *		  ��ʼ������������ǵ�ǰ���index.
		 *
		 * \param input_size ��Ҫ�����selection��buffer��С
		 * \param stream cuda��ID
		 */
		void AllocateAndInit(size_t input_size, cudaStream_t stream = 0);

		/**
		 * \brief ����flags���飬������ӦԪ�ظ���validSelectedIndex������.
		 *
		 * \param flags �����flags����
		 * \param stream cuda��ID
		 */
		void Select(const DeviceArray<char>& flags, cudaStream_t stream = 0);

		/**
		 * \brief ����flags���飬������ӦԪ�ظ���validSelectedIndex������(����ѡԪ��ֻ���޷�������ʱ).
		 *
		 * \param flags ����ı�־����
		 * \param selectFrom Ҫ��������ѡ�������
		 * \param selectToBuffer ������Ϊ�ݴ����飬ѡ���Ԫ�ؽ������Ƶ�����������н����ݴ�
		 * \param validSelectToArray ������飺ѡ���Ԫ�ؽ������Ƶ������������
		 * \param stream cuda��ID
		 */
		void SelectUnsigned(const DeviceArray<char>& flags, const DeviceArray<unsigned int>& selectFrom, DeviceArray<unsigned int>& selectToBuffer, DeviceArray<unsigned int>& validSelectToArray, cudaStream_t stream = 0);

		// ��ΪֵҪ���뺯�������ֱ�ӿ��ų�Ա����Ȩ�޹���ȡ
		DeviceArray<int> validSelectedIndex;			// ��Ч�ı�ѡ���index��validSelectedIndex��ָ��
		DeviceArray<char> selectIndicatorBuffer;		// �洢��־λ��Buffer����ΪFlagged������Ҫ����char���ͱ�־λ����

	};

	/**
	 * \brief ���������������в��Ҳ�ɾ���ظ���Ԫ��.
	 */
	struct UniqueSelection {
	private:
		DeviceArray<int> m_selected_element_buffer;
		DeviceArray<unsigned char> m_temp_storage;

		// ��ѡ�����������ڴ�
		int* m_device_num_selected;
		int* m_host_num_selected;

	public:
		/**
		 * \brief ������ѡ�����������ڴ�.
		 * 
		 */
		UniqueSelection() {
			CHECKCUDA(cudaMalloc((void**)(&m_device_num_selected), sizeof(int)));
			CHECKCUDA(cudaMallocHost((void**)(&m_host_num_selected), sizeof(int)));
		}
		/**
		 * \brief �ͷ���ѡ�����������ڴ�.
		 * 
		 */
		~UniqueSelection() {
			CHECKCUDA(cudaFree(m_device_num_selected));
			CHECKCUDA(cudaFreeHost(m_host_num_selected));
		}
		/**
		 * \brief ��������ɾ�������ظ���������Ĵ�С�����仺���Թ��㷨.
		 * 
		 * \param input_size ����ɾ�������ظ���������Ĵ�С
		 */
		void Allocate(size_t input_size);
		/**
		 * \brief ɾ��key_in�е��ظ�����.
		 * 
		 * \param key_in ��Ҫɸѡ������
		 * \param stream CUDA��ID
		 * \param debug_sync ����ָ���Ƿ�������֮ǰ����ͬ�����Խ��е���Ŀ��
		 */
		void Select(const DeviceArray<int>& key_in, cudaStream_t stream = 0, bool debug_sync = false);

		// ����Ǳ�ѡ�е�Ԫ�أ���Ϊָ�������ָ��
		DeviceArray<int> valid_selected_element;
	};


	/**
	 * \brief ִ�м� - ֵ����ģ�����ֵ���ݼ���������򣬼���λ����ֵ��λ��Ӧ����һ�µ�.
	 */
	template<typename KeyT, typename ValueT>
	struct KeyValueSort {
	private:
		//Shared buffer
		DeviceArray<unsigned char> m_temp_storage;
		DeviceArray<KeyT> m_sorted_key_buffer;
		DeviceArray<ValueT> m_sorted_value_buffer;

	public:
		/**
		 * \brief ��������������������m_temp_storage��m_sorted_key_buffer��m_sorted_value_buffer�Ļ���.
		 * 
		 * \param input_size ���仺��Ĵ�С
		 */
		void AllocateBuffer(size_t input_size);

		/**
		 * \brief ���ݼ�key_in����value_in��������.
		 * 
		 * \param key_in ��ֵ
		 * \param value_in ��ֵ
		 * \param stream CUDA��ID
		 * \param end_bit ���Ǽ��ĳ��ȣ��Զ����Ƶ���ʽ��sizeof(KeyT) -> �ֽ�����8 -> һ���ֽ��ж��ٶ����Ʒ�
		 * \param debug_sync ����ָ���Ƿ�������֮ǰ����ͬ�����Խ��е���Ŀ��
		 */
		void Sort(const DeviceArray<KeyT>& key_in, const DeviceArray<ValueT>& value_in, cudaStream_t stream = 0, int end_bit = sizeof(KeyT) * 8, bool debug_sync = false);

		/**
		 * \brief ���ݼ�key_in����value_in��������Ĭ��end_bit = sizeof(KeyT) * 8�� debug_sync = false.
		 * 
		 * \param key_in ��ֵ
		 * \param value_in ��ֵ
		 * \param stream CUDA��ID
		 */
		void Sort(const DeviceArrayView<KeyT>& key_in, const DeviceArrayView<ValueT>& value_in, cudaStream_t stream = 0);

		/**
		 * \brief ���ݼ�key_in����value_in��������debug_sync = false.
		 * 
		 * \param key_in ��ֵ
		 * \param value_in ��ֵ
		 * \param end_bit ���Ǽ��ĳ��ȣ��Զ����Ƶ���ʽ��sizeof(KeyT) -> �ֽ�����8 -> һ���ֽ��ж��ٶ����Ʒ�
		 * \param stream CUDA��ID
		 */
		void Sort(const DeviceArrayView<KeyT>& key_in, const DeviceArrayView<ValueT>& value_in, int end_bit, cudaStream_t stream = 0);

		/**
		 * \brief ֻ���������ͬ��ֵ.
		 * 
		 * \param key_in ��ֵ
		 * \param stream CUDA��
		 * \param end_bit ���Ǽ��ĳ��ȣ��Զ����Ƶ���ʽ��sizeof(KeyT) -> �ֽ�����8 -> һ���ֽ��ж��ٶ����Ʒ�
		 * \param debug_sync ����ָ���Ƿ�������֮ǰ����ͬ�����Խ��е���Ŀ��
		 */
		void Sort(const DeviceArray<KeyT>& key_in, cudaStream_t stream = 0, int end_bit = sizeof(KeyT) * 8, bool debug_sync = false);

		//Sorted value
		DeviceArray<KeyT> valid_sorted_key;			//��Ч���еļ�  �������׵�ַ��
		DeviceArray<ValueT> valid_sorted_value;		//��Ч���е�ֵ  �������׵�ַ��
	};

	/**
	 * \brief ����ǰ׺�͵Ĵ洢�ͽ����С�ṹ�壬Ӧ���ڱ��ط����ʹ��.
	 */
	struct PrefixSum {
	private:
		//�����ڴ�, ��ִ��ʱ�������̣߳�Thread����ȫ��飬���̰߳�ȫ
		DeviceArray<unsigned char> m_temp_storage;
		DeviceArray<unsigned int> m_prefixsum_buffer;		//������ɵ�ǰ׺������׵�ַ

	public:
		/**
		 * \brief ���仺��.
		 * 
		 * \param input_size �����С
		 */
		__host__ void AllocateBuffer(size_t input_size);
		/**
		 * \brief ����ǰ��ͣ�InclusiveSum(a[0],a[1],a[2]) = (a[0], a[0]+a[1], a[0]+a[1]+a[2]).
		 * 
		 * \param array_in ������ֵ
		 * \param stream CUDA��ID
		 * \param debug_sync ����ָ���Ƿ������֮ǰ����ͬ�����Խ��е���Ŀ�ġ�
		 */
		__host__ void InclusiveSum(const DeviceArray<unsigned>& array_in, cudaStream_t stream = 0, bool debug_sync = false);
		/**
		 * \brief ����InclusiveSum��InclusiveSum(a[0],a[1],a[2]) = (a[0], a[0]+a[1], a[0]+a[1]+a[2]).
		 * 
		 * \param array_in ������ֵ
		 * \param stream CUDA��ID
		 */
		__host__ void InclusiveSum(const DeviceArrayView<unsigned>& array_in, cudaStream_t stream = 0);
		/**
		 * \brief ����ExclusiveSum��ExclusiveSum(a[0],a[1],a[2],a[3]) = (1��a[0], a[0]+a[1], a[0]+a[1]+a[2]).
		 * 
		 * \param array_in ������ֵ
		 * \param stream CUDA��ID
		 * \param debug_sync ����ָ���Ƿ������֮ǰ����ͬ�����Խ��е���Ŀ��
		 * \return 
		 */
		__host__ void ExclusiveSum(const DeviceArray<unsigned>& array_in, cudaStream_t stream = 0, bool debug_sync = false);

		//�����ָ�룺��Ч��ǰ׺��
		DeviceArray<unsigned int> valid_prefixsum_array;		//��Чǰ׺��  ��������׵�ַ��
	};
}

#if defined(__CUDACC__)		//�����NVCC����������
/**
 * �ڰ��� cub/cub.cuh ͷ�ļ�ʱ��ͨ����ʹ�� __CUDACC__ ������ȷ��ֻ���� CUDA ���뻷���²Ż������ͷ�ļ���
 * ������Ϊ cub/cub.cuh �� CUDA Thrust ���е�һ��ͷ�ļ����ṩ��һЩ���ڲ��м���Ĺ��ܺ��㷨��
 */

#include "AlgorithmTypes.cuh"
#endif

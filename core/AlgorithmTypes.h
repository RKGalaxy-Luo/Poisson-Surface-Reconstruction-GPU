/*****************************************************************//**
 * \file   AlgorithmTypes.h
 * \brief  基础加速运算的方法，主要是调用cuh库的相关算法
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
	 * \brief 根据标志位数组，筛选值数组中的数据，生成新数组的小结构体.
	 */
	struct FlagSelection {
	private:
		DeviceArray<int> m_selection_input_buffer;
		DeviceArray<int> m_selected_idx_buffer;
		DeviceArray<unsigned char> m_temp_storage;


		//存储“被选中的索引”的数量
		int* m_device_num_selected;	// GPU上存储“被选中的索引”的数量
		int* m_host_num_selected;	// CPU上存储“被选中的索引”的数量

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
		 * \brief 分配并初始化内存，当Selection输入的Buffer大小比input_size大，则无需分配；当Selection输入的Buffer大小比输入的input_size小的时候，
		 *		  释放当前m_selection_input_buffer、m_selected_idx_buffer、m_temp_storage的缓存，并对其重新分配缓存，以input_size的1.5倍分配缓存,
		 *		  初始化数组的内容是当前点的index.
		 *
		 * \param input_size 需要处理的selection的buffer大小
		 * \param stream cuda流ID
		 */
		void AllocateAndInit(size_t input_size, cudaStream_t stream = 0);

		/**
		 * \brief 遍历flags数组，并将对应元素给到validSelectedIndex数组中.
		 *
		 * \param flags 传入的flags数组
		 * \param stream cuda流ID
		 */
		void Select(const DeviceArray<char>& flags, cudaStream_t stream = 0);

		/**
		 * \brief 遍历flags数组，并将对应元素给到validSelectedIndex数组中(当所选元素只是无符号数组时).
		 *
		 * \param flags 传入的标志数组
		 * \param selectFrom 要进行条件选择的数组
		 * \param selectToBuffer 此数组为暂存数组，选择的元素将被复制到该输出数组中进行暂存
		 * \param validSelectToArray 输出数组：选择的元素将被复制到该输出数组中
		 * \param stream cuda流ID
		 */
		void SelectUnsigned(const DeviceArray<char>& flags, const DeviceArray<unsigned int>& selectFrom, DeviceArray<unsigned int>& selectToBuffer, DeviceArray<unsigned int>& validSelectToArray, cudaStream_t stream = 0);

		// 因为值要传入函数，因此直接开放成员变量权限供读取
		DeviceArray<int> validSelectedIndex;			// 有效的被选择的index，validSelectedIndex是指针
		DeviceArray<char> selectIndicatorBuffer;		// 存储标志位的Buffer，因为Flagged函数需要传入char类型标志位数组

	};

	/**
	 * \brief 用于在输入数组中查找并删除重复的元素.
	 */
	struct UniqueSelection {
	private:
		DeviceArray<int> m_selected_element_buffer;
		DeviceArray<unsigned char> m_temp_storage;

		// 所选索引数量的内存
		int* m_device_num_selected;
		int* m_host_num_selected;

	public:
		/**
		 * \brief 分配所选索引数量的内存.
		 * 
		 */
		UniqueSelection() {
			CHECKCUDA(cudaMalloc((void**)(&m_device_num_selected), sizeof(int)));
			CHECKCUDA(cudaMallocHost((void**)(&m_host_num_selected), sizeof(int)));
		}
		/**
		 * \brief 释放所选索引数量的内存.
		 * 
		 */
		~UniqueSelection() {
			CHECKCUDA(cudaFree(m_device_num_selected));
			CHECKCUDA(cudaFreeHost(m_host_num_selected));
		}
		/**
		 * \brief 根据所需删除数据重复数据数组的大小，分配缓存以供算法.
		 * 
		 * \param input_size 所需删除数据重复数据数组的大小
		 */
		void Allocate(size_t input_size);
		/**
		 * \brief 删除key_in中的重复数据.
		 * 
		 * \param key_in 需要筛选的数组
		 * \param stream CUDA流ID
		 * \param debug_sync 用于指定是否在排序之前进行同步，以进行调试目的
		 */
		void Select(const DeviceArray<int>& key_in, cudaStream_t stream = 0, bool debug_sync = false);

		// 输出是被选中的元素，作为指向上面的指针
		DeviceArray<int> valid_selected_element;
	};


	/**
	 * \brief 执行键 - 值排序的，将数值根据键的情况排序，键的位置与值的位置应该是一致的.
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
		 * \brief 如果缓冲区不够，则分配m_temp_storage、m_sorted_key_buffer、m_sorted_value_buffer的缓存.
		 * 
		 * \param input_size 分配缓存的大小
		 */
		void AllocateBuffer(size_t input_size);

		/**
		 * \brief 根据键key_in，对value_in进行排序.
		 * 
		 * \param key_in 键值
		 * \param value_in 数值
		 * \param stream CUDA流ID
		 * \param end_bit 考虑键的长度，以二进制的形式，sizeof(KeyT) -> 字节数，8 -> 一个字节有多少二进制符
		 * \param debug_sync 用于指定是否在排序之前进行同步，以进行调试目的
		 */
		void Sort(const DeviceArray<KeyT>& key_in, const DeviceArray<ValueT>& value_in, cudaStream_t stream = 0, int end_bit = sizeof(KeyT) * 8, bool debug_sync = false);

		/**
		 * \brief 根据键key_in，对value_in进行排序，默认end_bit = sizeof(KeyT) * 8， debug_sync = false.
		 * 
		 * \param key_in 键值
		 * \param value_in 数值
		 * \param stream CUDA流ID
		 */
		void Sort(const DeviceArrayView<KeyT>& key_in, const DeviceArrayView<ValueT>& value_in, cudaStream_t stream = 0);

		/**
		 * \brief 根据键key_in，对value_in进行排序，debug_sync = false.
		 * 
		 * \param key_in 键值
		 * \param value_in 数值
		 * \param end_bit 考虑键的长度，以二进制的形式，sizeof(KeyT) -> 字节数，8 -> 一个字节有多少二进制符
		 * \param stream CUDA流ID
		 */
		void Sort(const DeviceArrayView<KeyT>& key_in, const DeviceArrayView<ValueT>& value_in, int end_bit, cudaStream_t stream = 0);

		/**
		 * \brief 只排序键，不同步值.
		 * 
		 * \param key_in 键值
		 * \param stream CUDA流
		 * \param end_bit 考虑键的长度，以二进制的形式，sizeof(KeyT) -> 字节数，8 -> 一个字节有多少二进制符
		 * \param debug_sync 用于指定是否在排序之前进行同步，以进行调试目的
		 */
		void Sort(const DeviceArray<KeyT>& key_in, cudaStream_t stream = 0, int end_bit = sizeof(KeyT) * 8, bool debug_sync = false);

		//Sorted value
		DeviceArray<KeyT> valid_sorted_key;			//有效排列的键  （数组首地址）
		DeviceArray<ValueT> valid_sorted_value;		//有效排列的值  （数组首地址）
	};

	/**
	 * \brief 保存前缀和的存储和结果的小结构体，应该在本地分配和使用.
	 */
	struct PrefixSum {
	private:
		//共享内存, 在执行时不进行线程（Thread）安全检查，非线程安全
		DeviceArray<unsigned char> m_temp_storage;
		DeviceArray<unsigned int> m_prefixsum_buffer;		//排列完成的前缀数组的首地址

	public:
		/**
		 * \brief 分配缓存.
		 * 
		 * \param input_size 缓存大小
		 */
		__host__ void AllocateBuffer(size_t input_size);
		/**
		 * \brief 计算前项和：InclusiveSum(a[0],a[1],a[2]) = (a[0], a[0]+a[1], a[0]+a[1]+a[2]).
		 * 
		 * \param array_in 输入数值
		 * \param stream CUDA流ID
		 * \param debug_sync 用于指定是否在求和之前进行同步，以进行调试目的。
		 */
		__host__ void InclusiveSum(const DeviceArray<unsigned>& array_in, cudaStream_t stream = 0, bool debug_sync = false);
		/**
		 * \brief 计算InclusiveSum：InclusiveSum(a[0],a[1],a[2]) = (a[0], a[0]+a[1], a[0]+a[1]+a[2]).
		 * 
		 * \param array_in 输入数值
		 * \param stream CUDA流ID
		 */
		__host__ void InclusiveSum(const DeviceArrayView<unsigned>& array_in, cudaStream_t stream = 0);
		/**
		 * \brief 计算ExclusiveSum：ExclusiveSum(a[0],a[1],a[2],a[3]) = (1，a[0], a[0]+a[1], a[0]+a[1]+a[2]).
		 * 
		 * \param array_in 输入数值
		 * \param stream CUDA流ID
		 * \param debug_sync 用于指定是否在求和之前进行同步，以进行调试目的
		 * \return 
		 */
		__host__ void ExclusiveSum(const DeviceArray<unsigned>& array_in, cudaStream_t stream = 0, bool debug_sync = false);

		//结果的指针：有效的前缀和
		DeviceArray<unsigned int> valid_prefixsum_array;		//有效前缀和  （数组的首地址）
	};
}

#if defined(__CUDACC__)		//如果由NVCC编译器编译
/**
 * 在包含 cub/cub.cuh 头文件时，通常会使用 __CUDACC__ 定义来确保只有在 CUDA 编译环境下才会包含该头文件。
 * 这是因为 cub/cub.cuh 是 CUDA Thrust 库中的一个头文件，提供了一些用于并行计算的功能和算法。
 */

#include "AlgorithmTypes.cuh"
#endif

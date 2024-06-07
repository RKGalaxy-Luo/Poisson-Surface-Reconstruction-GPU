/*****************************************************************//**
 * \file   DeviceArrayHandle.h
 * \brief  DeviceArrayHandle�Ƕ�GPU�Ļ�����ж�д���࣬����������������ӵ�������ڴ棬ֻ�Ƕ������һ�����û���ͼ��ͬʱд�˽ӿڷ�����������д�롣
 *		   DeviceSliceBufferArray�ǹ�������Ļ����������ҿ��Զ�DeviceArrayHandle������ĳһƬ�ν�����Ƭ��������
 * 
 * \author LUO
 * \date   January 8th 2024
 *********************************************************************/
#pragma once

#include <base/Logging.h>
#include <base/CommonTypes.h>
#include <base/DeviceReadWrite/DeviceArrayView.h>
#include <cuda_runtime_api.h>

namespace SparseSurfelFusion {

	// ����Ԫ����ǰ����
	template<typename T>
	class DeviceSliceBufferArray;

	/**
	 * \brief ������Ƕ�GPU������ж�ȡ��д�����.
	 */
	template<typename T>
	class DeviceArrayHandle {
	private:
		T* deviceArray;				//����GPU�ĵ�ַ�� û��const����deviceArray�ǿ��Ա���ֵ��
		size_t deviceArraySize;		//data�ĳ���(���ݸ���)
	public:
		// Ĭ�ϵĿ���/��ֵ/�ƶ�/����
		// ���캯����ָ��Ϊ�գ���СΪ0�����÷�Χ����host & device��
		__host__ __device__ DeviceArrayHandle() : deviceArray(nullptr), deviceArraySize(0) {}
		// ���캯�������������׵�ַ�����ݴ�С�����÷�Χ����host & device��
		__host__ __device__ DeviceArrayHandle(T* dev_arr, size_t size) : deviceArray(dev_arr), deviceArraySize(size) {}
		// ���캯���������׵�ַ��������ʼ��λ��(�ӵڼ�����ʼ)�����ݽ�����λ��(���ڼ�������) ���÷�Χ����host & device��
		__host__ __device__ DeviceArrayHandle(T* arr, size_t start, size_t end) {
			deviceArraySize = end - start;
			deviceArray = arr + start;
		}
		// ��ʾ���캯��������һ��DeviceArray<T>���ͣ���ֹ��ʽ����
		explicit __host__ DeviceArrayHandle(const DeviceArray<T>& arr) : deviceArray((T*)arr.ptr()), deviceArraySize(arr.size()) {}

		// �򵥽ӿ�
		// ���GPU��������
		__host__ __device__ __forceinline__ size_t Size() const { return deviceArraySize; }
		// ���GPU���⴮����ռ����byte
		__host__ __device__ __forceinline__ size_t ByteSize() const { return deviceArraySize * sizeof(T); }
		// ���������GPU�е��׵�ַ
		__host__ __device__ __forceinline__ const T* RawPtr() const { return deviceArray; }
		// ���������GPU�е��׵�ַ
		__host__ __device__ __forceinline__ T* RawPtr() { return deviceArray; }
		// �鿴GPU�е����ݣ����в鿴���ݶ�����View�鿴
		__host__ __device__ DeviceArrayView<T> ArrayView() const { return DeviceArrayView<T>(deviceArray, deviceArraySize); }

		// ��ʽ���죬����*��
		operator T* () { return deviceArray; }
		operator const T* () const { return deviceArray; }

		// ���ַ��ʷ�ʽֻ����device�в���  ���÷�Χ����device��
		__device__ __forceinline__ const T& operator[](size_t index) const { return deviceArray[index]; }
		// ���ַ��ʷ�ʽֻ����device�в���  ���÷�Χ����device��
		__device__ __forceinline__ T& operator[](size_t index) { return deviceArray[index]; }

		// ��GPU�������ص�CPU��vector��
		void SyncDeviceToHost(std::vector<T>& h_vec, cudaStream_t stream = 0) const {
			h_vec.resize(deviceArraySize);
			CHECKCUDA(cudaMemcpyAsync(h_vec.data(), deviceArray, ByteSize(), cudaMemcpyDeviceToHost, stream));
		}

		// ��CPU�����ϴ���GPU
		void SyncHostToDevice(const std::vector<T>& h_vec, cudaStream_t stream = 0) {
			FUNCTION_CHECK_EQ(h_vec.size(), deviceArraySize); //����С�Ƿ����
			CHECKCUDA(cudaMemcpyAsync(deviceArray, h_vec.data(), sizeof(T) * h_vec.size(), cudaMemcpyHostToDevice, stream));
		}

		// DeviceSliceBufferArray����Ȩ�޸�deviceArraySize
		friend class DeviceSliceBufferArray<T>;
	};

	/**
	 * \brief ��Ҫ��ȡGPU�����ĳһƬ����Ϣ��Buffer�����ݵ�ʵ������(������)��Array����������Buffer�ģ�Array��С�����Գ���Buffer.
	 */
	template<typename T>
	class DeviceSliceBufferArray {

	private:
		DeviceArrayHandle<T> Buffer;	// �����洢�������ݵ��ڴ滺��������Сһ�����ٻ������䡣Buffer ��һ��ָ���������ݵ�ָ�룬��ָ��һ���������ڴ�ռ䣬���ڴ洢�����Ԫ�ء�Buffer �Ĵ�С������Ĵ�С���������Ƿ�����ͷ��ڴ��ʵ������
		DeviceArrayHandle<T> Array;		// ��������Buffer��������С�ɱ䣬�������Դ��ڻ�����Buffer�����������ṩ��һЩ�ӿں������ڲ����͹����������ݡ�ͨ�� Array �࣬���Զ����������Ƭ������Ԫ�ء��޸�Ԫ�صȲ�����

	public:
		// Ĭ����ʽ����/��ֵ/�ƶ�/ɾ��
		__host__ __device__ DeviceSliceBufferArray() : Buffer(), Array() {}
		/**
		 * \brief ���������׵�ַ�ͻ�����Buffer������������DeviceSliceBufferArray����.
		 * 
		 * \param buffer �����׵�ַ
		 * \param capacity ���������������Buffer������
		 */
		__host__ __device__ DeviceSliceBufferArray(T* buffer, size_t capacity) : Buffer(buffer, capacity), Array(buffer, 0) {}

		// 
		/**
		 * \brief ���������׵�ַ�������DeviceSliceBufferArray������Buffer�����������������Buffer������Array�Ĵ�С������DeviceSliceBufferArray����(��host�Ĺ��캯�������С).
		 * 
		 * \param buffer �����׵�ַ
		 * \param capacity �����DeviceSliceBufferArray������Buffer������
		 * \param array_size ���������Buffer������Array�Ĵ�С
		 */
		__host__ DeviceSliceBufferArray(T* buffer, size_t capacity, size_t array_size) : Buffer(buffer, capacity), Array(buffer, array_size) {
			// ��������С
			if (array_size > capacity) {
				LOGGING(FATAL) << "�ṩ�Ļ��治����";
			}
		}

		/**
		 * \brief ����GPU����DeviceArray<T>���ͣ���GPU���ݴ���Buffer���������׵�ַ����Array(Array��������Ϊ0).
		 * 
		 * \param arr ����GPU����DeviceArray<T>����
		 */
		__host__ DeviceSliceBufferArray(const SparseSurfelFusion::DeviceArray<T>& arr) : Buffer(arr), Array((T*)arr.ptr(), 0) {}

		// ���������С
		/**
		 * \brief ����size����Array�Ĵ�С��size���������Array��Buffer��С�����򱨴�.
		 * 
		 * \param ��Ҫ������Array�Ĵ�С
		 */
		__host__ void ResizeArrayOrException(size_t size) {
			if (size > Buffer.Size()) {
				// �׳��쳣
				LOGGING(FATAL) << "�ṩ�Ļ��治����";
			}

			// ��ȫ����������Ĵ�С
			Array.deviceArraySize = size;
		}

		// �ӿڣ������ٷ���GPU�鿴
		/**
		 * \brief ��õ�ǰDeviceSliceBufferArray<T>����Buffer������(Buffer����һ�㹹��֮���ȷ����).
		 * 
		 * \return ��ǰDeviceSliceBufferArray<T>����Buffer������
		 */
		__host__ __forceinline__ size_t Capacity() const { return Buffer.Size(); }
		/**
		 * \brief ��õ�ǰDeviceSliceBufferArray<T>����Buffer������(Buffer����һ�㹹��֮���ȷ����).
		 * 
		 * \return ��ǰDeviceSliceBufferArray<T>����Buffer������
		 */
		__host__ __forceinline__ size_t BufferSize() const { return Buffer.Size(); }
		/**
		 * \brief ��õ�ǰDeviceSliceBufferArray<T>����Array������(Array�������ɱ䣬�������Գ���Buffer����).
		 * 
		 * \return ��ǰDeviceSliceBufferArray<T>����Array������
		 */
		__host__ __forceinline__ size_t ArraySize() const { return Array.Size(); }
		/**
		 * \brief ��û���������Buffer���׵�ַ.
		 * 
		 * \return ����������Buffer���׵�ַ
		 */
		__host__ __forceinline__ const T* Ptr() const { return Buffer.RawPtr(); }
		/**
		 * \brief ��û���������Buffer���׵�ַ.
		 *
		 * \return ����������Buffer���׵�ַ
		 */
		__host__ __forceinline__ T* Ptr() { return Buffer.RawPtr(); }
		/**
		 * \brief �������Array�����Զ�Buffer���ж�д�͹���.
		 * 
		 * \return ����Array
		 */
		__host__ __forceinline__ DeviceArrayHandle<T> ArrayHandle() const { return Array; }
		/**
		 * \brief �������Array��ֻ������(������DeviceArrayView<T>����).
		 * 
		 * \return ����Array��ֻ������
		 */
		__host__ __forceinline__ DeviceArrayView<T> ArrayReadOnly() const { return DeviceArrayView<T>(Ptr(), ArraySize()); }
		/**
		 * \brief �������Array��ֻ������(������DeviceArrayView<T>����).
		 *
		 * \return ����Array��ֻ������
		 */
		__host__ __forceinline__ DeviceArrayView<T> ArrayView() const { return DeviceArrayView<T>(Ptr(), ArraySize()); }

		/**
		 * \brief ������������Ƭ.
		 * 
		 * \param start ��ʼ��Ƭ��λ��
		 * \param end ������Ƭ��λ��
		 * \return ������Ƭ(��DeviceSliceBufferArray<T>���ͷ���)
		 */
		__host__ DeviceSliceBufferArray<T> BufferArraySlice(size_t start, size_t end) const {
			if (start > end || end > Buffer.Size()) {
				LOGGING(FATAL) << "��Ч������Ƭ�Σ���ȷ��start < end ������ end < Buffer.Size()��";
			}

			// ����������Ƭ
			return DeviceSliceBufferArray<T>((T*)Buffer.RawPtr() + start, end - start);
		}

	};
}

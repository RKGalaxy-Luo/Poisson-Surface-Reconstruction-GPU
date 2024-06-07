/*****************************************************************//**
 * \file   SynchronizeArray.h
 * \brief  ����ͬ�����ܵ����顣ע�⣬��������ݲ��ܱ�֤ͬ������Ҫ��ʽͬ���� ���ǣ��������к��豸���е����д�С������ͬ�ġ�
 * 
 * \author LUO
 * \date   February 4th 2024
 *********************************************************************/
#pragma once

#include <base/CommonUtils.h>
#include <base/CommonTypes.h>
#include "DeviceArrayView.h"
#include "DeviceBufferArray.h"

namespace SparseSurfelFusion {

	/**
	 * \brief ����ͬ�����ܵ����顣ע�⣬��������ݲ��ܱ�֤ͬ������Ҫ��ʽͬ�������ǣ��������к��豸���е����д�С������ͬ�ġ�
	 */
	template<typename T>
	class SynchronizeArray {
	private:
		std::vector<T> hostArray;			//CPU�ϵ�����
		DeviceBufferArray<T> deviceArray;	//GPU�ϵ�����

	public:
		explicit SynchronizeArray() : hostArray(), deviceArray() {}
		/**
		 * \brief ����SynchronizeArray���ͣ���Host����vector�����ڴ棬��GPU����DeviceBufferArray�����ڴ棬��������.
		 * 
		 * \param capacity Host��Device���ٵ�����
		 */
		explicit SynchronizeArray(size_t capacity) {
			AllocateBuffer(capacity);
		}
		~SynchronizeArray() = default;
		NO_COPY_ASSIGN_MOVE(SynchronizeArray);

		/**
		 * \brief ����GPU�����ڴ�������С.
		 * 
		 * \return ����GPU�����ڴ�������С
		 */
		size_t Capacity() const { return deviceArray.Capacity(); }
		/**
		 * \brief ����GPU�ڴ�洢�����ݴ�С(���ݴ�С �� ���ٵ��ڴ�����).
		 * 
		 * \return GPU�ڴ�洢�����ݴ�С
		 */
		size_t DeviceArraySize() const { return deviceArray.ArraySize(); }
		/**
		 * \brief ������Host�ϴ洢�����ݴ�С.
		 * 
		 * \return ��Host�ϴ洢�����ݴ�С
		 */
		size_t HostArraySize() const { return hostArray.size(); }
		/**
		 * \brief ���GPU�ϵ����ݣ�ֻ����ʽ.
		 * 
		 * \return GPU�ϵ�����(������DeviceArrayView<T>����)
		 */
		DeviceArrayView<T> DeviceArrayReadOnly() const { return DeviceArrayView<T>(deviceArray.Array()); }
		/**
		 * \brief ���GPU�ϵ����ݣ��ɶ�д��ʽ.
		 * 
		 * \return GPU�ϵ�����(������SparseSurfelFusion::DeviceArray<T>��ʽ)
		 */
		SparseSurfelFusion::DeviceArray<T> DeviceArray() const { return deviceArray.Array(); }
		/**
		 * \brief ���GPU�ϵ����ݣ��ɶ�д��ʽ.
		 * 
		 * \return GPU�ϵ�����(������DeviceArrayHandle<T>��ʽ)
		 */
		DeviceArrayHandle<T> DeviceArrayReadWrite() { return deviceArray.ArrayHandle(); }

		/**
		 * \brief ���Host�ϵ����ݣ��ǿ�����ʽ�����ô���.
		 * 
		 * \return Host�ϵ�����
		 */
		std::vector<T>& HostArray() { return hostArray; }
		/**
		 * \brief ���Host�ϵ����ݣ��ǿ�����ʽ�����ô���.
		 * 
		 * \return Host�ϵ�����
		 */
		const std::vector<T>& HostArray() const { return hostArray; }

		/**
		 * \brief ����GPU��ԭʼ�ĵ�ַ.
		 * 
		 * \return GPU��ԭʼ�ĵ�ַ(DeviceBufferArray).
		 */
		const T* DevicePtr() const { return deviceArray.Ptr(); }
		T* DevicePtr() { return deviceArray.Ptr(); }

		/**
		 * \brief ��Host����vector�����ڴ棬��GPU����DeviceBufferArray�����ڴ棬��������.
		 * 
		 * \param capacity ������С
		 */
		void AllocateBuffer(size_t capacity) {
			hostArray.reserve(capacity);
			deviceArray.AllocateBuffer(capacity);
		}

		/**
		 * \brief ���µ���host��device�ڴ��С.
		 * 
		 * \param size ��Ҫ������device��host�ڴ��С
		 * \param allocate �����С�����Ƿ����·����С
		 * \return ����device��host�ڴ��С�Ƿ�ɹ�
		 */
		bool ResizeArray(size_t size, bool allocate = false) {
			if (deviceArray.ResizeArray(size, allocate) == true) {
				hostArray.resize(size);
				return true;
			}

			//�豸�����޷��ɹ�������С
			//�������豸��Сһ��
			return false;
		}
		/**
		 * \brief ����host��Array�Ĵ�С�Լ�GPU��Array�Ĵ�С.
		 * 
		 * \param size ���Ԥ�����BufferС��size����ô�򱨴�������GPU�Ϸ���һ����СΪsize���ڴ�ռ�
		 */
		void ResizeArrayOrException(size_t size) {
			deviceArray.ResizeArrayOrException(size);
			hostArray.resize(size);
		}

		//����������к��豸���е�Array, ���ǲ����漰�ѷ���Ļ���
		void ClearArray() {
			ResizeArray(0);
		}

		
		/**
		 * \brief ��GPU��������CPUͬ��������ֱ�ӽ�hostArray.data()������deviceArray.Ptr().
		 * 
		 * \param stream cuda��ID
		 */
		void SynchronizeToDevice(cudaStream_t stream = 0) {
			//����GPU�������С
			deviceArray.ResizeArrayOrException(hostArray.size());

			//ʵ�ʵ�ͬ��
			CHECKCUDA(cudaMemcpyAsync(deviceArray.Ptr(), hostArray.data(), sizeof(T) * hostArray.size(), cudaMemcpyHostToDevice, stream));
		}

		/**
		 * \brief ��CPU��������GPUͬ�������ｫdeviceArray.Ptr()���ݿ�����hostArray.data()��.
		 * 
		 * \param stream cuda��ID
		 * \param sync �Ƿ����cuda��ͬ��(Ĭ�Ͻ���)
		 */
		void SynchronizeToHost(cudaStream_t stream = 0, bool sync = true) {
			//�����������д�С
			hostArray.resize(deviceArray.ArraySize());

			//ʵ�ʵ�ͬ��
			CHECKCUDA(cudaMemcpyAsync(hostArray.data(), deviceArray.Ptr(), sizeof(T) * hostArray.size(), cudaMemcpyDeviceToHost, stream));

			if (sync) {
				//��������ʹ��֮ǰ�����������ͬ��
				//����ص����ܻ��Ƴ�
				CHECKCUDA(cudaStreamSynchronize(stream));
			}
		}
	};
}

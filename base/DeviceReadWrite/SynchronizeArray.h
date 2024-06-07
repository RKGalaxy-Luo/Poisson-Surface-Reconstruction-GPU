/*****************************************************************//**
 * \file   SynchronizeArray.h
 * \brief  具有同步功能的数组。注意，数组的内容不能保证同步，需要显式同步。 但是，主机阵列和设备阵列的阵列大小总是相同的。
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
	 * \brief 具有同步功能的数组。注意，数组的内容不能保证同步，需要显式同步。但是，主机阵列和设备阵列的阵列大小总是相同的。
	 */
	template<typename T>
	class SynchronizeArray {
	private:
		std::vector<T> hostArray;			//CPU上的数组
		DeviceBufferArray<T> deviceArray;	//GPU上的数组

	public:
		explicit SynchronizeArray() : hostArray(), deviceArray() {}
		/**
		 * \brief 构造SynchronizeArray类型，给Host分配vector类型内存，给GPU分配DeviceBufferArray类型内存，开辟容量.
		 * 
		 * \param capacity Host和Device开辟的容量
		 */
		explicit SynchronizeArray(size_t capacity) {
			AllocateBuffer(capacity);
		}
		~SynchronizeArray() = default;
		NO_COPY_ASSIGN_MOVE(SynchronizeArray);

		/**
		 * \brief 返回GPU开辟内存容量大小.
		 * 
		 * \return 返回GPU开辟内存容量大小
		 */
		size_t Capacity() const { return deviceArray.Capacity(); }
		/**
		 * \brief 返回GPU内存存储的数据大小(数据大小 ≤ 开辟的内存容量).
		 * 
		 * \return GPU内存存储的数据大小
		 */
		size_t DeviceArraySize() const { return deviceArray.ArraySize(); }
		/**
		 * \brief 返回在Host上存储的数据大小.
		 * 
		 * \return 在Host上存储的数据大小
		 */
		size_t HostArraySize() const { return hostArray.size(); }
		/**
		 * \brief 获得GPU上的数据，只读形式.
		 * 
		 * \return GPU上的数据(传出以DeviceArrayView<T>类型)
		 */
		DeviceArrayView<T> DeviceArrayReadOnly() const { return DeviceArrayView<T>(deviceArray.Array()); }
		/**
		 * \brief 获得GPU上的数据，可读写形式.
		 * 
		 * \return GPU上的数据(传出以SparseSurfelFusion::DeviceArray<T>形式)
		 */
		SparseSurfelFusion::DeviceArray<T> DeviceArray() const { return deviceArray.Array(); }
		/**
		 * \brief 获得GPU上的数据，可读写形式.
		 * 
		 * \return GPU上的数据(传出以DeviceArrayHandle<T>形式)
		 */
		DeviceArrayHandle<T> DeviceArrayReadWrite() { return deviceArray.ArrayHandle(); }

		/**
		 * \brief 获得Host上的数据，非拷贝形式，引用传递.
		 * 
		 * \return Host上的数据
		 */
		std::vector<T>& HostArray() { return hostArray; }
		/**
		 * \brief 获得Host上的数据，非拷贝形式，引用传递.
		 * 
		 * \return Host上的数据
		 */
		const std::vector<T>& HostArray() const { return hostArray; }

		/**
		 * \brief 访问GPU上原始的地址.
		 * 
		 * \return GPU上原始的地址(DeviceBufferArray).
		 */
		const T* DevicePtr() const { return deviceArray.Ptr(); }
		T* DevicePtr() { return deviceArray.Ptr(); }

		/**
		 * \brief 给Host分配vector类型内存，给GPU分配DeviceBufferArray类型内存，开辟容量.
		 * 
		 * \param capacity 容量大小
		 */
		void AllocateBuffer(size_t capacity) {
			hostArray.reserve(capacity);
			deviceArray.AllocateBuffer(capacity);
		}

		/**
		 * \brief 重新调整host和device内存大小.
		 * 
		 * \param size 需要调整的device和host内存大小
		 * \param allocate 如果大小不够是否重新分配大小
		 * \return 调整device和host内存大小是否成功
		 */
		bool ResizeArray(size_t size, bool allocate = false) {
			if (deviceArray.ResizeArray(size, allocate) == true) {
				hostArray.resize(size);
				return true;
			}

			//设备阵列无法成功调整大小
			//主机和设备大小一致
			return false;
		}
		/**
		 * \brief 调整host上Array的大小以及GPU上Array的大小.
		 * 
		 * \param size 如果预分配的Buffer小于size，那么则报错，否则在GPU上分配一个大小为size的内存空间
		 */
		void ResizeArrayOrException(size_t size) {
			deviceArray.ResizeArrayOrException(size);
			hostArray.resize(size);
		}

		//清除主机阵列和设备阵列的Array, 但是不会涉及已分配的缓存
		void ClearArray() {
			ResizeArray(0);
		}

		
		/**
		 * \brief 将GPU的数据与CPU同步，这里直接将hostArray.data()拷贝给deviceArray.Ptr().
		 * 
		 * \param stream cuda流ID
		 */
		void SynchronizeToDevice(cudaStream_t stream = 0) {
			//调整GPU上数组大小
			deviceArray.ResizeArrayOrException(hostArray.size());

			//实际的同步
			CHECKCUDA(cudaMemcpyAsync(deviceArray.Ptr(), hostArray.data(), sizeof(T) * hostArray.size(), cudaMemcpyHostToDevice, stream));
		}

		/**
		 * \brief 将CPU的数据与GPU同步，这里将deviceArray.Ptr()数据拷贝到hostArray.data()中.
		 * 
		 * \param stream cuda流ID
		 * \param sync 是否进行cuda流同步(默认进行)
		 */
		void SynchronizeToHost(cudaStream_t stream = 0, bool sync = true) {
			//调整主机阵列大小
			hostArray.resize(deviceArray.ArraySize());

			//实际的同步
			CHECKCUDA(cudaMemcpyAsync(hostArray.data(), deviceArray.Ptr(), sizeof(T) * hostArray.size(), cudaMemcpyDeviceToHost, stream));

			if (sync) {
				//在主机上使用之前，必须调用流同步
				//这个回调可能会推迟
				CHECKCUDA(cudaStreamSynchronize(stream));
			}
		}
	};
}

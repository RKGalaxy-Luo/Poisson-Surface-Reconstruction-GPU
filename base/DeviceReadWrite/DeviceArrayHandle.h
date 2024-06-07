/*****************************************************************//**
 * \file   DeviceArrayHandle.h
 * \brief  DeviceArrayHandle是对GPU的缓存进行读写的类，属于轻量级，并不拥有数组内存，只是对数组的一个引用或视图，同时写了接口方便对数组进行写入。
 *		   DeviceSliceBufferArray是管理数组的缓冲区，并且可以对DeviceArrayHandle数组中某一片段进行切片，并管理。
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

	// 对友元的向前声明
	template<typename T>
	class DeviceSliceBufferArray;

	/**
	 * \brief 这个类是对GPU缓存进行读取和写入的类.
	 */
	template<typename T>
	class DeviceArrayHandle {
	private:
		T* deviceArray;				//存入GPU的地址， 没有const，即deviceArray是可以被赋值的
		size_t deviceArraySize;		//data的长度(数据个数)
	public:
		// 默认的拷贝/赋值/移动/析构
		// 构造函数：指针为空，大小为0。适用范围：“host & device”
		__host__ __device__ DeviceArrayHandle() : deviceArray(nullptr), deviceArraySize(0) {}
		// 构造函数：传入数据首地址，数据大小。适用范围：“host & device”
		__host__ __device__ DeviceArrayHandle(T* dev_arr, size_t size) : deviceArray(dev_arr), deviceArraySize(size) {}
		// 构造函数：传入首地址，数据起始的位置(从第几个开始)，数据结束的位置(到第几个结束) 适用范围：“host & device”
		__host__ __device__ DeviceArrayHandle(T* arr, size_t start, size_t end) {
			deviceArraySize = end - start;
			deviceArray = arr + start;
		}
		// 显示构造函数：传给一个DeviceArray<T>类型，防止隐式调用
		explicit __host__ DeviceArrayHandle(const DeviceArray<T>& arr) : deviceArray((T*)arr.ptr()), deviceArraySize(arr.size()) {}

		// 简单接口
		// 获得GPU数据数量
		__host__ __device__ __forceinline__ size_t Size() const { return deviceArraySize; }
		// 获得GPU中这串数据占多少byte
		__host__ __device__ __forceinline__ size_t ByteSize() const { return deviceArraySize * sizeof(T); }
		// 获得数据在GPU中的首地址
		__host__ __device__ __forceinline__ const T* RawPtr() const { return deviceArray; }
		// 获得数据在GPU中的首地址
		__host__ __device__ __forceinline__ T* RawPtr() { return deviceArray; }
		// 查看GPU中的数据，所有查看数据都是用View查看
		__host__ __device__ DeviceArrayView<T> ArrayView() const { return DeviceArrayView<T>(deviceArray, deviceArraySize); }

		// 隐式构造，重载*号
		operator T* () { return deviceArray; }
		operator const T* () const { return deviceArray; }

		// 这种访问方式只能在device中操作  适用范围：“device”
		__device__ __forceinline__ const T& operator[](size_t index) const { return deviceArray[index]; }
		// 这种访问方式只能在device中操作  适用范围：“device”
		__device__ __forceinline__ T& operator[](size_t index) { return deviceArray[index]; }

		// 将GPU数据下载到CPU的vector中
		void SyncDeviceToHost(std::vector<T>& h_vec, cudaStream_t stream = 0) const {
			h_vec.resize(deviceArraySize);
			CHECKCUDA(cudaMemcpyAsync(h_vec.data(), deviceArray, ByteSize(), cudaMemcpyDeviceToHost, stream));
		}

		// 将CPU数据上传到GPU
		void SyncHostToDevice(const std::vector<T>& h_vec, cudaStream_t stream = 0) {
			FUNCTION_CHECK_EQ(h_vec.size(), deviceArraySize); //检查大小是否相等
			CHECKCUDA(cudaMemcpyAsync(deviceArray, h_vec.data(), sizeof(T) * h_vec.size(), cudaMemcpyHostToDevice, stream));
		}

		// DeviceSliceBufferArray被授权修改deviceArraySize
		friend class DeviceSliceBufferArray<T>;
	};

	/**
	 * \brief 需要读取GPU数组的某一片段信息，Buffer是数据的实际载体(缓冲区)，Array是用来管理Buffer的，Array大小不可以超过Buffer.
	 */
	template<typename T>
	class DeviceSliceBufferArray {

	private:
		DeviceArrayHandle<T> Buffer;	// 用来存储数组数据的内存缓冲区，大小一旦开辟基本不变。Buffer 是一个指向数组数据的指针，它指向一块连续的内存空间，用于存储数组的元素。Buffer 的大小由数组的大小决定，它是分配和释放内存的实际载体
		DeviceArrayHandle<T> Array;		// 用来管理Buffer，容量大小可变，但不可以大于缓冲区Buffer的容量。它提供了一些接口函数用于操作和管理数组数据。通过 Array 类，可以对数组进行切片、访问元素、修改元素等操作。

	public:
		// 默认隐式拷贝/赋值/移动/删除
		__host__ __device__ DeviceSliceBufferArray() : Buffer(), Array() {}
		/**
		 * \brief 传入数据首地址和缓冲区Buffer的容量，构造DeviceSliceBufferArray类型.
		 * 
		 * \param buffer 数据首地址
		 * \param capacity 构造这个数据类型Buffer的容量
		 */
		__host__ __device__ DeviceSliceBufferArray(T* buffer, size_t capacity) : Buffer(buffer, capacity), Array(buffer, 0) {}

		// 
		/**
		 * \brief 传入数据首地址，分配给DeviceSliceBufferArray缓冲区Buffer的容量，分配给管理Buffer的数组Array的大小，构造DeviceSliceBufferArray类型(在host的构造函数会检查大小).
		 * 
		 * \param buffer 数据首地址
		 * \param capacity 分配给DeviceSliceBufferArray缓冲区Buffer的容量
		 * \param array_size 分配给管理Buffer的数组Array的大小
		 */
		__host__ DeviceSliceBufferArray(T* buffer, size_t capacity, size_t array_size) : Buffer(buffer, capacity), Array(buffer, array_size) {
			// 检查数组大小
			if (array_size > capacity) {
				LOGGING(FATAL) << "提供的缓存不够！";
			}
		}

		/**
		 * \brief 传入GPU数组DeviceArray<T>类型，将GPU数据存入Buffer，将数据首地址存入Array(Array分配容量为0).
		 * 
		 * \param arr 传入GPU数组DeviceArray<T>类型
		 */
		__host__ DeviceSliceBufferArray(const SparseSurfelFusion::DeviceArray<T>& arr) : Buffer(arr), Array((T*)arr.ptr(), 0) {}

		// 调整数组大小
		/**
		 * \brief 根据size调整Array的大小，size如果调整的Array比Buffer大小还大，则报错.
		 * 
		 * \param 需要调整的Array的大小
		 */
		__host__ void ResizeArrayOrException(size_t size) {
			if (size > Buffer.Size()) {
				// 抛出异常
				LOGGING(FATAL) << "提供的缓存不够！";
			}

			// 安全地重置数组的大小
			Array.deviceArraySize = size;
		}

		// 接口：无需再访问GPU查看
		/**
		 * \brief 获得当前DeviceSliceBufferArray<T>类型Buffer的容量(Buffer容量一般构造之后就确定了).
		 * 
		 * \return 当前DeviceSliceBufferArray<T>类型Buffer的容量
		 */
		__host__ __forceinline__ size_t Capacity() const { return Buffer.Size(); }
		/**
		 * \brief 获得当前DeviceSliceBufferArray<T>类型Buffer的容量(Buffer容量一般构造之后就确定了).
		 * 
		 * \return 当前DeviceSliceBufferArray<T>类型Buffer的容量
		 */
		__host__ __forceinline__ size_t BufferSize() const { return Buffer.Size(); }
		/**
		 * \brief 获得当前DeviceSliceBufferArray<T>类型Array的容量(Array的容量可变，但不可以超过Buffer容量).
		 * 
		 * \return 当前DeviceSliceBufferArray<T>类型Array的容量
		 */
		__host__ __forceinline__ size_t ArraySize() const { return Array.Size(); }
		/**
		 * \brief 获得缓冲区数据Buffer的首地址.
		 * 
		 * \return 缓冲区数据Buffer的首地址
		 */
		__host__ __forceinline__ const T* Ptr() const { return Buffer.RawPtr(); }
		/**
		 * \brief 获得缓冲区数据Buffer的首地址.
		 *
		 * \return 缓冲区数据Buffer的首地址
		 */
		__host__ __forceinline__ T* Ptr() { return Buffer.RawPtr(); }
		/**
		 * \brief 获得数组Array，可以对Buffer进行读写和管理.
		 * 
		 * \return 数组Array
		 */
		__host__ __forceinline__ DeviceArrayHandle<T> ArrayHandle() const { return Array; }
		/**
		 * \brief 获得数组Array的只读类型(传出以DeviceArrayView<T>类型).
		 * 
		 * \return 数组Array的只读类型
		 */
		__host__ __forceinline__ DeviceArrayView<T> ArrayReadOnly() const { return DeviceArrayView<T>(Ptr(), ArraySize()); }
		/**
		 * \brief 获得数组Array的只读类型(传出以DeviceArrayView<T>类型).
		 *
		 * \return 数组Array的只读类型
		 */
		__host__ __forceinline__ DeviceArrayView<T> ArrayView() const { return DeviceArrayView<T>(Ptr(), ArraySize()); }

		/**
		 * \brief 对数组制作切片.
		 * 
		 * \param start 开始切片的位置
		 * \param end 结束切片的位置
		 * \return 数组切片(以DeviceSliceBufferArray<T>类型返回)
		 */
		__host__ DeviceSliceBufferArray<T> BufferArraySlice(size_t start, size_t end) const {
			if (start > end || end > Buffer.Size()) {
				LOGGING(FATAL) << "无效的数组片段，请确认start < end 或者是 end < Buffer.Size()！";
			}

			// 返回内在切片
			return DeviceSliceBufferArray<T>((T*)Buffer.RawPtr() + start, end - start);
		}

	};
}

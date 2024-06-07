/*****************************************************************//**
 * \file   DeviceArrayView.h
 * \brief  对GPU的缓存内容只读的类
 * 
 * \author LUO
 * \date   January 2024
 *********************************************************************/
#pragma once

#include <base/CommonTypes.h>

namespace SparseSurfelFusion {
	
	// 这个类是一个只读类，读取Device中的内容， 不对GPU做分配内存操作
	template<typename T>
	class DeviceArrayView {
	private:
		const T* deviceArray;	// 在GPU中数据的首地址，const就确定了是一个常量不会被赋值
		size_t deviceArraySize;	// GPU中数据的数量
	public:
		// 默认的拷贝/赋值/移动/析构
		// 构造函数：指针为空，大小为0。适用范围：“host & device”
		__host__ __device__ DeviceArrayView() : deviceArray(nullptr), deviceArraySize(0) {}
		// 构造函数：传入数据首地址，数据大小。适用范围：“host & device”
		__host__ __device__ DeviceArrayView(const T* arr, size_t size) : deviceArray(arr), deviceArraySize(size) {}
		// 构造函数：传入首地址，数据起始的位置(从第几个开始)，数据结束的位置(到第几个结束) 适用范围：“host & device”
		__host__ __device__ DeviceArrayView(const T* arr, size_t start, size_t end) {
			deviceArraySize = end - start;
			deviceArray = arr + start;
		}
		// 显示构造函数：传给一个DeviceArray<T>类型，防止隐式调用
		explicit __host__ DeviceArrayView(const DeviceArray<T>& arr) : deviceArray(arr.ptr()), deviceArraySize(arr.size()) {}

		// 分配操作，重载“=”使得DeviceArray能给DeviceArrayView赋值  适用范围：“host”
		__host__ DeviceArrayView<T>& operator=(const DeviceArray<T>& arr) {
			deviceArray = arr.ptr();
			deviceArraySize = arr.size();
			return *this;
		}

		// 简单接口
		// 获得GPU数据数量
		__host__ __device__ size_t Size() const { return deviceArraySize; }
		// 获得GPU中这串数据占多少byte
		__host__ __device__ size_t ByteSize() const { return deviceArraySize * sizeof(T); }
		// 获得数据在GPU中的首地址
		__host__ __device__ const T* RawPtr() const { return deviceArray; }
		// 获得数据在GPU中的首地址
		__host__ __device__ operator const T* () const { return deviceArray; }

		// 这种访问方式只能在device中操作  适用范围：“device”
		__device__ const T& operator[](size_t index) const { return deviceArray[index]; }

		// 将GPU数据拷贝到CPU，以便于后续Debug
		__host__ void Download(std::vector<T>& h_vec) const {
			h_vec.resize(Size());
			CHECKCUDA(cudaMemcpy(h_vec.data(), deviceArray, Size() * sizeof(T), cudaMemcpyDeviceToHost));
		}
	};

	// GPU中2D数组的查看，是一个只读类
	template<typename T>
	class DeviceArrayView2D {
	private:
		unsigned short rows, cols;	// 2D数组的行，列
		unsigned int byte_step;		// 步长都是以字节byte为单位的，一行有多少个字节
		const T* ptr;				// 数组的首地址

	public:
		// 构造函数，初始化：长宽为0
		__host__ __device__ DeviceArrayView2D() : rows(0), cols(0), byte_step(0), ptr(nullptr) {}
		// 构造函数：用DeviceArray2D类来构造DeviceArrayView2D，以供查看GPU中的数据
		__host__ DeviceArrayView2D(const DeviceArray2D<T>& array2D)
			: rows(array2D.rows()), cols(array2D.cols()),
			byte_step(array2D.step()), ptr(array2D.ptr())
		{}

		// 接口
		// 二维数据的行数
		__host__ __device__ __forceinline__ unsigned short Rows() const { return rows; }
		// 二维数据的列数
		__host__ __device__ __forceinline__ unsigned short Cols() const { return cols; }
		// 二维数据中一行有多少个字节 
		__host__ __device__ __forceinline__ unsigned ByteStep() const { return byte_step; }
		// 二维数据的首地址
		__host__ __device__ __forceinline__ const T* RawPtr() const { return ptr; }
		// 二维数据中第row行的首地址
		__host__ __device__ __forceinline__ const T* RawPtr(int row) const {
			return ((const T*)((const char*)(ptr)+row * byte_step));
		}
		// 二维数据中第row行,第col列的地址
		__host__ __device__ __forceinline__ const T& operator()(int row, int col) const {
			return RawPtr(row)[col];
		}
	};
}
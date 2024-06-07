/*****************************************************************//**
 * \file   DeviceArrayView.h
 * \brief  ��GPU�Ļ�������ֻ������
 * 
 * \author LUO
 * \date   January 2024
 *********************************************************************/
#pragma once

#include <base/CommonTypes.h>

namespace SparseSurfelFusion {
	
	// �������һ��ֻ���࣬��ȡDevice�е����ݣ� ����GPU�������ڴ����
	template<typename T>
	class DeviceArrayView {
	private:
		const T* deviceArray;	// ��GPU�����ݵ��׵�ַ��const��ȷ������һ���������ᱻ��ֵ
		size_t deviceArraySize;	// GPU�����ݵ�����
	public:
		// Ĭ�ϵĿ���/��ֵ/�ƶ�/����
		// ���캯����ָ��Ϊ�գ���СΪ0�����÷�Χ����host & device��
		__host__ __device__ DeviceArrayView() : deviceArray(nullptr), deviceArraySize(0) {}
		// ���캯�������������׵�ַ�����ݴ�С�����÷�Χ����host & device��
		__host__ __device__ DeviceArrayView(const T* arr, size_t size) : deviceArray(arr), deviceArraySize(size) {}
		// ���캯���������׵�ַ��������ʼ��λ��(�ӵڼ�����ʼ)�����ݽ�����λ��(���ڼ�������) ���÷�Χ����host & device��
		__host__ __device__ DeviceArrayView(const T* arr, size_t start, size_t end) {
			deviceArraySize = end - start;
			deviceArray = arr + start;
		}
		// ��ʾ���캯��������һ��DeviceArray<T>���ͣ���ֹ��ʽ����
		explicit __host__ DeviceArrayView(const DeviceArray<T>& arr) : deviceArray(arr.ptr()), deviceArraySize(arr.size()) {}

		// ������������ء�=��ʹ��DeviceArray�ܸ�DeviceArrayView��ֵ  ���÷�Χ����host��
		__host__ DeviceArrayView<T>& operator=(const DeviceArray<T>& arr) {
			deviceArray = arr.ptr();
			deviceArraySize = arr.size();
			return *this;
		}

		// �򵥽ӿ�
		// ���GPU��������
		__host__ __device__ size_t Size() const { return deviceArraySize; }
		// ���GPU���⴮����ռ����byte
		__host__ __device__ size_t ByteSize() const { return deviceArraySize * sizeof(T); }
		// ���������GPU�е��׵�ַ
		__host__ __device__ const T* RawPtr() const { return deviceArray; }
		// ���������GPU�е��׵�ַ
		__host__ __device__ operator const T* () const { return deviceArray; }

		// ���ַ��ʷ�ʽֻ����device�в���  ���÷�Χ����device��
		__device__ const T& operator[](size_t index) const { return deviceArray[index]; }

		// ��GPU���ݿ�����CPU���Ա��ں���Debug
		__host__ void Download(std::vector<T>& h_vec) const {
			h_vec.resize(Size());
			CHECKCUDA(cudaMemcpy(h_vec.data(), deviceArray, Size() * sizeof(T), cudaMemcpyDeviceToHost));
		}
	};

	// GPU��2D����Ĳ鿴����һ��ֻ����
	template<typename T>
	class DeviceArrayView2D {
	private:
		unsigned short rows, cols;	// 2D������У���
		unsigned int byte_step;		// �����������ֽ�byteΪ��λ�ģ�һ���ж��ٸ��ֽ�
		const T* ptr;				// ������׵�ַ

	public:
		// ���캯������ʼ��������Ϊ0
		__host__ __device__ DeviceArrayView2D() : rows(0), cols(0), byte_step(0), ptr(nullptr) {}
		// ���캯������DeviceArray2D��������DeviceArrayView2D���Թ��鿴GPU�е�����
		__host__ DeviceArrayView2D(const DeviceArray2D<T>& array2D)
			: rows(array2D.rows()), cols(array2D.cols()),
			byte_step(array2D.step()), ptr(array2D.ptr())
		{}

		// �ӿ�
		// ��ά���ݵ�����
		__host__ __device__ __forceinline__ unsigned short Rows() const { return rows; }
		// ��ά���ݵ�����
		__host__ __device__ __forceinline__ unsigned short Cols() const { return cols; }
		// ��ά������һ���ж��ٸ��ֽ� 
		__host__ __device__ __forceinline__ unsigned ByteStep() const { return byte_step; }
		// ��ά���ݵ��׵�ַ
		__host__ __device__ __forceinline__ const T* RawPtr() const { return ptr; }
		// ��ά�����е�row�е��׵�ַ
		__host__ __device__ __forceinline__ const T* RawPtr(int row) const {
			return ((const T*)((const char*)(ptr)+row * byte_step));
		}
		// ��ά�����е�row��,��col�еĵ�ַ
		__host__ __device__ __forceinline__ const T& operator()(int row, int col) const {
			return RawPtr(row)[col];
		}
	};
}
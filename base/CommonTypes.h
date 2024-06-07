#pragma once
#include "GlobalConfigs.h"
#include <base/DeviceAPI/convenience.cuh>
#include <base/DeviceAPI/device_array.hpp>
#include <base/DeviceAPI/kernel_containers.hpp>
#include <base/DeviceAPI/safe_call.hpp>
//CUDA����
#include <vector_functions.h>
#include <vector>

namespace SparseSurfelFusion {
	/* host��device���ʵ� gpu ����
	*/
	//��DeviceArray����DeviceArrayPCL
	template<typename T>
	using DeviceArray = DeviceArrayPCL<T>;
	//��DeviceArray2D����DeviceArray2DPCL
	template<typename T>
	using DeviceArray2D = DeviceArray2DPCL<T>;

	namespace device {
		//device���� gpu ����������
		//����Device��ַ��ָ��
		template<typename T>
		using DevicePtr = DevPtr<T>;

		//����Device��ַ�Լ����ݴ�С
		template<typename T>
		using PtrSize = PtrSzPCL<T>;

		//����Device��ַ�Լ������ֽ�Ϊ��λ������������֮��Ĳ��������Խ�DeviceArray����ת��PtrStep�����ں˺�����ֱ�ӷ���(DeviceArray�ں˺����в���ֱ�ӷ���)
		template<typename T>
		using PtrStep = PtrStepPCL<T>;

		//����Device��ַ�����ݵĴ�С�Լ������ֽ�Ϊ��λ������������֮��Ĳ���
		template<typename T>
		using PtrStepSize = PtrStepSzPCL<T>;
	}

	/**
	 * ����ڲ�
	 */
	struct Intrinsic
	{
		/**
		 * \brief ������host��device�������Intrinsic��������
		 * \return
		 */
		__host__ __device__ Intrinsic()
			: principal_x(0), principal_y(0),
			focal_x(0), focal_y(0) {}

		__host__ __device__ Intrinsic(
			const float focal_x_, const float focal_y_,
			const float principal_x_, const float principal_y_
		) : principal_x(principal_x_), principal_y(principal_y_),
			focal_x(focal_x_), focal_y(focal_y_) {}

		//����float4   [ֱ������float4(),��������ֱ���Ҹ�ֵ -> float4 A = (Intrinsic) B]
		__host__ operator float4() {
			return make_float4(principal_x, principal_y, focal_x, focal_y);
		}

		// ����ڲ�
		float principal_x, principal_y;
		float focal_x, focal_y;
	};

	/**
	 * ����ڲε���
	 */
	struct IntrinsicInverse
	{
		//Ĭ�ϸ���ֵ0
		__host__ __device__ IntrinsicInverse() : principal_x(0), principal_y(0), inv_focal_x(0), inv_focal_y(0) {}

		// ����ڲ�
		float principal_x, principal_y; //����ڲ����ĵ�
		float inv_focal_x, inv_focal_y; //�������ĵ���
	};

	/**
	 * \brief ��������������ϣ�������cudaSurfaceObject_t��cudaTextureObject_t�������ͣ��Լ���Ӧ����cudaArray.
	 */
	struct CudaTextureSurface {
		cudaTextureObject_t texture;	//�����ڴ棬�ɶ�����д�����ص�ֱ�ӿ��Խ���Ӳ����ֵ
		cudaSurfaceObject_t surface;	//�����ڴ棬�ɶ���д��������OpenGL�е�Resource�ռ����ӳ����߹�����ͨ��CPU
		cudaArray_t cudaArray;			//����һ��Array������ݣ�����cudaArray_t����Ҳ����������GPU��ʵ�ʵ�����
	};

	//����ȡ�������������������ģ�ͬconvenience.cuh�е�getGridDim()����
	using pcl::gpu::divUp;

	/**
	 * \brief �����ṹ���¼����.
	 */
	struct PixelCoordinate {
		unsigned int row;	// ͼ��ĸߣ����ص�y����
		unsigned int col;	// ͼ��Ŀ����ص�x����
		__host__ __device__ PixelCoordinate() : row(0), col(0) {}
		__host__ __device__ PixelCoordinate(const unsigned row_, const unsigned col_)
			: row(row_), col(col_) {}

		__host__ __device__ const unsigned int& x() const { return col; }
		__host__ __device__ const unsigned int& y() const { return row; }
		__host__ __device__ unsigned int& x() { return col; }
		__host__ __device__ unsigned int& y() { return row; }
	};

	/**
	 * \brief �����ͼ�񹹽���surfel�ṹӦ�����豸�Ϸ���.
	 */
	struct DepthSurfel {
		PixelCoordinate pixelCoordinate;	// pixelCoordinate��Ԫ��������
		float4 VertexAndConfidence;			// VertexAndConfidence (x, y, z)Ϊ���֡�е�λ�ã�(w)Ϊ���Ŷ�ֵ��
		float4 NormalAndRadius;				// NormalAndRadius (x, y, z)�ǹ�һ�����߷���w�ǰ뾶
		float4 ColorAndTime;				// ColorAndTime (x)�Ǹ�������RGBֵ;(z)Ϊ���һ�ι۲�ʱ��;(w)Ϊ��ʼ��ʱ��
	};

	struct KNNAndWeight {
		ushort4 knn;		// �ٽ�4�����ID
		float4 weight;		// �ٽ�4�����Ȩ��

		__host__ __device__ void setInvalid() {
			knn.x = knn.y = knn.z = knn.w = 0xFFFF;
			weight.x = weight.y = weight.z = weight.w = 0.0f;
		}
	};

	struct markCompact {
		/**
		 * \brief ��� x �ĵ� 32 λ�Ƿ�Ϊ 1����� x �ĵ� 32 λΪ 1���򷵻� true�����򷵻� false.
		 *
		 * \param x ����32Ϊ������
		 * \return ���ؼ����
		 */
		__device__ bool operator()(const long long& x) {
			return (x & (1ll << 31)) > 0;
		}
	};

}

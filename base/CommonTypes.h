#pragma once
#include "GlobalConfigs.h"
#include <base/DeviceAPI/convenience.cuh>
#include <base/DeviceAPI/device_array.hpp>
#include <base/DeviceAPI/kernel_containers.hpp>
#include <base/DeviceAPI/safe_call.hpp>
//CUDA类型
#include <vector_functions.h>
#include <vector>

namespace SparseSurfelFusion {
	/* host和device访问的 gpu 容器
	*/
	//用DeviceArray代替DeviceArrayPCL
	template<typename T>
	using DeviceArray = DeviceArrayPCL<T>;
	//用DeviceArray2D代替DeviceArray2DPCL
	template<typename T>
	using DeviceArray2D = DeviceArray2DPCL<T>;

	namespace device {
		//device访问 gpu 容器的类型
		//存了Device地址的指针
		template<typename T>
		using DevicePtr = DevPtr<T>;

		//存了Device地址以及数据大小
		template<typename T>
		using PtrSize = PtrSzPCL<T>;

		//存了Device地址以及在以字节为单位的两个连续行之间的步长，可以将DeviceArray类型转成PtrStep，并在核函数中直接访问(DeviceArray在核函数中不能直接访问)
		template<typename T>
		using PtrStep = PtrStepPCL<T>;

		//存了Device地址、数据的大小以及在以字节为单位的两个连续行之间的步长
		template<typename T>
		using PtrStepSize = PtrStepSzPCL<T>;
	}

	/**
	 * 相机内参
	 */
	struct Intrinsic
	{
		/**
		 * \brief 允许在host和device上面进行Intrinsic函数构造
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

		//构造float4   [直接重载float4(),这样可以直接右赋值 -> float4 A = (Intrinsic) B]
		__host__ operator float4() {
			return make_float4(principal_x, principal_y, focal_x, focal_y);
		}

		// 相机内参
		float principal_x, principal_y;
		float focal_x, focal_y;
	};

	/**
	 * 相机内参倒数
	 */
	struct IntrinsicInverse
	{
		//默认赋初值0
		__host__ __device__ IntrinsicInverse() : principal_x(0), principal_y(0), inv_focal_x(0), inv_focal_y(0) {}

		// 相机内参
		float principal_x, principal_y; //相机内参中心点
		float inv_focal_x, inv_focal_y; //相机焦距的倒数
	};

	/**
	 * \brief 给定数组的纹理集合，其中有cudaSurfaceObject_t和cudaTextureObject_t两种类型，以及对应数据cudaArray.
	 */
	struct CudaTextureSurface {
		cudaTextureObject_t texture;	//纹理内存，可读不可写，像素点直接可以进行硬件插值
		cudaSurfaceObject_t surface;	//表面内存，可读可写，可以与OpenGL中的Resource空间进行映射或者共享，不通过CPU
		cudaArray_t cudaArray;			//创建一个Array存放数据，类型cudaArray_t，这也就是数据在GPU上实际的载体
	};

	//向上取整函数，算网格数量的，同convenience.cuh中的getGridDim()函数
	using pcl::gpu::divUp;

	/**
	 * \brief 辅助结构体记录像素.
	 */
	struct PixelCoordinate {
		unsigned int row;	// 图像的高，像素点y坐标
		unsigned int col;	// 图像的宽，像素点x坐标
		__host__ __device__ PixelCoordinate() : row(0), col(0) {}
		__host__ __device__ PixelCoordinate(const unsigned row_, const unsigned col_)
			: row(row_), col(col_) {}

		__host__ __device__ const unsigned int& x() const { return col; }
		__host__ __device__ const unsigned int& y() const { return row; }
		__host__ __device__ unsigned int& x() { return col; }
		__host__ __device__ unsigned int& y() { return row; }
	};

	/**
	 * \brief 从深度图像构建的surfel结构应该在设备上访问.
	 */
	struct DepthSurfel {
		PixelCoordinate pixelCoordinate;	// pixelCoordinate面元来自哪里
		float4 VertexAndConfidence;			// VertexAndConfidence (x, y, z)为相机帧中的位置，(w)为置信度值。
		float4 NormalAndRadius;				// NormalAndRadius (x, y, z)是归一化法线方向，w是半径
		float4 ColorAndTime;				// ColorAndTime (x)是浮点编码的RGB值;(z)为最后一次观测时间;(w)为初始化时间
	};

	struct KNNAndWeight {
		ushort4 knn;		// 临近4个点的ID
		float4 weight;		// 临近4个点的权重

		__host__ __device__ void setInvalid() {
			knn.x = knn.y = knn.z = knn.w = 0xFFFF;
			weight.x = weight.y = weight.z = weight.w = 0.0f;
		}
	};

	struct markCompact {
		/**
		 * \brief 检查 x 的第 32 位是否为 1。如果 x 的第 32 位为 1，则返回 true，否则返回 false.
		 *
		 * \param x 输入32为长整型
		 * \return 返回检查结果
		 */
		__device__ bool operator()(const long long& x) {
			return (x & (1ll << 31)) > 0;
		}
	};

}

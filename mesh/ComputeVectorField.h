/*****************************************************************//**
 * \file   ComputeVectorField.h
 * \brief  计算向量场
 * 
 * \author LUOJIAXUAN
 * \date   May 15th 2024
 *********************************************************************/
#pragma once
#include <chrono>
#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>
#include <math/PPolynomial.h>
#include <base/Constants.h>
#include <base/GlobalConfigs.h>
#include <base/DeviceReadWrite/DeviceBufferArray.h>
#include "ConfirmedPPolynomial.h"
#include "FunctionData.h"
#include "Geometry.h"
#include "BuildOctree.h"
#include "BinaryNode.h"

namespace SparseSurfelFusion {
	namespace device {

		__device__ float FCenterWidthPoint(int idx, const ConfirmedPPolynomial<CONVTIMES, CONVTIMES + 2>& BaseFunctionMaxDepth_d, const Point3D<float>& center, const float& width, const Point3D<float>& point);

		__device__ void getFunctionIdxNode(const int& key, const int& maxDepth, int* idx);

		__global__ void CalculateVectorFieldKernel(ConfirmedPPolynomial<CONVTIMES, CONVTIMES + 2>* BaseFunctionMaxDepth_Device, DeviceArrayView<OrientedPoint3D<float>> DenseOrientedPoints, DeviceArrayView<OctNode> NodeArray, const unsigned int DLevelOffset, const unsigned int DLevelNodeNum, Point3D<float>* VectorField);
	}

	class ComputeVectorField
	{
	public:

		ComputeVectorField(cudaStream_t stream);

		~ComputeVectorField();

		using Ptr = std::shared_ptr<ComputeVectorField>;

		/**
		 * \brief 构建VectorField【无阻塞】.
		 * 
		 * \param orientedPoints 稠密有向点
		 * \param NodeArray 八叉树节点数组
		 * \param NodeArrayCount 每一层节点的数量
		 * \param BaseAddressArray 每一层首节点的偏移
		 * \param stream cuda流
		 */
		void BuildVectorField(DeviceArrayView<OrientedPoint3D<float>> orientedPoints, DeviceArrayView<OctNode> NodeArray, const int* NodeArrayCount, const int* BaseAddressArray, cudaStream_t stream);

		/**
		 * \brief 获得向量场.
		 * 
		 * \return 向量场
		 */
		DeviceArrayView<Point3D<float>> GetVectorField() { return VectorField.ArrayView(); }
			
		/**
		 * \brief 获得点积表<F, F>.
		 * 
		 * \return 点积表<F, F>.
		 */
		DeviceArrayView<double> GetValueTable_Dot_F_F() { return dot_F_F.ArrayView(); }

		/**
		 * \brief 获得一阶导数点积表<F, dF>.
		 * 
		 * \return 一阶导数点积表<F, dF>.
		 */
		DeviceArrayView<double> GetValueTable_Dot_F_dF() { return dot_F_DF.ArrayView(); }

		/**
		 * \brief 获得二阶导数点积表<F, d2F>.
		 * 
		 * \return 二阶导数点积表<F, d2F>
		 */
		DeviceArrayView<double> GetValueTable_Dot_F_d2F() { return dot_F_D2F.ArrayView(); }
		/**
		 * \brief 获得基函数.
		 * 
		 * \return 基函数
		 */
		DeviceArrayView<ConfirmedPPolynomial<CONVTIMES + 1, CONVTIMES + 2>> GetBaseFunction() { return baseFunctions_Device.ArrayView(); }
	private:

		const int normalize = NORMALIZE;

		PPolynomial<CONVTIMES> ReconstructionFunction;		// 预先计算重建基函数

		void AllocateBuffer();

		/**
		 * \brief 预先计算构建点积表.
		 * 
		 * \param stream cuda流
		 */
		void BuildInnerProductTable(cudaStream_t stream);


		DeviceBufferArray<double> dot_F_F; 	   // 基函数的点积表
		DeviceBufferArray<double> dot_F_DF;	   // 基函数一阶导数的点积表
		DeviceBufferArray<double> dot_F_D2F;   // 基函数二阶导数的点积表
		DeviceBufferArray<ConfirmedPPolynomial<CONVTIMES + 1, CONVTIMES + 2>> baseFunctions_Device;		// 基函数拷贝到GPU
		DeviceBufferArray<ConfirmedPPolynomial<CONVTIMES, CONVTIMES + 2>> BaseFunctionMaxDepth_Device;	// 最大层的基函数【基函数每一层[0, maxDepth]都放缩，每一层的节点都有与之匹配的基函数匹配】
		DeviceBufferArray<Point3D<float>> VectorField;	// 向量流(有1‰的误差)

		/**
		 * \brief 计算VectorField.
		 *
		 * \param BaseFunctionMaxDepth_Device 最大层的基函数
		 * \param DenseOrientedPoints 稠密有向点
		 * \param NodeArray 八叉树节点数组
		 * \param DLevelOffset D层首节点在NodeArray中的偏移
		 * \param DLevelNodeNum D层节点数量
		 * \param VectorField 向量场
		 * \param stream cuda流
		 */
		void CalculateVectorField(ConfirmedPPolynomial<CONVTIMES, CONVTIMES + 2>* BaseFunctionMaxDepth_Device, DeviceArrayView<OrientedPoint3D<float>> DenseOrientedPoints, DeviceArrayView<OctNode> NodeArray, const unsigned int DLevelOffset, const unsigned int DLevelNodeNum, DeviceBufferArray<Point3D<float>>& VectorField, cudaStream_t stream);
	};
}



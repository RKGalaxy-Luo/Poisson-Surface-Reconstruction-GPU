/*****************************************************************//**
 * \file   LaplacianSolver.h
 * \brief  拉普拉斯求解器
 * 
 * \author LUOJIAXUAN
 * \date   May 26th 2024
 *********************************************************************/
#pragma once
#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>
#include <thrust/device_vector.h>
#include <base/ThreadPool.h>
#include <base/DeviceReadWrite/DeviceBufferArray.h>
#include <mesh/Geometry.h>
#include <mesh/BuildOctree.h>
#include <mesh/ConfirmedPPolynomial.h>

#if defined(__CUDACC__)		//如果由NVCC编译器编译
#include <mesh/solver/CGAlgorithm.cuh>
#endif

namespace SparseSurfelFusion {
	namespace device {
		/**
		 * \brief 根据点积表计算矩阵每个元素的Laplace元素值，并标记无效值防止减少元素个数，防止后续参与计算.
		 * 
		 * \param dot_F_F 基函数内积表
		 * \param dot_F_D2F 基函数二阶导函数内积表
		 * \param encodeNodeIndexInFunction 编码节点的在函数中索引
		 * \param NodeArray 八叉树一维节点
		 * \param begin 当前核函数遍历NodeArray的起始位置
		 * \param calculatedNodeNum 当前核函数需要遍历的节点数量
		 * \param rowCount 记录一个节点及其邻居有效的colIndex的个数，以便后面计算所需开辟的空间
		 * \param colIndex 记录一个节点及其邻居有效的colIndex(有效 <==> fabs(LaplacianEntryValue) > device::eps)
		 * \param val 记录一个节点及其邻居有效的LaplacianEntryValue的值
		 */
		__global__ void GenerateSingleNodeLaplacian(DeviceArrayView<double> dot_F_F, DeviceArrayView<double> dot_F_D2F, DeviceArrayView<int> encodeNodeIndexInFunction, DeviceArrayView<OctNode> NodeArray, const unsigned int begin, const unsigned int calculatedNodeNum, int* rowCount, int* colIndex, float* val);
	
		/**
		 * \brief 计算获得Laplace矩阵的元素.
		 * 
		 * \param dot_F_F 基函数内积元素
		 * \param dot_F_D2F 基函数二阶导函数内积元素
		 * \param index 访问基函数点积表的index
		 */
		__device__ double GetLaplacianEntry(DeviceArrayView<double> dot_F_F, DeviceArrayView<double> dot_F_D2F, const int* index);

		/**
		 * \brief 标记有效的col列数，小于device::eps会被认为是无效的.
		 * 
		 * \param 传入需要判别的colIndex
		 * \param 一共有多少列
		 * \param 传出标志位，true是有效的col
		 */
		__global__ void markValidColIndex(const int* colIndex, const unsigned int nodeNum, bool* flag);

		/**
		 * \brief 计算稠密点的隐函数的值.
		 * 
		 * \param DensePoints 稠密点
		 * \param PointToNodeArrayDLevel 从原始稠密sampleOrientedPoints数组中点对应NodeArrayD中node的位置，没有对应的一律写为-1
		 * \param NodeArray 八叉树节点数组
		 * \param encodeNodeIndexInFunction 编码节点在基函数中索引
		 * \param BaseFunctions 基函数
		 * \param DLevelOffset maxDepth层在NodeArray中首节点的位置
		 * \param DenseVertexCount 稠密点数量
		 * \param pointsValue 稠密点的隐函数值
		 */
		__global__ void CalculatePointsImplicitFunctionValueKernel(DeviceArrayView<OrientedPoint3D<float>> DensePoints, DeviceArrayView<int> PointToNodeArrayDLevel, DeviceArrayView<OctNode> NodeArray, DeviceArrayView<int> encodeNodeIndexInFunction, DeviceArrayView<ConfirmedPPolynomial<CONVTIMES + 1, CONVTIMES + 2>> BaseFunctions, DeviceArrayView<float> dx, const unsigned int DLevelOffset, const unsigned int DenseVertexCount, float* pointsValue);
	}

	class LaplacianSolver
	{
	public:
		LaplacianSolver();

		~LaplacianSolver();

		using Ptr = std::shared_ptr<LaplacianSolver>;


		/**
		 * \brief 共轭梯度法(CG)求解散度拉普拉斯算子【后续速度优化：1、可加入多线程   2、如果内存可控可以预先开辟内存】.
		 * 
		 * \param BaseAddressArray 每层首节点在NodeArray中的偏移
		 * \param NodeArrayCount 每层的节点数量
		 * \param encodeNodeIndexInFunction 编码节点在基函数中索引
		 * \param NodeArray 八叉树节点数组
		 * \param Divergence 节点散度
		 * \param dot_F_F 基函数内积表
		 * \param dot_F_D2F 二阶基函数内积表
		 * \param streams cuda流数组
		 */
		void LaplacianCGSolver(const int* BaseAddressArray, const int* NodeArrayCount, DeviceArrayView<int> encodeNodeIndexInFunction, DeviceArrayView<OctNode> NodeArray, float* Divergence, DeviceArrayView<double> dot_F_F, DeviceArrayView<double> dot_F_D2F, cudaStream_t stream);

		/**
		 * \brief 计算稠密点的隐式函数值.
		 * 
		 * \param DensePoints 稠密点
		 * \param PointToNodeArrayDLevel 从原始稠密sampleOrientedPoints数组中点对应NodeArrayD中node的位置，没有对应的一律写为-1
		 * \param NodeArray 八叉树节点数组
		 * \param encodeNodeIndexInFunction 编码节点在基函数中索引
		 * \param BaseFunctions 基函数
		 * \param DLevelOffset maxDepth层在NodeArray中首节点的位置
		 * \param DenseVertexCount 稠密点数量
		 */
		void CalculatePointsImplicitFunctionValue(DeviceArrayView<OrientedPoint3D<float>> DensePoints, DeviceArrayView<int> PointToNodeArrayDLevel, DeviceArrayView<OctNode> NodeArray, DeviceArrayView<int> encodeNodeIndexInFunction, DeviceArrayView<ConfirmedPPolynomial<CONVTIMES + 1, CONVTIMES + 2>> BaseFunctions, const unsigned int DLevelOffset, const unsigned int DenseVertexCount, cudaStream_t stream);

		/**
		 * \brief 获得dx.
		 * 
		 * \return 
		 */
		DeviceArrayView<float> GetDx() { return dx.ArrayView(); }

		/**
		 * \brief 获得等值.
		 * 
		 * \return 获得等值
		 */
		float GetIsoValue() const { return isoValue; }

	private:
		//std::shared_ptr<ThreadPool> pool;
		DeviceBufferArray<float> dx;	// 散度的Laplace迭代后的解

		DeviceBufferArray<float> DensePointsImplicitFunctionValue;	// 稠密点隐函数值

		float isoValue = -1.0f;
	};
}



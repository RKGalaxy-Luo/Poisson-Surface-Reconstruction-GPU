/*****************************************************************//**
 * \file   ComputeNodesDivergence.h
 * \brief  计算节点的散度
 * 
 * \author LUOJIAXUAN
 * \date   May 24th 2024
 *********************************************************************/
#pragma once
#include <chrono>
#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>
#include <mesh/BuildOctree.h>
#include <core/AlgorithmTypes.h>
#include <mesh/Geometry.h>
#include <base/DeviceReadWrite/DeviceBufferArray.h>

namespace SparseSurfelFusion {

	namespace device {
		/**
		 * \brief 计算精细节点的散度核函数.
		 * 
		 * \param BaseAddressArray 节点偏移数组
		 * \param encodeNodeIndexInFunction 编码节点的在函数中索引
		 * \param NodeArray 八叉树一维节点
		 * \param VectorField 向量场
		 * \param dot_F_DF 点积表
		 * \param begin 在NodeArray偏移开始
		 * \param calculatedNodeNum 需要参与计算的节点总数
		 * \param Divergence 节点散度
		 */
		__global__ void computeFinerNodesDivergenceKernel(DeviceArrayView<int> BaseAddressArray, DeviceArrayView<int> encodeNodeIndexInFunction, DeviceArrayView<OctNode> NodeArray, DeviceArrayView<Point3D<float>> VectorField, DeviceArrayView<double> dot_F_DF, const unsigned int begin, const unsigned int calculatedNodeNum, float* Divergence);

		/**
		 * \brief 两个向量点乘.
		 * 
		 * \param p1 向量1
		 * \param p2 向量2
		 */
		__device__ float DotProduct(const Point3D<float>& p1, const Point3D<float>& p2);

		/**
		 * \brief 计算该节点及其邻居节点所覆盖的maxDepth层节点的数量.
		 * 
		 * \param NodeArray 八叉树一维节点
		 * \param index 当前点在NodeArray中的位置
		 * \param coverNums 覆盖的节点数量
		 */
		__global__ void computeCoverNums(DeviceArrayView<OctNode> NodeArray, const unsigned int index, unsigned int* coverNums);

		/**
		 * \brief 生成：idx = [0, totalCoverNum)个D层节点，将这些节点映射到D层对应的D_idx.
		 * 
		 * \param NodeArray 八叉树一维节点
		 * \param index 当前点在NodeArray中的位置
		 * \param coverNums 覆盖的节点数量
		 * \param totalCoverNum 当前节点及其邻居节点覆盖的D层节点的节点总数
		 * \param DLevelIndexArray 映射关系数组
		 */
		__global__ void generateDLevelIndexArrayKernel(DeviceArrayView<OctNode> NodeArray, const unsigned int index, const unsigned int* coverNums, const unsigned int totalCoverNum, unsigned int* DLevelIndexArray);

		/**
		 * \brief 计算当前节点散度的核函数.
		 * 
		 * \param BaseAddressArrayDevice 节点偏移数组
		 * \param encodeNodeIndexInFunction 编码节点的在函数中索引
		 * \param VectorField 向量场
		 * \param dot_F_DF 点积表
		 * \param index 当前所需要计算的节点在NodeArray中的index
		 * \param DLevelIndexArray 映射关系数组
		 * \param totalCoverNum 当前节点及其邻居节点覆盖的D层节点的节点总数
		 * \param divg 需要计算的散度值，多个值
		 */
		__global__ void computeCoarserNodesDivergenceKernel(DeviceArrayView<int> BaseAddressArrayDevice, DeviceArrayView<int> encodeNodeIndexInFunction, DeviceArrayView<Point3D<float>> VectorField, DeviceArrayView<double> dot_F_DF, const unsigned int index, const unsigned int* DLevelIndexArray, const unsigned int totalCoverNum, float* divg);
	}
	/**
	 * \brief 计算节点的散度.
	 */
	class ComputeNodesDivergence
	{
	public:
		ComputeNodesDivergence();

		~ComputeNodesDivergence();

		using Ptr = std::shared_ptr<ComputeNodesDivergence>;


		/**
		 * \brief 计算节点的散度【两个流分别并行执行：时间减少44%】.
		 *
		 * \param BaseAddressArrayDevice 节点偏移数组(Host)
		 * \param NodeArrayCount 每一层节点数量(Host)
		 * \param encodeNodeIndexInFunction 编码节点的在函数中索引
		 * \param NodeArray 八叉树一维节点
		 * \param VectorField 向量场
		 * \param dot_F_DF 点积表
		 * \param stream_1 cuda流1
		 * \param stream_2 cuda流2
		 */
		void CalculateNodesDivergence(const int* BaseAddressArray, const int* NodeArrayCount, DeviceArrayView<int> BaseAddressArrayDevice, DeviceArrayView<int> encodeNodeIndexInFunction, DeviceArrayView<OctNode> NodeArray, DeviceArrayView<Point3D<float>> VectorField, DeviceArrayView<double> dot_F_DF, cudaStream_t stream_1, cudaStream_t stream_2);

		/**
		 * \brief 获得节点散度(只读).
		 * 
		 * \return NodeArray中节点的散度
		 */
		DeviceArrayView<float> GetDivergenceView() { return Divergence.ArrayView(); }

		/**
		 * \brief 获得节点散度指针.
		 * 
		 * \return 节点散度指针
		 */
		float* GetDivergenceRawPtr() { return Divergence.Array().ptr(); }
	private:

		DeviceBufferArray<float> Divergence;			// 节点的散度

		/**
		 * \brief 计算精细节点的散度【计算[maxDepth - FinerLevelNum, maxDepth]层的节点散度，数据有相差30%，多数是1‰】.
		 *
		 * \param BaseAddressArrayDevice 节点偏移数组(GPU)
		 * \param encodeNodeIndexInFunction 编码节点的在函数中索引
		 * \param NodeArray 八叉树一维节点
		 * \param VectorField 向量场
		 * \param dot_F_DF 点积表
		 * \param left 参与计算的八叉树节点数组的左边界index  【参与计算节点index的区间范围为[left, right]】
		 * \param right 参与计算的八叉树节点数组的右边界index 【参与计算节点index的区间范围为[left, right]】
		 * \param cuda流
		 */
		void computeFinerNodesDivergence(DeviceArrayView<int> BaseAddressArrayDevice, DeviceArrayView<int> encodeNodeIndexInFunction, DeviceArrayView<OctNode> NodeArray, DeviceArrayView<Point3D<float>> VectorField, DeviceArrayView<double> dot_F_DF, const unsigned int left, const unsigned int right, cudaStream_t stream);

		/**
		 * \brief 计算粗糙节点的散度【计算[1, CoarserLevelNum]层节点的散度】【不阻塞线程】.
		 *
		 * \param BaseAddressArrayDevice 节点偏移数组(GPU)
		 * \param encodeNodeIndexInFunction 编码节点的在函数中索引
		 * \param NodeArray 八叉树一维节点
		 * \param VectorField 向量场
		 * \param dot_F_DF 点积表
		 * \param left 参与计算的八叉树节点数组的左边界index  【参与计算节点index的区间范围为[left, right]】
		 * \param right 参与计算的八叉树节点数组的右边界index 【参与计算节点index的区间范围为[left, right]】
		 * \param stream cuda流
		 */
		void computeCoarserNodesDivergence(const int* BaseAddressArray, DeviceArrayView<int> BaseAddressArrayDevice, DeviceArrayView<int> encodeNodeIndexInFunction, DeviceArrayView<OctNode> NodeArray, DeviceArrayView<Point3D<float>> VectorField, DeviceArrayView<double> dot_F_DF, const unsigned int left, const unsigned int right, cudaStream_t stream);
	};
}



/*****************************************************************//**
 * \file   ComputeNodesDivergence.cpp
 * \brief  计算节点散度方法
 * 
 * \author LUOJIAXUAN
 * \date   May 24th 2024
 *********************************************************************/
#include "ComputeNodesDivergence.h"
SparseSurfelFusion::ComputeNodesDivergence::ComputeNodesDivergence()
{
	Divergence.AllocateBuffer(TOTAL_NODEARRAY_MAX_COUNT);			// 节点散度
}

SparseSurfelFusion::ComputeNodesDivergence::~ComputeNodesDivergence()
{
	Divergence.ReleaseBuffer();
}

void SparseSurfelFusion::ComputeNodesDivergence::CalculateNodesDivergence(const int* BaseAddressArray, const int* NodeArrayCount, DeviceArrayView<int> BaseAddressArrayDevice, DeviceArrayView<int> encodeNodeIndexInFunction, DeviceArrayView<OctNode> NodeArray, DeviceArrayView<Point3D<float>> VectorField, DeviceArrayView<double> dot_F_DF, cudaStream_t stream_1, cudaStream_t stream_2)
{
#ifdef CHECK_MESH_BUILD_TIME_COST
	auto start = std::chrono::high_resolution_clock::now();						// 记录开始时间点
#endif // CHECK_MESH_BUILD_TIME_COST

	// 两个流分别并行执行：时间减少44%
	computeFinerNodesDivergence(BaseAddressArrayDevice, encodeNodeIndexInFunction, NodeArray, VectorField, dot_F_DF, BaseAddressArray[COARSER_DIVERGENCE_LEVEL_NUM + 1], BaseAddressArray[MAX_DEPTH_OCTREE] + NodeArrayCount[MAX_DEPTH_OCTREE], stream_1);
	computeCoarserNodesDivergence(BaseAddressArray, BaseAddressArrayDevice, encodeNodeIndexInFunction, NodeArray, VectorField, dot_F_DF, BaseAddressArray[0], BaseAddressArray[COARSER_DIVERGENCE_LEVEL_NUM] + NodeArrayCount[COARSER_DIVERGENCE_LEVEL_NUM], stream_2);

#ifdef CHECK_MESH_BUILD_TIME_COST
	// 所有参与的流均同步
	CHECKCUDA(cudaStreamSynchronize(stream_1));
	CHECKCUDA(cudaStreamSynchronize(stream_2));

	auto end = std::chrono::high_resolution_clock::now();							// 记录结束时间点
	std::chrono::duration<double, std::milli> duration = end - start;				// 计算执行时间（以ms为单位）
	std::cout << "计算节点散度的时间: " << duration.count() << " ms" << std::endl;		// 输出
	std::cout << std::endl;
	std::cout << "-----------------------------------------------------" << std::endl;	// 输出
	std::cout << std::endl;
#endif // CHECK_MESH_BUILD_TIME_COST
}
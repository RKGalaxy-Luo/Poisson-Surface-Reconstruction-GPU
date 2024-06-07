/*****************************************************************//**
 * \file   ComputeTriangleIndices.cpp
 * \brief  插入修复三角网格，构建网格
 * 
 * \author LUOJIAXUAN
 * \date   June 3rd 2024
 *********************************************************************/
#include "ComputeTriangleIndices.h"

SparseSurfelFusion::ComputeTriangleIndices::ComputeTriangleIndices()
{
	vvalue.AllocateBuffer(TOTAL_VERTEXARRAY_MAX_COUNT);
	vexNums.AllocateBuffer(TOTAL_EDGEARRAY_MAX_COUNT);
	vexAddress.AllocateBuffer(TOTAL_EDGEARRAY_MAX_COUNT);
	triNums.AllocateBuffer(D_LEVEL_MAX_NODE);
	triAddress.AllocateBuffer(D_LEVEL_MAX_NODE);
	cubeCatagory.AllocateBuffer(D_LEVEL_MAX_NODE);
	markValidVertex.AllocateBuffer(TOTAL_EDGEARRAY_MAX_COUNT);
	SubdivideNode.AllocateBuffer(TOTAL_NODEARRAY_MAX_COUNT - D_LEVEL_MAX_NODE);				// 非maxDepth层的最大节点数量
	markValidSubdividedNode.AllocateBuffer(TOTAL_NODEARRAY_MAX_COUNT - D_LEVEL_MAX_NODE);	// 非maxDepth层的最大节点数量
	SubdivideDepthBuffer.AllocateBuffer(TOTAL_NODEARRAY_MAX_COUNT - D_LEVEL_MAX_NODE);
	markValidSubdivideVertex.AllocateBuffer(int(1 << 21));									// 设置最大为8^7
	markValidSubdivideEdge.AllocateBuffer(int(1 << 21));
	markValidSubdivedeVexNum.AllocateBuffer(int(1 << 21));
	markValidFinerVexArray.AllocateBuffer(int(1 << 22));
	markValidFinerEdge.AllocateBuffer(int(1 << 22));
	markValidFinerVexNum.AllocateBuffer(int(1 << 22));
}

SparseSurfelFusion::ComputeTriangleIndices::~ComputeTriangleIndices()
{
	vvalue.ReleaseBuffer();
	vexNums.ReleaseBuffer();
	vexAddress.ReleaseBuffer();
	triNums.ReleaseBuffer();
	triAddress.ReleaseBuffer();
	cubeCatagory.ReleaseBuffer();
	markValidVertex.ReleaseBuffer();
	SubdivideNode.ReleaseBuffer();
	markValidSubdividedNode.ReleaseBuffer();
	SubdivideDepthBuffer.DeviceArray().release();
	SubdivideDepthBuffer.HostArray().clear();
	markValidSubdivideVertex.ReleaseBuffer();
	markValidSubdivideEdge.ReleaseBuffer();
	markValidSubdivedeVexNum.ReleaseBuffer();
	markValidFinerVexArray.ReleaseBuffer();
	markValidFinerEdge.ReleaseBuffer();
	markValidFinerVexNum.ReleaseBuffer();
}

void SparseSurfelFusion::ComputeTriangleIndices::calculateTriangleIndices(DeviceArrayView<VertexNode> VertexArray, DeviceArrayView<EdgeNode> EdgeArray, DeviceArrayView<FaceNode> FaceArray, DeviceBufferArray<OctNode>& NodeArray, DeviceArrayView<ConfirmedPPolynomial<CONVTIMES + 1, CONVTIMES + 2>> BaseFunction, DeviceArrayView<float> dx, DeviceArrayView<int> encodeNodeIndexInFunction, DeviceArrayView<unsigned int> DepthBuffer, DeviceArrayView<Point3D<float>> CenterBuffer, const unsigned int isoValue, const unsigned int DLevelOffset, const unsigned int DLevelNodeCount, CoredVectorMeshData& mesh, cudaStream_t stream)
{
#ifdef CHECK_MESH_BUILD_TIME_COST
	auto time1 = std::chrono::high_resolution_clock::now();					// 记录开始时间点
#endif // CHECK_MESH_BUILD_TIME_COST

	//printf("VertexCount = %d   EdgeCount = %d   FaceCount = %d\n", VertexArray.Size(), EdgeArray.Size(), FaceArray.Size());

	/**************************** Step 1: 计算八叉树顶点的隐式函数值 ****************************/
	ComputeVertexImplicitFunctionValue(VertexArray, NodeArray.ArrayView(), BaseFunction, dx, encodeNodeIndexInFunction, isoValue, stream);
#ifdef CHECK_MESH_BUILD_TIME_COST
	CHECKCUDA(cudaStreamSynchronize(stream));
	auto time2 = std::chrono::high_resolution_clock::now();					// 记录结束时间点
	std::chrono::duration<double, std::milli> duration1 = time2 - time1;		// 计算执行时间（以ms为单位）
	std::cout << "计算顶点隐式函数值的时间: " << duration1.count() << " ms" << std::endl;
#endif // CHECK_MESH_BUILD_TIME_COST

	/**************************** Step 2: 计算边的数量以及偏移 ****************************/
	generateVertexNumsAndVertexAddress(EdgeArray, NodeArray.ArrayView(), vvalue.ArrayView(), DLevelOffset, stream);
#ifdef CHECK_MESH_BUILD_TIME_COST
	CHECKCUDA(cudaStreamSynchronize(stream));
	auto time3 = std::chrono::high_resolution_clock::now();							// 记录结束时间点
	std::chrono::duration<double, std::milli> duration2 = time3 - time2;			// 计算执行时间（以ms为单位）
	std::cout << "计算顶点数量以及顶点位置偏移的时间: " << duration2.count() << " ms" << std::endl;
#endif // CHECK_MESH_BUILD_TIME_COST

	/**************************** Step 3: 计算三角形数量以及地址 ****************************/
	generateTriangleNumsAndTriangleAddress(NodeArray.ArrayView(), vvalue.ArrayView(), DLevelOffset, DLevelNodeCount, mesh, stream);
#ifdef CHECK_MESH_BUILD_TIME_COST
	CHECKCUDA(cudaStreamSynchronize(stream));
	auto time4 = std::chrono::high_resolution_clock::now();							// 记录结束时间点
	std::chrono::duration<double, std::milli> duration3 = time4 - time3;			// 计算执行时间（以ms为单位）
	std::cout << "计算三角剖分数量以及三角形偏移位置的时间: " << duration3.count() << " ms" << std::endl;
#endif // CHECK_MESH_BUILD_TIME_COST

	/**************************** Step 4 & 5: 生成节点及三角形网格 ****************************/
	generateVerticesAndTriangle(NodeArray, VertexArray, EdgeArray, FaceArray, DLevelOffset, DLevelNodeCount, mesh, stream);
#ifdef CHECK_MESH_BUILD_TIME_COST
	CHECKCUDA(cudaStreamSynchronize(stream));
	auto time5 = std::chrono::high_resolution_clock::now();							// 记录结束时间点
	std::chrono::duration<double, std::milli> duration4 = time5 - time4;			// 计算执行时间（以ms为单位）
	std::cout << "生成节点和网格Mesh的时间: " << duration4.count() << " ms" << std::endl;
#endif // CHECK_MESH_BUILD_TIME_COST

	/**************************** Step 6: 生成细分节点数组及其每层细分节点偏移和细分节点数量 ****************************/
	generateSubdivideNodeArrayCountAndAddress(NodeArray, DepthBuffer, DLevelOffset, stream);
#ifdef CHECK_MESH_BUILD_TIME_COST
	CHECKCUDA(cudaStreamSynchronize(stream));
	auto time6 = std::chrono::high_resolution_clock::now();							// 记录结束时间点
	std::chrono::duration<double, std::milli> duration5 = time6 - time5;			// 计算执行时间（以ms为单位）
	std::cout << "生成细分节点数组及其每层偏移和数量的时间: " << duration5.count() << " ms" << std::endl;
#endif // CHECK_MESH_BUILD_TIME_COST

	/**************************** Step 7: 处理粗节点细分重构网格【Coarser和Finer后续可以用两个线程两个流执行】 ****************************/
	CoarserSubdivideNodeAndRebuildMesh(NodeArray, DepthBuffer, CenterBuffer, BaseFunction, dx, encodeNodeIndexInFunction, isoValue, mesh, stream);
#ifdef CHECK_MESH_BUILD_TIME_COST	
	CHECKCUDA(cudaStreamSynchronize(stream));
	auto time7 = std::chrono::high_resolution_clock::now();							// 记录结束时间点
	std::chrono::duration<double, std::milli> duration6 = time7 - time6;			// 计算执行时间（以ms为单位）
	std::cout << "Coarser节点细分重构网格的时间: " << duration6.count() << " ms" << std::endl;
#endif // CHECK_MESH_BUILD_TIME_COST

	/**************************** Step 8: 处理精节点细分重构网格【Coarser和Finer后续可以用两个线程两个流执行】 ****************************/
	FinerSubdivideNodeAndRebuildMesh(NodeArray, DepthBuffer, CenterBuffer, BaseFunction, dx, encodeNodeIndexInFunction, isoValue, mesh, stream);
#ifdef CHECK_MESH_BUILD_TIME_COST		
	CHECKCUDA(cudaStreamSynchronize(stream));
	auto time8 = std::chrono::high_resolution_clock::now();							// 记录结束时间点
	std::chrono::duration<double, std::milli> duration7 = time8 - time7;			// 计算执行时间（以ms为单位）
	std::cout << "Finer节点细分重构网格的时间: " << duration7.count() << " ms" << std::endl;
#endif // CHECK_MESH_BUILD_TIME_COST

#ifdef CHECK_MESH_BUILD_TIME_COST
	CHECKCUDA(cudaStreamSynchronize(stream));	// 流同步
	auto time9 = std::chrono::high_resolution_clock::now();						// 记录结束时间点
	std::chrono::duration<double, std::milli> duration8 = time9 - time1;			// 计算执行时间（以ms为单位）
	std::cout << "计算三角剖分网格索引所需时间: " << duration8.count() << " ms" << std::endl;		// 输出
	std::cout << std::endl;
	std::cout << "-----------------------------------------------------" << std::endl;	// 输出
	std::cout << std::endl;
#endif // CHECK_MESH_BUILD_TIME_COST
}

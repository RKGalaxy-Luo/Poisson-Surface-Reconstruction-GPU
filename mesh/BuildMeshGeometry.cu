/*****************************************************************//**
 * \file   BuildMeshGeometry.cu
 * \brief  构建网格cuda方法实现
 * 
 * \author LUOJIAXUAN
 * \date   June 1st 2024
 *********************************************************************/
#include "BuildMeshGeometry.h"
#if defined(__CUDACC__)		//如果由NVCC编译器编译
#include <cub/cub.cuh>
#endif

namespace SparseSurfelFusion {
	namespace device {
		__device__ __constant__ int maxIntValue = 0x7fffffff;		// 最大int值

		__device__ __constant__ int maxDepth = MAX_DEPTH_OCTREE;

		__device__ __constant__ int parentFaceKind[8][6] = {{ 0, -1,  2, -1,  4, -1},
															{-1,  1,  2, -1,  4, -1},
															{ 0, -1, -1,  3,  4, -1},
															{-1,  1, -1,  3,  4, -1},
															{ 0, -1,  2, -1, -1,  5},
															{-1,  1,  2, -1, -1,  5},
															{ 0, -1, -1,  3, -1,  5},
															{ 0,  1, -1,  3, -1,  5}};
		//__device__ __constant__ extern int parentFaceKind[8][6] =
		//{
		//	{ 0, -1,  2, -1,  4, -1},
		//	{ 0, -1,  2, -1, -1,  5},
		//	{ 0, -1, -1,  3,  4, -1},
		//	{ 0, -1, -1,  3, -1,  5},
		//	{-1,  1,  2, -1,  4, -1},
		//	{-1,  1,  2, -1, -1,  5},
		//	{-1,  1, -1,  3,  4, -1},
		//	{ 0,  1, -1,  3, -1,  5}
		//};
	}
}


__global__ void SparseSurfelFusion::device::initVertexOwner(DeviceArrayView<OctNode> NodeArray, const unsigned int NodeArraySize, DeviceArrayView<unsigned int> depthBuffer, DeviceArrayView<Point3D<float>> centerBuffer, VertexNode* preVertexArray, bool* markPreVertexArray)
{
	const unsigned int idx = threadIdx.x + blockDim.x * blockIdx.x;
	if (idx >= NodeArraySize)	return;

	int NodeOwnerKey[8] = { device::maxIntValue, device::maxIntValue, device::maxIntValue, device::maxIntValue ,
							device::maxIntValue, device::maxIntValue, device::maxIntValue, device::maxIntValue };	// 初始无效值
	int NodeOwnerIdx[8] = { -1, -1, -1, -1, -1, -1, -1, -1 };				// 初始无效值
	int depth = depthBuffer[idx];
	float halfWidth = 1.0f / (1 << (depth + 1));							// 节点体素一半的宽
	float Width = 1.0f / (1 << depth);										// 节点体素的宽
	float WidthSquare = Width * Width;										// 节点体素宽的平方
	Point3D<float> neighborCenter[27] = { Point3D<float>(0, 0, 0) };		// 节点的27个邻居节点的中心位置
	int neighbor[27] = { -1 };														// 节点的27个邻居的index
#pragma unroll
	for (int i = 0; i < 27; i++) {
		neighbor[i] = NodeArray[idx].neighs[i];
		if (neighbor[i] != -1) {	// 邻居不为空
			neighborCenter[i] = centerBuffer[neighbor[i]];
		}
	}
	const Point3D<float>& nodeCenter = neighborCenter[13];	// 中心节点就是当前节点自己

	Point3D<float> vertexPos[8];	// 顶点位置，标记节点体素的顶点位置，一个节点 = 一个体素块，一个体素块(正方体)有8个顶点
#pragma unroll
	for (int i = 0; i < 8; i++) {	// 计算顶点位置(顶点的顺序是，顶点的顺序是，以立方体为参考：x从前到后，y从左到右，z从下到上)
		vertexPos[i].coords[0] = nodeCenter.coords[0] + (2 * (i & 1) - 1) * halfWidth;
		vertexPos[i].coords[1] = nodeCenter.coords[1] + (2 * ((i & 2) >> 1) - 1) * halfWidth;
		vertexPos[i].coords[2] = nodeCenter.coords[2] + (2 * ((i & 4) >> 2) - 1) * halfWidth;
	}

#pragma unroll
	for (int i = 0; i < 8; i++) {			// 为每个顶点找一个对应的节点
		for (int j = 0; j < 27; j++) {		// 遍历节点及邻居(供27个节点)，顶点对应key最小的节点，即key最小的节点拥有这个vertex
			if ((neighbor[j] != -1) && (device::SquareDistance(vertexPos[i], neighborCenter[j]) < WidthSquare)) { // 邻居节点必须有效，拥有这个vertex的节点不可以超过一个体素的宽
				int neighborKey = NodeArray[neighbor[j]].key;
				if (NodeOwnerKey[i] > neighborKey) {	// 如果neighborKey更小
					NodeOwnerKey[i] = neighborKey;
					NodeOwnerIdx[i] = neighbor[j];		// 将这个节点的index给NodeOwnerIdx
				}
			}
		}
	}
#pragma unroll
	for (int i = 0; i < 8; i++) {		// 遍历这8个顶点
		int vertexIdx = 8 * idx + i;	// 将preVertexArray每隔8个存一次，并且对应的位置是8个节点中的第几个
		if (NodeOwnerIdx[i] == idx) {	// 如果这个顶点是NodeArray中这个节点所拥有
			preVertexArray[vertexIdx].ownerNodeIdx = idx;						// vertex被当前这个NodeArray中的节点拥有
			preVertexArray[vertexIdx].pos.coords[0] = vertexPos[i].coords[0];	// 将vertex的位置赋值给pos
			preVertexArray[vertexIdx].pos.coords[1] = vertexPos[i].coords[1];	// 将vertex的位置赋值给pos
			preVertexArray[vertexIdx].pos.coords[2] = vertexPos[i].coords[2];	// 将vertex的位置赋值给pos
			preVertexArray[vertexIdx].vertexKind = i;							// 立方体的几号顶点
			preVertexArray[vertexIdx].depth = depth;							// 当前vertex的深度 = 当前节点的深度
			markPreVertexArray[vertexIdx] = true;
		}
		else {
			markPreVertexArray[vertexIdx] = false;	// 不是有效的全部标记为false
		}
	}
}

__forceinline__ __device__ double SparseSurfelFusion::device::SquareDistance(const Point3D<float>& p1, const Point3D<float>& p2)
{
	return (p1.coords[0] - p2.coords[0]) * (p1.coords[0] - p2.coords[0]) + (p1.coords[1] - p2.coords[1]) * (p1.coords[1] - p2.coords[1]) + (p1.coords[2] - p2.coords[2]) * (p1.coords[2] - p2.coords[2]);
}

__global__ void SparseSurfelFusion::device::maintainVertexNodePointer(DeviceArrayView<unsigned int> DepthBuffer, DeviceArrayView<Point3D<float>> CenterBuffer, const unsigned int VertexArraySize, VertexNode* VertexArray, OctNode* NodeArray)
{
	const unsigned int idx = threadIdx.x + blockDim.x * blockIdx.x;
	if (idx >= VertexArraySize)	return;
	int owner = VertexArray[idx].ownerNodeIdx;			// 拥有此顶点的Node
	int depth = DepthBuffer[owner];						// 当前顶点对应的节点的深度
	float halfWidth = 1.0f / (1 << (depth + 1));
	float Width = 1.0f / (1 << depth);
	float WidthSquare = Width * Width;
	Point3D<float> neighCenter[27];						// 邻居的中心点
	Point3D<float> vertexPos = VertexArray[idx].pos;	// 当前这个顶点的位置

	int neighbor[27];									// Owner节点的邻居节点
	for (int i = 0; i < 27; i++) {
		neighbor[i] = NodeArray[owner].neighs[i];
		if (neighbor[i] != -1) {						// 邻居是有效点
			neighCenter[i] = CenterBuffer[neighbor[i]];
		}
	}
	int count = 0;
	for (int i = 0; i < 27; i++) {
		// 邻居有效，并且是我这个邻居非跨一个邻居的邻居
		if (neighbor[i] != -1 && SquareDistance(vertexPos, neighCenter[i]) < WidthSquare) {
			VertexArray[idx].nodes[count] = neighbor[i];
			count++;
			int index = 0;	// 记录顶点相对于节点的位置编号，x先后再前，y先左再右，z先下再上
			if (neighCenter[i].coords[0] - vertexPos.coords[0] < 0) index |= 1;		// 从x方向看，邻居节点在vertex位置的后面
			if (neighCenter[i].coords[2] - vertexPos.coords[2] < 0) index |= 4;		// 从z方向看，邻居节点在vertex位置的下面
			if (neighCenter[i].coords[1] - vertexPos.coords[1] < 0) {				// 从y方向看，邻居节点在vertex位置的左边
				if (index & 1) {
					index += 1;
				}
				else {
					index += 3;
				}
			}
			NodeArray[neighbor[i]].vertices[index] = idx + 1;	// 这里给Vertice数据写入，并不影响对NodeArray的OctNode中其他数据值进行读取，因此可以并行
		}
	}
}

__global__ void SparseSurfelFusion::device::initEdgeArray(DeviceArrayView<OctNode> NodeArray, const unsigned int DLevelOffset, const unsigned int DLevelNodeCount, DeviceArrayView<unsigned int> DepthBuffer, DeviceArrayView<Point3D<float>> CenterBuffer, EdgeNode* preEdgeArray, bool* markPreEdgeArray)
{
	const unsigned int idx = threadIdx.x + blockDim.x * blockIdx.x;
	if (idx >= DLevelNodeCount)	return;
	const unsigned int offset = idx + DLevelOffset;
	int NodeOwnerKey[12] = { device::maxIntValue, device::maxIntValue, device::maxIntValue, 
							 device::maxIntValue, device::maxIntValue, device::maxIntValue,
							 device::maxIntValue, device::maxIntValue, device::maxIntValue, 
							 device::maxIntValue, device::maxIntValue, device::maxIntValue };
	int NodeOwnerIdx[12] = { -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 };
	int depth = DepthBuffer[offset];
	float halfWidth = 1.0f / (1 << (depth + 1));
	float Width = 1.0f / (1 << depth);
	float WidthSquare = Width * Width;
	Point3D<float> neighCenter[27];
	int neighbor[27];
#pragma unroll
	for (int i = 0; i < 27; i++) {
		neighbor[i] = NodeArray[offset].neighs[i];
		if (neighbor[i] != -1) {
			neighCenter[i] = CenterBuffer[neighbor[i]];
		}
	}
	const Point3D<float>& nodeCenter = neighCenter[13];
	Point3D<float> edgeCenterPos[12];
	int orientation[12];
	int off[24];
#pragma unroll
	for (int i = 0; i < 12; i++) {
		orientation[i] = i >> 2;
		off[2 * i] = i & 1;
		off[2 * i + 1] = (i & 2) >> 1;
		int multi[3];
		int dim = 2 * i;
		for (int j = 0; j < 3; j++) {
			if (orientation[i] == j) {
				multi[j] = 0;
			}
			else {
				multi[j] = (2 * off[dim] - 1);
				dim++;
			}
		}
		edgeCenterPos[i].coords[0] = nodeCenter.coords[0] + multi[0] * halfWidth;
		edgeCenterPos[i].coords[1] = nodeCenter.coords[1] + multi[1] * halfWidth;
		edgeCenterPos[i].coords[2] = nodeCenter.coords[2] + multi[2] * halfWidth;
	}
#pragma unroll
	for (int i = 0; i < 12; i++) {
		for (int j = 0; j < 27; j++) {
			if (neighbor[j] != -1 && SquareDistance(edgeCenterPos[i], neighCenter[j]) < WidthSquare) {
				int neighKey = NodeArray[neighbor[j]].key;
				if (NodeOwnerKey[i] > neighKey) {
					NodeOwnerKey[i] = neighKey;
					NodeOwnerIdx[i] = neighbor[j];
				}
			}
		}
	}
#pragma unroll
	for (int i = 0; i < 12; i++) {
		int edgeIdx = 12 * idx + i;
		if (NodeOwnerIdx[i] == offset) {
			preEdgeArray[edgeIdx].ownerNodeIdx = offset;
			preEdgeArray[edgeIdx].edgeKind = i;
			markPreEdgeArray[edgeIdx] = true;
		}
		else {
			markPreEdgeArray[edgeIdx] = false;
		}
	}
}

__global__ void SparseSurfelFusion::device::maintainEdgeNodePointer(DeviceArrayView<unsigned int> DepthBuffer, DeviceArrayView<Point3D<float>> CenterBuffer, const unsigned int EdgeArraySize, EdgeNode* EdgeArray, OctNode* NodeArray)
{
	const unsigned int idx = threadIdx.x + blockDim.x * blockIdx.x;
	if (idx >= EdgeArraySize)	return;

	int owner = EdgeArray[idx].ownerNodeIdx;

	int depth = DepthBuffer[owner];
	float halfWidth = 1.0f / (1 << (depth + 1));
	float Width = 1.0f / (1 << depth);
	float WidthSquare = Width * Width;

	Point3D<float> neighCenter[27];
	int neigh[27];
	for (int i = 0; i < 27; i++) {
		neigh[i] = NodeArray[owner].neighs[i];
		if (neigh[i] != -1) {
			neighCenter[i] = CenterBuffer[neigh[i]];
		}
	}

	const Point3D<float>& nodeCenter = neighCenter[13];
	Point3D<float> edgeCenterPos;
	int multi[3];
	int dim = 0;
	int orientation = EdgeArray[idx].edgeKind >> 2;
	int off[2];
	off[0] = EdgeArray[idx].edgeKind & 1;
	off[1] = (EdgeArray[idx].edgeKind & 2) >> 1;
	for (int i = 0; i < 3; i++) {
		if (orientation == i) {
			multi[i] = 0;
		}
		else {
			multi[i] = (2 * off[dim] - 1);
			dim++;
		}
	}
	edgeCenterPos.coords[0] = nodeCenter.coords[0] + multi[0] * halfWidth;
	edgeCenterPos.coords[1] = nodeCenter.coords[1] + multi[1] * halfWidth;
	edgeCenterPos.coords[2] = nodeCenter.coords[2] + multi[2] * halfWidth;

	int count = 0;
	for (int i = 0; i < 27; ++i) {
		if (neigh[i] != -1 && SquareDistance(edgeCenterPos, neighCenter[i]) < WidthSquare) {
			EdgeArray[idx].nodes[count] = neigh[i];
			count++;
			int index = orientation << 2;
			int dim = 0;
			for (int j = 0; j < 3; j++) {
				if (orientation != j) {
					if (neighCenter[i].coords[j] - edgeCenterPos.coords[j] < 0) index |= (1 << dim);
					dim++;
				}
			}
			NodeArray[neigh[i]].edges[index] = idx + 1;
		}
	}
}

__global__ void SparseSurfelFusion::device::initFaceArray(DeviceArrayView<OctNode> NodeArray, DeviceArrayView<unsigned int> DepthBuffer, DeviceArrayView<Point3D<float>> CenterBuffer, const unsigned int NodeArraySize, FaceNode* preFaceArray, bool* markPreFaceArray)
{
	const unsigned int idx = threadIdx.x + blockDim.x * blockIdx.x;
	if (idx >= NodeArraySize)	return;
	int NodeOwnerKey[6] = { device::maxIntValue, device::maxIntValue, device::maxIntValue,
						    device::maxIntValue, device::maxIntValue, device::maxIntValue };
	int NodeOwnerIdx[6] = { -1,-1,-1,-1,-1,-1 };
	int nowDepth = DepthBuffer[idx];
	float halfWidth = 1.0f / (1 << (nowDepth + 1));
	float Width = 1.0f / (1 << nowDepth);
	float WidthSquare = Width * Width;
	Point3D<float> neighCenter[27];
	int neighbor[27];
#pragma unroll
	for (int i = 0; i < 27; i++) {
		neighbor[i] = NodeArray[idx].neighs[i];
		if (neighbor[i] != -1) {
			neighCenter[i] = CenterBuffer[neighbor[i]];
		}
	}
	const Point3D<float>& nodeCenter = neighCenter[13];

	Point3D<float> faceCenterPos[6];
	int orientation;
	int off;
	int multi;
	for (int i = 0; i < 6; i++) {
		orientation = i >> 1;
		off = i & 1;
		multi = (2 * off) - 1;
		faceCenterPos[i].coords[0] = nodeCenter.coords[0];
		faceCenterPos[i].coords[1] = nodeCenter.coords[1];
		faceCenterPos[i].coords[2] = nodeCenter.coords[2];
		faceCenterPos[i].coords[orientation] += multi * halfWidth;
	}

	for (int i = 0; i < 6; i++) {
		for (int j = 0; j < 27; j++) {
			if (neighbor[j] != -1 && SquareDistance(faceCenterPos[i], neighCenter[j]) < WidthSquare) {
				int neighKey = NodeArray[neighbor[j]].key;
				if (NodeOwnerKey[i] > neighKey) {
					NodeOwnerKey[i] = neighKey;
					NodeOwnerIdx[i] = neighbor[j];
				}
			}
		}
	}

	int parent = NodeArray[idx].parent;
	int sonKey = (NodeArray[idx].key >> (3 * (device::maxDepth - nowDepth))) & 7;
	for (int i = 0; i < 6; i++) {
		int faceIdx = 6 * idx + i;
		if (NodeOwnerIdx[i] == idx) {
			preFaceArray[faceIdx].ownerNodeIdx = idx;
			preFaceArray[faceIdx].faceKind = i;
			if (parent == -1) {
				preFaceArray[faceIdx].hasParentFace = -1;
			}
			else {
				if (device::parentFaceKind[sonKey][i] != -1) {
					preFaceArray[faceIdx].hasParentFace = 1;
				}
				else {
					preFaceArray[faceIdx].hasParentFace = -1;
				}
			}
			markPreFaceArray[faceIdx] = true;
		}
		else {
			markPreFaceArray[faceIdx] = false;
		}
	}
}

__global__ void SparseSurfelFusion::device::maintainFaceNodePointer(DeviceArrayView<unsigned int> DepthBuffer, DeviceArrayView<Point3D<float>> CenterBuffer, const unsigned int FaceArraySize, OctNode* NodeArray, FaceNode* FaceArray)
{
	const unsigned int idx = threadIdx.x + blockDim.x * blockIdx.x;
	if (idx >= FaceArraySize)	return;
	int owner = FaceArray[idx].ownerNodeIdx;
	int depth = DepthBuffer[owner];
	float halfWidth = 1.0f / (1 << (depth + 1));
	float Width = 1.0f / (1 << depth);
	float WidthSquare = Width * Width;

	Point3D<float> neighCenter[27];
	int neighbor[27];
	for (int i = 0; i < 27; i++) {
		neighbor[i] = NodeArray[owner].neighs[i];
		if (neighbor[i] != -1) {
			neighCenter[i] = CenterBuffer[neighbor[i]];
		}
	}

	const Point3D<float>& nodeCenter = neighCenter[13];
	Point3D<float> faceCenterPos;
	int kind = FaceArray[idx].faceKind;
	int orientation = kind >> 1;
	int off = kind & 1;
	int multi = (2 * off) - 1;

	faceCenterPos.coords[0] = nodeCenter.coords[0];
	faceCenterPos.coords[1] = nodeCenter.coords[1];
	faceCenterPos.coords[2] = nodeCenter.coords[2];
	faceCenterPos.coords[orientation] += multi * halfWidth;

	int count = 0;
	for (int i = 0; i < 27; i++) {
		if (neighbor[i] != -1 && SquareDistance(faceCenterPos, neighCenter[i]) < WidthSquare) {
			FaceArray[idx].nodes[count] = neighbor[i];
			count++;
			int index = orientation << 1;
			if (neighCenter[i].coords[orientation] - faceCenterPos.coords[orientation] < 0) {
				index++;
			}
			NodeArray[neighbor[i]].faces[index] = idx + 1;
		}
	}

}


void SparseSurfelFusion::BuildMeshGeometry::GenerateVertexArray(DeviceBufferArray<OctNode>& NodeArray, DeviceArrayView<unsigned int> NodeArrayDepthIndex, DeviceArrayView<Point3D<float>> NodeArrayNodeCenter, cudaStream_t stream)
{
#ifdef CHECK_MESH_BUILD_TIME_COST
	auto start = std::chrono::high_resolution_clock::now();						// 记录开始时间点
#endif // CHECK_MESH_BUILD_TIME_COST

	const unsigned int NodeArraySize = NodeArray.ArraySize();
	markValidVertexArray.ResizeArrayOrException(NodeArraySize * 8);
	VertexNode* preVertexArray = NULL;	//【中间变量】预先计算每个节点的Vertex数组，供以后筛选，占用内存巨大，内存不充足的情况下建议动态分配
	CHECKCUDA(cudaMallocAsync(reinterpret_cast<void**>(&preVertexArray), sizeof(VertexNode) * NodeArraySize * 8, stream));

	int* VertexArraySize = NULL;		// 【中间变量，用完即释放】获得压缩val后数量，理论上应该和valNums相等
	CHECKCUDA(cudaMallocAsync(reinterpret_cast<void**>(&VertexArraySize), sizeof(int), stream));

	dim3 block_1(128);
	dim3 grid_1(divUp(NodeArraySize, block_1.x));
	device::initVertexOwner << <grid_1, block_1, 0, stream >> > (NodeArray.ArrayView(), NodeArraySize, NodeArrayDepthIndex, NodeArrayNodeCenter, preVertexArray, markValidVertexArray.Array().ptr());

	void* d_temp_storage = NULL;
	size_t temp_storage_bytes = 0;
	CHECKCUDA(cub::DeviceSelect::Flagged(d_temp_storage, temp_storage_bytes, preVertexArray, markValidVertexArray.Array().ptr(), VertexArray.Array().ptr(), VertexArraySize, NodeArraySize * 8, stream, false));	// 确定临时设备存储需求
	CHECKCUDA(cudaMallocAsync(&d_temp_storage, temp_storage_bytes, stream));
	CHECKCUDA(cub::DeviceSelect::Flagged(d_temp_storage, temp_storage_bytes, preVertexArray, markValidVertexArray.Array().ptr(), VertexArray.Array().ptr(), VertexArraySize, NodeArraySize * 8, stream, false));	// 筛选	
	int VertexArraySizeHost = -1;
	CHECKCUDA(cudaMemcpyAsync(&VertexArraySizeHost, VertexArraySize, sizeof(int), cudaMemcpyDeviceToHost, stream));
	CHECKCUDA(cudaStreamSynchronize(stream));	// 流同步
	VertexArray.ResizeArrayOrException(VertexArraySizeHost);
	//printf("NodeArrayCount = %d\nVertexArrayCount = %d\n", NodeArraySize, VertexArraySizeHost);

	CHECKCUDA(cudaFreeAsync(d_temp_storage, stream));	// 释放中间遍历
	CHECKCUDA(cudaFreeAsync(VertexArraySize, stream));	// 释放中间遍历
	CHECKCUDA(cudaFreeAsync(preVertexArray, stream));	// 释放中间遍历

	dim3 block_2(128);
	dim3 grid_2(divUp(VertexArraySizeHost, block_2.x));
	device::maintainVertexNodePointer << <grid_2, block_2, 0, stream >> > (NodeArrayDepthIndex, NodeArrayNodeCenter, VertexArraySizeHost, VertexArray.Array().ptr(), NodeArray.Array().ptr());


#ifdef CHECK_MESH_BUILD_TIME_COST
	CHECKCUDA(cudaStreamSynchronize(stream));	// 流同步
	auto end = std::chrono::high_resolution_clock::now();						// 记录结束时间点
	std::chrono::duration<double, std::milli> duration = end - start;			// 计算执行时间（以ms为单位）
	std::cout << "生成顶点Vertex索引数组的时间: " << duration.count() << " ms" << std::endl;		// 输出
	std::cout << std::endl;
	std::cout << "-----------------------------------------------------" << std::endl;	// 输出
	std::cout << std::endl;
#endif // CHECK_MESH_BUILD_TIME_COST
}

void SparseSurfelFusion::BuildMeshGeometry::GenerateEdgeArray(DeviceBufferArray<OctNode>& NodeArray, const unsigned int DLevelOffset, const unsigned int DLevelNodeCount, DeviceArrayView<unsigned int> NodeArrayDepthIndex, DeviceArrayView<Point3D<float>> NodeArrayNodeCenter, cudaStream_t stream)
{
#ifdef CHECK_MESH_BUILD_TIME_COST
	auto start = std::chrono::high_resolution_clock::now();						// 记录开始时间点
#endif // CHECK_MESH_BUILD_TIME_COST

	//printf("BaseAddressArray = %d  NodeArrayCount = %d\n", DLevelOffset, DLevelNodeCount);

	EdgeNode* preEdgeArray = NULL;
	CHECKCUDA(cudaMallocAsync(reinterpret_cast<void**>(&preEdgeArray), sizeof(EdgeNode) * 12 * DLevelNodeCount, stream));

	int* EdgeArraySize = NULL;		// 【中间变量，用完即释放】获得压缩val后数量，理论上应该和valNums相等
	CHECKCUDA(cudaMallocAsync(reinterpret_cast<void**>(&EdgeArraySize), sizeof(int), stream));

	dim3 block_1(128);
	dim3 grid_1(divUp(DLevelNodeCount, block_1.x));
	device::initEdgeArray << <grid_1, block_1, 0, stream >> > (NodeArray.ArrayView(), DLevelOffset, DLevelNodeCount, NodeArrayDepthIndex, NodeArrayNodeCenter, preEdgeArray, markValidEdgeArray.Array().ptr());

	void* d_temp_storage = NULL;
	size_t temp_storage_bytes = 0;
	CHECKCUDA(cub::DeviceSelect::Flagged(d_temp_storage, temp_storage_bytes, preEdgeArray, markValidEdgeArray.Array().ptr(), EdgeArray.Array().ptr(), EdgeArraySize, DLevelNodeCount * 12, stream, false));	// 确定临时设备存储需求
	CHECKCUDA(cudaMallocAsync(&d_temp_storage, temp_storage_bytes, stream));			  
	CHECKCUDA(cub::DeviceSelect::Flagged(d_temp_storage, temp_storage_bytes, preEdgeArray, markValidEdgeArray.Array().ptr(), EdgeArray.Array().ptr(), EdgeArraySize, DLevelNodeCount * 12, stream, false));	// 筛选	
	int EdgeArraySizeHost = -1;
	CHECKCUDA(cudaMemcpyAsync(&EdgeArraySizeHost, EdgeArraySize, sizeof(int), cudaMemcpyDeviceToHost, stream));
	CHECKCUDA(cudaStreamSynchronize(stream));	// 流同步
	EdgeArray.ResizeArrayOrException(EdgeArraySizeHost);
	//printf("EdgeArrayCount = %d\n", EdgeArraySizeHost);

	CHECKCUDA(cudaFreeAsync(preEdgeArray, stream));		// 释放中间遍历
	CHECKCUDA(cudaFreeAsync(d_temp_storage, stream));	// 释放中间遍历
	CHECKCUDA(cudaFreeAsync(EdgeArraySize, stream));	// 释放中间遍历

	dim3 block_2(128);
	dim3 grid_2(divUp(EdgeArraySizeHost, block_2.x));
	device::maintainEdgeNodePointer << <grid_2, block_2, 0, stream >> > (NodeArrayDepthIndex, NodeArrayNodeCenter, EdgeArraySizeHost, EdgeArray.Array().ptr(), NodeArray.Array().ptr());


#ifdef CHECK_MESH_BUILD_TIME_COST
	CHECKCUDA(cudaStreamSynchronize(stream));	// 流同步
	auto end = std::chrono::high_resolution_clock::now();						// 记录结束时间点
	std::chrono::duration<double, std::milli> duration = end - start;			// 计算执行时间（以ms为单位）
	std::cout << "生成边Edge索引数组的时间: " << duration.count() << " ms" << std::endl;		// 输出
	std::cout << std::endl;
	std::cout << "-----------------------------------------------------" << std::endl;	// 输出
	std::cout << std::endl;
#endif // CHECK_MESH_BUILD_TIME_COST
}

void SparseSurfelFusion::BuildMeshGeometry::GenerateFaceArray(DeviceBufferArray<OctNode>& NodeArray, DeviceArrayView<unsigned int> NodeArrayDepthIndex, DeviceArrayView<Point3D<float>> NodeArrayNodeCenter, cudaStream_t stream)
{
#ifdef CHECK_MESH_BUILD_TIME_COST
	auto start = std::chrono::high_resolution_clock::now();						// 记录开始时间点
#endif // CHECK_MESH_BUILD_TIME_COST
	const unsigned int NodeArraySize = NodeArray.ArraySize();

	FaceNode* preFaceArray = NULL;
	CHECKCUDA(cudaMallocAsync(reinterpret_cast<void**>(&preFaceArray), sizeof(FaceNode) * 6 * NodeArraySize, stream));

	int* FaceArraySize = NULL;		// 【中间变量，用完即释放】获得压缩val后数量，理论上应该和valNums相等
	CHECKCUDA(cudaMallocAsync(reinterpret_cast<void**>(&FaceArraySize), sizeof(int), stream));

	dim3 block_1(128);
	dim3 grid_1(divUp(NodeArraySize, block_1.x));
	device::initFaceArray << <grid_1, block_1, 0, stream >> > (NodeArray.ArrayView(), NodeArrayDepthIndex, NodeArrayNodeCenter, NodeArraySize, preFaceArray, markValidFaceArray.Array().ptr());

	void* d_temp_storage = NULL;
	size_t temp_storage_bytes = 0;
	CHECKCUDA(cub::DeviceSelect::Flagged(d_temp_storage, temp_storage_bytes, preFaceArray, markValidFaceArray.Array().ptr(), FaceArray.Array().ptr(), FaceArraySize, NodeArraySize * 6, stream, false));	// 确定临时设备存储需求
	CHECKCUDA(cudaMallocAsync(&d_temp_storage, temp_storage_bytes, stream));
	CHECKCUDA(cub::DeviceSelect::Flagged(d_temp_storage, temp_storage_bytes, preFaceArray, markValidFaceArray.Array().ptr(), FaceArray.Array().ptr(), FaceArraySize, NodeArraySize * 6, stream, false));	// 筛选	
	int FaceArraySizeHost = -1;
	CHECKCUDA(cudaMemcpyAsync(&FaceArraySizeHost, FaceArraySize, sizeof(int), cudaMemcpyDeviceToHost, stream));
	CHECKCUDA(cudaStreamSynchronize(stream));	// 流同步
	FaceArray.ResizeArrayOrException(FaceArraySizeHost);
	//printf("FaceArraySizeHost = %d\n",FaceArraySizeHost);

	CHECKCUDA(cudaFreeAsync(preFaceArray, stream));		// 释放中间遍历
	CHECKCUDA(cudaFreeAsync(FaceArraySize, stream));	// 释放中间遍历
	CHECKCUDA(cudaFreeAsync(d_temp_storage, stream));	// 释放中间遍历

	dim3 block_2(128);
	dim3 grid_2(divUp(FaceArraySizeHost, block_2.x));
	device::maintainFaceNodePointer << <grid_2, block_2, 0, stream >> > (NodeArrayDepthIndex, NodeArrayNodeCenter, FaceArraySizeHost, NodeArray.Array().ptr(), FaceArray.Array().ptr());

#ifdef CHECK_MESH_BUILD_TIME_COST
	CHECKCUDA(cudaStreamSynchronize(stream));	// 流同步
	auto end = std::chrono::high_resolution_clock::now();						// 记录结束时间点
	std::chrono::duration<double, std::milli> duration = end - start;			// 计算执行时间（以ms为单位）
	std::cout << "生成面Face索引数组的时间: " << duration.count() << " ms" << std::endl;		// 输出
	std::cout << std::endl;
	std::cout << "-----------------------------------------------------" << std::endl;	// 输出
	std::cout << std::endl;
#endif // CHECK_MESH_BUILD_TIME_COST
}

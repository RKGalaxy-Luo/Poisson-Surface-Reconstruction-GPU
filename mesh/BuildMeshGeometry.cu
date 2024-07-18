/*****************************************************************//**
 * \file   BuildMeshGeometry.cu
 * \brief  ��������cuda����ʵ��
 * 
 * \author LUOJIAXUAN
 * \date   June 1st 2024
 *********************************************************************/
#include "BuildMeshGeometry.h"
#if defined(__CUDACC__)		//�����NVCC����������
#include <cub/cub.cuh>
#endif

namespace SparseSurfelFusion {
	namespace device {
		__device__ __constant__ int maxIntValue = 0x7fffffff;		// ���intֵ

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
							device::maxIntValue, device::maxIntValue, device::maxIntValue, device::maxIntValue };	// ��ʼ��Чֵ
	int NodeOwnerIdx[8] = { -1, -1, -1, -1, -1, -1, -1, -1 };				// ��ʼ��Чֵ
	int depth = depthBuffer[idx];
	float halfWidth = 1.0f / (1 << (depth + 1));							// �ڵ�����һ��Ŀ�
	float Width = 1.0f / (1 << depth);										// �ڵ����صĿ�
	float WidthSquare = Width * Width;										// �ڵ����ؿ��ƽ��
	Point3D<float> neighborCenter[27] = { Point3D<float>(0, 0, 0) };		// �ڵ��27���ھӽڵ������λ��
	int neighbor[27] = { -1 };														// �ڵ��27���ھӵ�index
#pragma unroll
	for (int i = 0; i < 27; i++) {
		neighbor[i] = NodeArray[idx].neighs[i];
		if (neighbor[i] != -1) {	// �ھӲ�Ϊ��
			neighborCenter[i] = centerBuffer[neighbor[i]];
		}
	}
	const Point3D<float>& nodeCenter = neighborCenter[13];	// ���Ľڵ���ǵ�ǰ�ڵ��Լ�

	Point3D<float> vertexPos[8];	// ����λ�ã���ǽڵ����صĶ���λ�ã�һ���ڵ� = һ�����ؿ飬һ�����ؿ�(������)��8������
#pragma unroll
	for (int i = 0; i < 8; i++) {	// ���㶥��λ��(�����˳���ǣ������˳���ǣ���������Ϊ�ο���x��ǰ����y�����ң�z���µ���)
		vertexPos[i].coords[0] = nodeCenter.coords[0] + (2 * (i & 1) - 1) * halfWidth;
		vertexPos[i].coords[1] = nodeCenter.coords[1] + (2 * ((i & 2) >> 1) - 1) * halfWidth;
		vertexPos[i].coords[2] = nodeCenter.coords[2] + (2 * ((i & 4) >> 2) - 1) * halfWidth;
	}

#pragma unroll
	for (int i = 0; i < 8; i++) {			// Ϊÿ��������һ����Ӧ�Ľڵ�
		for (int j = 0; j < 27; j++) {		// �����ڵ㼰�ھ�(��27���ڵ�)�������Ӧkey��С�Ľڵ㣬��key��С�Ľڵ�ӵ�����vertex
			if ((neighbor[j] != -1) && (device::SquareDistance(vertexPos[i], neighborCenter[j]) < WidthSquare)) { // �ھӽڵ������Ч��ӵ�����vertex�Ľڵ㲻���Գ���һ�����صĿ�
				int neighborKey = NodeArray[neighbor[j]].key;
				if (NodeOwnerKey[i] > neighborKey) {	// ���neighborKey��С
					NodeOwnerKey[i] = neighborKey;
					NodeOwnerIdx[i] = neighbor[j];		// ������ڵ��index��NodeOwnerIdx
				}
			}
		}
	}
#pragma unroll
	for (int i = 0; i < 8; i++) {		// ������8������
		int vertexIdx = 8 * idx + i;	// ��preVertexArrayÿ��8����һ�Σ����Ҷ�Ӧ��λ����8���ڵ��еĵڼ���
		if (NodeOwnerIdx[i] == idx) {	// ������������NodeArray������ڵ���ӵ��
			preVertexArray[vertexIdx].ownerNodeIdx = idx;						// vertex����ǰ���NodeArray�еĽڵ�ӵ��
			preVertexArray[vertexIdx].pos.coords[0] = vertexPos[i].coords[0];	// ��vertex��λ�ø�ֵ��pos
			preVertexArray[vertexIdx].pos.coords[1] = vertexPos[i].coords[1];	// ��vertex��λ�ø�ֵ��pos
			preVertexArray[vertexIdx].pos.coords[2] = vertexPos[i].coords[2];	// ��vertex��λ�ø�ֵ��pos
			preVertexArray[vertexIdx].vertexKind = i;							// ������ļ��Ŷ���
			preVertexArray[vertexIdx].depth = depth;							// ��ǰvertex����� = ��ǰ�ڵ�����
			markPreVertexArray[vertexIdx] = true;
		}
		else {
			markPreVertexArray[vertexIdx] = false;	// ������Ч��ȫ�����Ϊfalse
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
	int owner = VertexArray[idx].ownerNodeIdx;			// ӵ�д˶����Node
	int depth = DepthBuffer[owner];						// ��ǰ�����Ӧ�Ľڵ�����
	float halfWidth = 1.0f / (1 << (depth + 1));
	float Width = 1.0f / (1 << depth);
	float WidthSquare = Width * Width;
	Point3D<float> neighCenter[27];						// �ھӵ����ĵ�
	Point3D<float> vertexPos = VertexArray[idx].pos;	// ��ǰ��������λ��

	int neighbor[27];									// Owner�ڵ���ھӽڵ�
	for (int i = 0; i < 27; i++) {
		neighbor[i] = NodeArray[owner].neighs[i];
		if (neighbor[i] != -1) {						// �ھ�����Ч��
			neighCenter[i] = CenterBuffer[neighbor[i]];
		}
	}
	int count = 0;
	for (int i = 0; i < 27; i++) {
		// �ھ���Ч��������������ھӷǿ�һ���ھӵ��ھ�
		if (neighbor[i] != -1 && SquareDistance(vertexPos, neighCenter[i]) < WidthSquare) {
			VertexArray[idx].nodes[count] = neighbor[i];
			count++;
			int index = 0;	// ��¼��������ڽڵ��λ�ñ�ţ�x�Ⱥ���ǰ��y�������ң�z��������
			if (neighCenter[i].coords[0] - vertexPos.coords[0] < 0) index |= 1;		// ��x���򿴣��ھӽڵ���vertexλ�õĺ���
			if (neighCenter[i].coords[2] - vertexPos.coords[2] < 0) index |= 4;		// ��z���򿴣��ھӽڵ���vertexλ�õ�����
			if (neighCenter[i].coords[1] - vertexPos.coords[1] < 0) {				// ��y���򿴣��ھӽڵ���vertexλ�õ����
				if (index & 1) {
					index += 1;
				}
				else {
					index += 3;
				}
			}
			NodeArray[neighbor[i]].vertices[index] = idx + 1;	// �����Vertice����д�룬����Ӱ���NodeArray��OctNode����������ֵ���ж�ȡ����˿��Բ���
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
	auto start = std::chrono::high_resolution_clock::now();						// ��¼��ʼʱ���
#endif // CHECK_MESH_BUILD_TIME_COST

	const unsigned int NodeArraySize = NodeArray.ArraySize();
	markValidVertexArray.ResizeArrayOrException(NodeArraySize * 8);
	VertexNode* preVertexArray = NULL;	//���м������Ԥ�ȼ���ÿ���ڵ��Vertex���飬���Ժ�ɸѡ��ռ���ڴ�޴��ڴ治���������½��鶯̬����
	CHECKCUDA(cudaMallocAsync(reinterpret_cast<void**>(&preVertexArray), sizeof(VertexNode) * NodeArraySize * 8, stream));

	int* VertexArraySize = NULL;		// ���м���������꼴�ͷš����ѹ��val��������������Ӧ�ú�valNums���
	CHECKCUDA(cudaMallocAsync(reinterpret_cast<void**>(&VertexArraySize), sizeof(int), stream));

	dim3 block_1(128);
	dim3 grid_1(divUp(NodeArraySize, block_1.x));
	device::initVertexOwner << <grid_1, block_1, 0, stream >> > (NodeArray.ArrayView(), NodeArraySize, NodeArrayDepthIndex, NodeArrayNodeCenter, preVertexArray, markValidVertexArray.Array().ptr());

	void* d_temp_storage = NULL;
	size_t temp_storage_bytes = 0;
	CHECKCUDA(cub::DeviceSelect::Flagged(d_temp_storage, temp_storage_bytes, preVertexArray, markValidVertexArray.Array().ptr(), VertexArray.Array().ptr(), VertexArraySize, NodeArraySize * 8, stream, false));	// ȷ����ʱ�豸�洢����
	CHECKCUDA(cudaMallocAsync(&d_temp_storage, temp_storage_bytes, stream));
	CHECKCUDA(cub::DeviceSelect::Flagged(d_temp_storage, temp_storage_bytes, preVertexArray, markValidVertexArray.Array().ptr(), VertexArray.Array().ptr(), VertexArraySize, NodeArraySize * 8, stream, false));	// ɸѡ	
	int VertexArraySizeHost = -1;
	CHECKCUDA(cudaMemcpyAsync(&VertexArraySizeHost, VertexArraySize, sizeof(int), cudaMemcpyDeviceToHost, stream));
	CHECKCUDA(cudaStreamSynchronize(stream));	// ��ͬ��
	VertexArray.ResizeArrayOrException(VertexArraySizeHost);
	//printf("NodeArrayCount = %d\nVertexArrayCount = %d\n", NodeArraySize, VertexArraySizeHost);

	CHECKCUDA(cudaFreeAsync(d_temp_storage, stream));	// �ͷ��м����
	CHECKCUDA(cudaFreeAsync(VertexArraySize, stream));	// �ͷ��м����
	CHECKCUDA(cudaFreeAsync(preVertexArray, stream));	// �ͷ��м����

	dim3 block_2(128);
	dim3 grid_2(divUp(VertexArraySizeHost, block_2.x));
	device::maintainVertexNodePointer << <grid_2, block_2, 0, stream >> > (NodeArrayDepthIndex, NodeArrayNodeCenter, VertexArraySizeHost, VertexArray.Array().ptr(), NodeArray.Array().ptr());


#ifdef CHECK_MESH_BUILD_TIME_COST
	CHECKCUDA(cudaStreamSynchronize(stream));	// ��ͬ��
	auto end = std::chrono::high_resolution_clock::now();						// ��¼����ʱ���
	std::chrono::duration<double, std::milli> duration = end - start;			// ����ִ��ʱ�䣨��msΪ��λ��
	std::cout << "���ɶ���Vertex���������ʱ��: " << duration.count() << " ms" << std::endl;		// ���
	std::cout << std::endl;
	std::cout << "-----------------------------------------------------" << std::endl;	// ���
	std::cout << std::endl;
#endif // CHECK_MESH_BUILD_TIME_COST
}

void SparseSurfelFusion::BuildMeshGeometry::GenerateEdgeArray(DeviceBufferArray<OctNode>& NodeArray, const unsigned int DLevelOffset, const unsigned int DLevelNodeCount, DeviceArrayView<unsigned int> NodeArrayDepthIndex, DeviceArrayView<Point3D<float>> NodeArrayNodeCenter, cudaStream_t stream)
{
#ifdef CHECK_MESH_BUILD_TIME_COST
	auto start = std::chrono::high_resolution_clock::now();						// ��¼��ʼʱ���
#endif // CHECK_MESH_BUILD_TIME_COST

	//printf("BaseAddressArray = %d  NodeArrayCount = %d\n", DLevelOffset, DLevelNodeCount);

	EdgeNode* preEdgeArray = NULL;
	CHECKCUDA(cudaMallocAsync(reinterpret_cast<void**>(&preEdgeArray), sizeof(EdgeNode) * 12 * DLevelNodeCount, stream));

	int* EdgeArraySize = NULL;		// ���м���������꼴�ͷš����ѹ��val��������������Ӧ�ú�valNums���
	CHECKCUDA(cudaMallocAsync(reinterpret_cast<void**>(&EdgeArraySize), sizeof(int), stream));

	dim3 block_1(128);
	dim3 grid_1(divUp(DLevelNodeCount, block_1.x));
	device::initEdgeArray << <grid_1, block_1, 0, stream >> > (NodeArray.ArrayView(), DLevelOffset, DLevelNodeCount, NodeArrayDepthIndex, NodeArrayNodeCenter, preEdgeArray, markValidEdgeArray.Array().ptr());

	void* d_temp_storage = NULL;
	size_t temp_storage_bytes = 0;
	CHECKCUDA(cub::DeviceSelect::Flagged(d_temp_storage, temp_storage_bytes, preEdgeArray, markValidEdgeArray.Array().ptr(), EdgeArray.Array().ptr(), EdgeArraySize, DLevelNodeCount * 12, stream, false));	// ȷ����ʱ�豸�洢����
	CHECKCUDA(cudaMallocAsync(&d_temp_storage, temp_storage_bytes, stream));			  
	CHECKCUDA(cub::DeviceSelect::Flagged(d_temp_storage, temp_storage_bytes, preEdgeArray, markValidEdgeArray.Array().ptr(), EdgeArray.Array().ptr(), EdgeArraySize, DLevelNodeCount * 12, stream, false));	// ɸѡ	
	int EdgeArraySizeHost = -1;
	CHECKCUDA(cudaMemcpyAsync(&EdgeArraySizeHost, EdgeArraySize, sizeof(int), cudaMemcpyDeviceToHost, stream));
	CHECKCUDA(cudaStreamSynchronize(stream));	// ��ͬ��
	EdgeArray.ResizeArrayOrException(EdgeArraySizeHost);
	//printf("EdgeArrayCount = %d\n", EdgeArraySizeHost);

	CHECKCUDA(cudaFreeAsync(preEdgeArray, stream));		// �ͷ��м����
	CHECKCUDA(cudaFreeAsync(d_temp_storage, stream));	// �ͷ��м����
	CHECKCUDA(cudaFreeAsync(EdgeArraySize, stream));	// �ͷ��м����

	dim3 block_2(128);
	dim3 grid_2(divUp(EdgeArraySizeHost, block_2.x));
	device::maintainEdgeNodePointer << <grid_2, block_2, 0, stream >> > (NodeArrayDepthIndex, NodeArrayNodeCenter, EdgeArraySizeHost, EdgeArray.Array().ptr(), NodeArray.Array().ptr());


#ifdef CHECK_MESH_BUILD_TIME_COST
	CHECKCUDA(cudaStreamSynchronize(stream));	// ��ͬ��
	auto end = std::chrono::high_resolution_clock::now();						// ��¼����ʱ���
	std::chrono::duration<double, std::milli> duration = end - start;			// ����ִ��ʱ�䣨��msΪ��λ��
	std::cout << "���ɱ�Edge���������ʱ��: " << duration.count() << " ms" << std::endl;		// ���
	std::cout << std::endl;
	std::cout << "-----------------------------------------------------" << std::endl;	// ���
	std::cout << std::endl;
#endif // CHECK_MESH_BUILD_TIME_COST
}

void SparseSurfelFusion::BuildMeshGeometry::GenerateFaceArray(DeviceBufferArray<OctNode>& NodeArray, DeviceArrayView<unsigned int> NodeArrayDepthIndex, DeviceArrayView<Point3D<float>> NodeArrayNodeCenter, cudaStream_t stream)
{
#ifdef CHECK_MESH_BUILD_TIME_COST
	auto start = std::chrono::high_resolution_clock::now();						// ��¼��ʼʱ���
#endif // CHECK_MESH_BUILD_TIME_COST
	const unsigned int NodeArraySize = NodeArray.ArraySize();

	FaceNode* preFaceArray = NULL;
	CHECKCUDA(cudaMallocAsync(reinterpret_cast<void**>(&preFaceArray), sizeof(FaceNode) * 6 * NodeArraySize, stream));

	int* FaceArraySize = NULL;		// ���м���������꼴�ͷš����ѹ��val��������������Ӧ�ú�valNums���
	CHECKCUDA(cudaMallocAsync(reinterpret_cast<void**>(&FaceArraySize), sizeof(int), stream));

	dim3 block_1(128);
	dim3 grid_1(divUp(NodeArraySize, block_1.x));
	device::initFaceArray << <grid_1, block_1, 0, stream >> > (NodeArray.ArrayView(), NodeArrayDepthIndex, NodeArrayNodeCenter, NodeArraySize, preFaceArray, markValidFaceArray.Array().ptr());

	void* d_temp_storage = NULL;
	size_t temp_storage_bytes = 0;
	CHECKCUDA(cub::DeviceSelect::Flagged(d_temp_storage, temp_storage_bytes, preFaceArray, markValidFaceArray.Array().ptr(), FaceArray.Array().ptr(), FaceArraySize, NodeArraySize * 6, stream, false));	// ȷ����ʱ�豸�洢����
	CHECKCUDA(cudaMallocAsync(&d_temp_storage, temp_storage_bytes, stream));
	CHECKCUDA(cub::DeviceSelect::Flagged(d_temp_storage, temp_storage_bytes, preFaceArray, markValidFaceArray.Array().ptr(), FaceArray.Array().ptr(), FaceArraySize, NodeArraySize * 6, stream, false));	// ɸѡ	
	int FaceArraySizeHost = -1;
	CHECKCUDA(cudaMemcpyAsync(&FaceArraySizeHost, FaceArraySize, sizeof(int), cudaMemcpyDeviceToHost, stream));
	CHECKCUDA(cudaStreamSynchronize(stream));	// ��ͬ��
	FaceArray.ResizeArrayOrException(FaceArraySizeHost);
	//printf("FaceArraySizeHost = %d\n",FaceArraySizeHost);

	CHECKCUDA(cudaFreeAsync(preFaceArray, stream));		// �ͷ��м����
	CHECKCUDA(cudaFreeAsync(FaceArraySize, stream));	// �ͷ��м����
	CHECKCUDA(cudaFreeAsync(d_temp_storage, stream));	// �ͷ��м����

	dim3 block_2(128);
	dim3 grid_2(divUp(FaceArraySizeHost, block_2.x));
	device::maintainFaceNodePointer << <grid_2, block_2, 0, stream >> > (NodeArrayDepthIndex, NodeArrayNodeCenter, FaceArraySizeHost, NodeArray.Array().ptr(), FaceArray.Array().ptr());

#ifdef CHECK_MESH_BUILD_TIME_COST
	CHECKCUDA(cudaStreamSynchronize(stream));	// ��ͬ��
	auto end = std::chrono::high_resolution_clock::now();						// ��¼����ʱ���
	std::chrono::duration<double, std::milli> duration = end - start;			// ����ִ��ʱ�䣨��msΪ��λ��
	std::cout << "������Face���������ʱ��: " << duration.count() << " ms" << std::endl;		// ���
	std::cout << std::endl;
	std::cout << "-----------------------------------------------------" << std::endl;	// ���
	std::cout << std::endl;
#endif // CHECK_MESH_BUILD_TIME_COST
}

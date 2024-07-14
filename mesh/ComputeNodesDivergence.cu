/*****************************************************************//**
 * \file   ComputeNodesDivergence.cu
 * \brief  ����ڵ�ɢ��cuda����ʵ��
 * 
 * \author LUOJIAXUAN
 * \date   May 24th 2024
 *********************************************************************/
#include "ComputeNodesDivergence.h"
#if defined(__CUDACC__)		//�����NVCC����������
#include <cub/cub.cuh>
#endif

namespace SparseSurfelFusion {
	namespace device {
		__device__ __constant__ int maxDepth = MAX_DEPTH_OCTREE;	// Octree������

		__device__ __constant__ int res = RESOLUTION;				// �ֱ���

		__device__ __constant__ int decodeOffset_1 = (1 << (MAX_DEPTH_OCTREE + 1));

		__device__ __constant__ int decodeOffset_2 = (1 << (2 * (MAX_DEPTH_OCTREE + 1)));
	}
}

__global__ void SparseSurfelFusion::device::computeFinerNodesDivergenceKernel(DeviceArrayView<int> BaseAddressArray, DeviceArrayView<int> encodeNodeIndexInFunction, DeviceArrayView<OctNode> NodeArray, DeviceArrayView<Point3D<float>> VectorField, DeviceArrayView<double> dot_F_DF, const unsigned int begin, const unsigned int calculatedNodeNum, float* Divergence)
{
	const unsigned int idx = threadIdx.x + blockDim.x * blockIdx.x;
	if (idx >= calculatedNodeNum)	return;
	const unsigned int offset = begin + idx;
	const unsigned int startDLevel = BaseAddressArray[device::maxDepth];
	double val = 0;
#pragma unroll
	for (int i = 0; i < 27; i++) {
		int neighborIdx = NodeArray[offset].neighs[i];
		if (neighborIdx == -1)	continue;
		for (int j = 0; j < NodeArray[neighborIdx].dnum; j++) {						// ������ǰ�ڵ���ھӽڵ���maxDepth���������ĵ�����Ҷ�ӽڵ�
			int NodeIndexDLevel = NodeArray[neighborIdx].didx + j;					// ��maxDepth��Ҷ�ӽڵ��index
			const Point3D<float>& vo = VectorField[NodeIndexDLevel];
			int idxO_1[3], idxO_2[3];

			int encodeIndex = encodeNodeIndexInFunction[offset];						// ��õ�ǰ�ڵ����������
			idxO_1[0] = encodeIndex % decodeOffset_1;								// ȡ�������11λ	[0 , 10]
			idxO_1[1] = (encodeIndex / decodeOffset_1) % decodeOffset_1;			// ȡ�����м�11λ	[11, 21]
			idxO_1[2] = encodeIndex / decodeOffset_2;								// ȡ������ǰ10λ	[22, 31]

			encodeIndex = encodeNodeIndexInFunction[startDLevel + NodeIndexDLevel];	// ��õ�ǰ�ڵ���maxdepth���Ҷ�ӽڵ����������
			idxO_2[0] = encodeIndex % decodeOffset_1;								// ȡ�������11λ	[0 , 10]
			idxO_2[1] = (encodeIndex / decodeOffset_1) % decodeOffset_1;			// ȡ�����м�11λ	[11, 21]
			idxO_2[2] = encodeIndex / decodeOffset_2;								// ȡ������ǰ10λ	[22, 31]

			int scratch[3];
			scratch[0] = idxO_1[0] + idxO_2[0] * res;
			scratch[1] = idxO_1[1] + idxO_2[1] * res;
			scratch[2] = idxO_1[2] + idxO_2[2] * res;

			Point3D<float> uo;
			uo.coords[0] = dot_F_DF[scratch[0]];
			uo.coords[1] = dot_F_DF[scratch[1]];
			uo.coords[2] = dot_F_DF[scratch[2]];

			val += DotProduct(vo, uo);
		}
	}
	Divergence[offset] = val;	// ��ǰ�ڵ�ɢ��
}

__device__ float SparseSurfelFusion::device::DotProduct(const Point3D<float>& p1, const Point3D<float>& p2)
{
	float ans = 0;
	ans += p1.coords[0] * p2.coords[0];
	ans += p1.coords[1] * p2.coords[1];
	ans += p1.coords[2] * p2.coords[2];
	return ans;
}

__global__ void SparseSurfelFusion::device::computeCoverNums(DeviceArrayView<OctNode> NodeArray, const unsigned int index, unsigned int* coverNums)
{
	coverNums[0] = 0;
	for (int i = 0; i < 27; i++) {
		int neighbor = NodeArray[index].neighs[i];	// �ھӽڵ�idx
		if (neighbor != -1) {
			coverNums[i + 1] = NodeArray[neighbor].dnum + coverNums[i];
		}
		else {
			coverNums[i + 1] = coverNums[i];
		}
	}
}

__global__ void SparseSurfelFusion::device::generateDLevelIndexArrayKernel(DeviceArrayView<OctNode> NodeArray, const unsigned int index, const unsigned int* coverNums, const unsigned int totalCoverNum, unsigned int* DLevelIndexArray)
{
	const unsigned int idx = threadIdx.x + blockDim.x * blockIdx.x;
	if (idx >= totalCoverNum)	return;

	int neighborIdx;	// �жϵ�ǰ���D��ڵ���������һ���ϲ�ڵ�(1-4����ڵ㼰���ھӵ�27����)
	for (neighborIdx = 0; neighborIdx < 27; neighborIdx++) {
		if (coverNums[neighborIdx] <= idx && idx < coverNums[neighborIdx + 1]) {
			break;
		}
	}
	int Current27NodesDLevelStartIndex = NodeArray[NodeArray[index].neighs[neighborIdx]].didx;	// ��ǰ���neighborIdx�ڵ���D���е���ʼλ��
	DLevelIndexArray[idx] = Current27NodesDLevelStartIndex + idx - coverNums[neighborIdx];		// idx - coverNums[neighborIdx]��������ڵ�ǰneighborIdx�ڵ���ʵλ�õľ���
}

__global__ void SparseSurfelFusion::device::computeCoarserNodesDivergenceKernel(DeviceArrayView<int> BaseAddressArrayDevice, DeviceArrayView<int> encodeNodeIndexInFunction, DeviceArrayView<Point3D<float>> VectorField, DeviceArrayView<double> dot_F_DF, const unsigned int index, const unsigned int* DLevelIndexArray, const unsigned int totalCoverNum, float* divg)
{
	const unsigned int idx = threadIdx.x + blockDim.x * blockIdx.x;
	if (idx >= totalCoverNum)	return;
	const unsigned int startD = BaseAddressArrayDevice[device::maxDepth];
	int DIdx = DLevelIndexArray[idx];
	const Point3D<float>& vo = VectorField[DIdx];

	int idxO_1[3], idxO_2[3];

	int encodeIdx = encodeNodeIndexInFunction[index];
	idxO_1[0] = encodeIdx % decodeOffset_1;
	idxO_1[1] = (encodeIdx / decodeOffset_1) % decodeOffset_1;
	idxO_1[2] = encodeIdx / decodeOffset_2;

	encodeIdx = encodeNodeIndexInFunction[startD + DIdx];
	idxO_2[0] = encodeIdx % decodeOffset_1;
	idxO_2[1] = (encodeIdx / decodeOffset_1) % decodeOffset_1;
	idxO_2[2] = encodeIdx / decodeOffset_2;

	int scratch[3];
	scratch[0] = idxO_1[0] + idxO_2[0] * device::res;
	scratch[1] = idxO_1[1] + idxO_2[1] * device::res;
	scratch[2] = idxO_1[2] + idxO_2[2] * device::res;

	Point3D<float> uo;
	uo.coords[0] = dot_F_DF[scratch[0]];
	uo.coords[1] = dot_F_DF[scratch[1]];
	uo.coords[2] = dot_F_DF[scratch[2]];

	divg[idx] = DotProduct(vo, uo);
}

void SparseSurfelFusion::ComputeNodesDivergence::computeFinerNodesDivergence(DeviceArrayView<int> BaseAddressArrayDevice, DeviceArrayView<int> encodeNodeIndexInFunction, DeviceArrayView<OctNode> NodeArray, DeviceArrayView<Point3D<float>> VectorField, DeviceArrayView<double> dot_F_DF, const unsigned int left, const unsigned int right, cudaStream_t stream)
{
//#ifdef CHECK_MESH_BUILD_TIME_COST
//	auto start = std::chrono::high_resolution_clock::now();						// ��¼��ʼʱ���
//#endif // CHECK_MESH_BUILD_TIME_COST

	Divergence.ResizeArrayOrException(NodeArray.Size());	// ��NodeArray��Сһ��
	const unsigned int CalculatedNodeNum = right - left;	// �������Ľڵ�����
	dim3 block(128);
	dim3 grid(divUp(CalculatedNodeNum, block.x));
	device::computeFinerNodesDivergenceKernel << <grid, block, 0, stream >> > (BaseAddressArrayDevice, encodeNodeIndexInFunction, NodeArray, VectorField, dot_F_DF, left, CalculatedNodeNum, Divergence.Array().ptr());

//#ifdef CHECK_MESH_BUILD_TIME_COST
//	CHECKCUDA(cudaStreamSynchronize(stream));
//	auto end = std::chrono::high_resolution_clock::now();							// ��¼����ʱ���
//	std::chrono::duration<double, std::milli> duration = end - start;				// ����ִ��ʱ�䣨��msΪ��λ��
//	std::cout << "����[CoarserLevelNum + 1, maxDepth]��ڵ�ɢ�ȵ�ʱ��: " << duration.count() << " ms" << std::endl;	// ���
//#endif // CHECK_MESH_BUILD_TIME_COST

}

void SparseSurfelFusion::ComputeNodesDivergence::computeCoarserNodesDivergence(const int* BaseAddressArray, DeviceArrayView<int> BaseAddressArrayDevice, DeviceArrayView<int> encodeNodeIndexInFunction, DeviceArrayView<OctNode> NodeArray, DeviceArrayView<Point3D<float>> VectorField, DeviceArrayView<double> dot_F_DF, const unsigned int left, const unsigned int right, cudaStream_t stream)
{
//#ifdef CHECK_MESH_BUILD_TIME_COST
//	auto start = std::chrono::high_resolution_clock::now();						// ��¼��ʼʱ���
//#endif // CHECK_MESH_BUILD_TIME_COST

	Divergence.ResizeArrayOrException(NodeArray.Size());	// ��NodeArray��Сһ��
	const unsigned int CalculatedNodeNum = right - left;	// �������Ľڵ�����
	for (int i = left; i < CalculatedNodeNum; i++) {			// [0, CoarserLevelNum]��ڵ�����
		int depth = 0;	// ��ǰ�ڵ����
		for (int j = 0; j <= COARSER_DIVERGENCE_LEVEL_NUM; j++) {
			if (BaseAddressArray[j] <= i && i < BaseAddressArray[j + 1]) {
				depth = j;
				break;
			}
		}
		unsigned int* coverNums = NULL;
		CHECKCUDA(cudaMallocAsync(reinterpret_cast<void**>(&coverNums), sizeof(unsigned int) * 28, stream));

		device::computeCoverNums << <1, 1, 0, stream >> > (NodeArray, i, coverNums);
		unsigned int totalCoverNum;		// ��ǰ�ڵ㼰���ھӸ��ǵ�maxDepth��Ľڵ�����
		CHECKCUDA(cudaMemcpyAsync(&totalCoverNum, coverNums + 27, sizeof(unsigned int), cudaMemcpyDeviceToHost, stream));
		
		float* divg = NULL;
		CHECKCUDA(cudaMallocAsync(reinterpret_cast<void**>(&divg), sizeof(float) * totalCoverNum, stream));

		unsigned int* DLevelIndexArray = NULL;	// ����idx = [0, totalCoverNum)��D��ڵ㣬����Щ�ڵ�ӳ�䵽D���Ӧ��D_idx�ϡ����磺idx = 3��D��ڵ���NodeArray�е�λ�ÿ���Ϊ100��
		CHECKCUDA(cudaMallocAsync(reinterpret_cast<void**>(&DLevelIndexArray), sizeof(unsigned int) * totalCoverNum, stream));

		dim3 block(128);
		dim3 grid(divUp(totalCoverNum, block.x));
		device::generateDLevelIndexArrayKernel << <grid, block, 0, stream >> > (NodeArray, i, coverNums, totalCoverNum, DLevelIndexArray);
		device::computeCoarserNodesDivergenceKernel << <grid, block, 0, stream >> > (BaseAddressArrayDevice, encodeNodeIndexInFunction, VectorField, dot_F_DF, i, DLevelIndexArray, totalCoverNum, divg);
		// ��Լ�ӷ�
		float* divgSum = NULL;
		CHECKCUDA(cudaMallocAsync(reinterpret_cast<void**>(&divgSum), sizeof(float), stream));

		void* d_temp_storage = NULL;
		size_t temp_storage_bytes = 0;
		cub::DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, divg, divgSum, totalCoverNum, stream);
		CHECKCUDA(cudaMallocAsync(&d_temp_storage, temp_storage_bytes, stream));
		cub::DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, divg, divgSum, totalCoverNum, stream);
		CHECKCUDA(cudaMemcpyAsync(Divergence.Array().ptr() + i, divgSum, sizeof(float), cudaMemcpyDeviceToDevice, stream));
		CHECKCUDA(cudaFreeAsync(coverNums, stream));			// ��ʱ����, ���꼴ɾ
		CHECKCUDA(cudaFreeAsync(divg, stream));					// ��ʱ����, ���꼴ɾ
		CHECKCUDA(cudaFreeAsync(DLevelIndexArray, stream));		// ��ʱ����, ���꼴ɾ
		CHECKCUDA(cudaFreeAsync(divgSum, stream));				// ��ʱ����, ���꼴ɾ
		CHECKCUDA(cudaFreeAsync(d_temp_storage, stream));		// ��ʱ����, ���꼴ɾ
	}

//#ifdef CHECK_MESH_BUILD_TIME_COST
//	// ���в��������ͬ��
//	for (int i = 0; i < MAX_MESH_STREAM - 1; i++) {
//		CHECKCUDA(cudaStreamSynchronize(streams[i]));
//	}
//	auto end = std::chrono::high_resolution_clock::now();							// ��¼����ʱ���
//	std::chrono::duration<double, std::milli> duration = end - start;				// ����ִ��ʱ�䣨��msΪ��λ��
//	std::cout << "����[1, CoarserLevelNum]��ڵ�ɢ�ȵ�ʱ��: " << duration.count() << " ms" << std::endl;	// ���
//#endif // CHECK_MESH_BUILD_TIME_COST
}

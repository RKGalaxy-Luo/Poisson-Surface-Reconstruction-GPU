/*****************************************************************//**
 * \file   BuildOctree.cu
 * \brief  ����˲���cuda����ʵ��
 * 
 * \author LUOJIAXUAN
 * \date   May 5th 2024
 *********************************************************************/
#include "BuildOctree.h"

namespace SparseSurfelFusion {
	namespace device {
		__device__ __constant__ int BaseAddressArray_Device[Constants::maxDepth_Host + 1] = { 0 };	// ��¼ÿ��ڵ��׵�ַ��λ��

		__device__ __constant__ double eps = EPSILON;

		__device__ __constant__ int maxDepth = MAX_DEPTH_OCTREE;

		__device__ __constant__ int maxIntValue = 0x7fffffff;		// ���intֵ

		__device__ __constant__ int LUTparent[8][27] =
		{
			{0,1,1,3,4,4,3,4,4,9,10,10,12,13,13,12,13,13,9,10,10,12,13,13,12,13,13},
			{1,1,2,4,4,5,4,4,5,10,10,11,13,13,14,13,13,14,10,10,11,13,13,14,13,13,14},
			{3,4,4,3,4,4,6,7,7,12,13,13,12,13,13,15,16,16,12,13,13,12,13,13,15,16,16},
			{4,4,5,4,4,5,7,7,8,13,13,14,13,13,14,16,16,17,13,13,14,13,13,14,16,16,17},
			{9,10,10,12,13,13,12,13,13,9,10,10,12,13,13,12,13,13,18,19,19,21,22,22,21,22,22},
			{10,10,11,13,13,14,13,13,14,10,10,11,13,13,14,13,13,14,19,19,20,22,22,23,22,22,23},
			{12,13,13,12,13,13,15,16,16,12,13,13,12,13,13,15,16,16,21,22,22,21,22,22,24,25,25},
			{13,13,14,13,13,14,16,16,17,13,13,14,13,13,14,16,16,17,22,22,23,22,22,23,25,25,26}
		};

		__constant__ int LUTchild[8][27] =
		{
			{7,6,7,5,4,5,7,6,7,3,2,3,1,0,1,3,2,3,7,6,7,5,4,5,7,6,7},
			{6,7,6,4,5,4,6,7,6,2,3,2,0,1,0,2,3,2,6,7,6,4,5,4,6,7,6},
			{5,4,5,7,6,7,5,4,5,1,0,1,3,2,3,1,0,1,5,4,5,7,6,7,5,4,5},
			{4,5,4,6,7,6,4,5,4,0,1,0,2,3,2,0,1,0,4,5,4,6,7,6,4,5,4},
			{3,2,3,1,0,1,3,2,3,7,6,7,5,4,5,7,6,7,3,2,3,1,0,1,3,2,3},
			{2,3,2,0,1,0,2,3,2,6,7,6,4,5,4,6,7,6,2,3,2,0,1,0,2,3,2},
			{1,0,1,3,2,3,1,0,1,5,4,5,7,6,7,5,4,5,1,0,1,3,2,3,1,0,1},
			{0,1,0,2,3,2,0,1,0,4,5,4,6,7,6,4,5,4,0,1,0,2,3,2,0,1,0}
		};
	}
}

__device__ float SparseSurfelFusion::device::Length(const float3& vec)
{
	return sqrtf(vec.x * vec.x + vec.y * vec.y + vec.z * vec.z);
}

__global__ void SparseSurfelFusion::device::reduceMaxMinKernel(Point3D<float>* maxBlockData, Point3D<float>* minBlockData, DeviceArrayView<OrientedPoint3D<float>> points, const unsigned int pointsCount)
{
	unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
	__shared__ float MaxPointX[MaxCudaThreadsPerBlock];
	__shared__ float MaxPointY[MaxCudaThreadsPerBlock];
	__shared__ float MaxPointZ[MaxCudaThreadsPerBlock];
	__shared__ float MinPointX[MaxCudaThreadsPerBlock];
	__shared__ float MinPointY[MaxCudaThreadsPerBlock];
	__shared__ float MinPointZ[MaxCudaThreadsPerBlock];
	if (idx < pointsCount) {	// �·�reduce��ʱ����ܻ�ô����(threadIdx.x + stride >= pointsCount)���������ֱ�ӷ��صĻ���__shared__������ĳ���߳��ϱ�Ȼ��û��ֵ�����
		MaxPointX[threadIdx.x] = points[idx].point.coords[0];
		MaxPointY[threadIdx.x] = points[idx].point.coords[1];
		MaxPointZ[threadIdx.x] = points[idx].point.coords[2];
		MinPointX[threadIdx.x] = points[idx].point.coords[0];
		MinPointY[threadIdx.x] = points[idx].point.coords[1];
		MinPointZ[threadIdx.x] = points[idx].point.coords[2];
	}
	else {
		MaxPointX[threadIdx.x] = -1e6;
		MaxPointY[threadIdx.x] = -1e6;
		MaxPointZ[threadIdx.x] = -1e6;
		MinPointX[threadIdx.x] = 1e6;
		MinPointY[threadIdx.x] = 1e6;
		MinPointZ[threadIdx.x] = 1e6;
	}

	__syncthreads();
	// ˳��Ѱַ
	for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
		if (threadIdx.x < stride) {
			float lhs, rhs;
			/** ���ֵ **/
			lhs = MaxPointX[threadIdx.x];
			rhs = MaxPointX[threadIdx.x + stride];
			MaxPointX[threadIdx.x] = lhs < rhs ? rhs : lhs;

			lhs = MaxPointY[threadIdx.x];
			rhs = MaxPointY[threadIdx.x + stride];
			MaxPointY[threadIdx.x] = lhs < rhs ? rhs : lhs;

			lhs = MaxPointZ[threadIdx.x];
			rhs = MaxPointZ[threadIdx.x + stride];
			MaxPointZ[threadIdx.x] = lhs < rhs ? rhs : lhs;

			/** ��Сֵ **/
			lhs = MinPointX[threadIdx.x];
			rhs = MinPointX[threadIdx.x + stride];
			MinPointX[threadIdx.x] = lhs > rhs ? rhs : lhs;

			lhs = MinPointY[threadIdx.x];
			rhs = MinPointY[threadIdx.x + stride];
			MinPointY[threadIdx.x] = lhs > rhs ? rhs : lhs;

			lhs = MinPointZ[threadIdx.x];
			rhs = MinPointZ[threadIdx.x + stride];
			MinPointZ[threadIdx.x] = lhs > rhs ? rhs : lhs;
		}
		__syncthreads();
	}
	if (threadIdx.x == 0) {	// ��block�ĵ�һ���߳�
		maxBlockData[blockIdx.x].coords[0] = MaxPointX[threadIdx.x];
		maxBlockData[blockIdx.x].coords[1] = MaxPointY[threadIdx.x];
		maxBlockData[blockIdx.x].coords[2] = MaxPointZ[threadIdx.x];

		minBlockData[blockIdx.x].coords[0] = MinPointX[threadIdx.x];
		minBlockData[blockIdx.x].coords[1] = MinPointY[threadIdx.x];
		minBlockData[blockIdx.x].coords[2] = MinPointZ[threadIdx.x];
	}
}

__host__ void SparseSurfelFusion::device::findMaxMinPoint(Point3D<float>& MaxPoint, Point3D<float>& MinPoint, Point3D<float>* maxArray, Point3D<float>* minArray, const unsigned int GridNum)
{
	// ͨ��������1000
	for (unsigned int i = 0; i < GridNum; i++) {
		MaxPoint.coords[0] = MaxPoint.coords[0] < maxArray[i].coords[0] ? maxArray[i].coords[0] : MaxPoint.coords[0];
		MaxPoint.coords[1] = MaxPoint.coords[1] < maxArray[i].coords[1] ? maxArray[i].coords[1] : MaxPoint.coords[1];
		MaxPoint.coords[2] = MaxPoint.coords[2] < maxArray[i].coords[2] ? maxArray[i].coords[2] : MaxPoint.coords[2];

		MinPoint.coords[0] = MinPoint.coords[0] > minArray[i].coords[0] ? minArray[i].coords[0] : MinPoint.coords[0];
		MinPoint.coords[1] = MinPoint.coords[1] > minArray[i].coords[1] ? minArray[i].coords[1] : MinPoint.coords[1];
		MinPoint.coords[2] = MinPoint.coords[2] > minArray[i].coords[2] ? minArray[i].coords[2] : MinPoint.coords[2];
	}
}

__global__ void SparseSurfelFusion::device::getCoordinateAndNormalKernel(OrientedPoint3D<float>* point, DeviceArrayView<DepthSurfel> PointCloud, const unsigned int pointsCount)
{
	const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= pointsCount) return;
	point[idx].point = Point3D<float>(PointCloud[idx].VertexAndConfidence.x, PointCloud[idx].VertexAndConfidence.y, PointCloud[idx].VertexAndConfidence.z);
	point[idx].normal = Point3D<float>(PointCloud[idx].NormalAndRadius.x, PointCloud[idx].NormalAndRadius.y, PointCloud[idx].NormalAndRadius.z);
}

__global__ void SparseSurfelFusion::device::adjustPointsCoordinateAndNormalKernel(OrientedPoint3D<float>* points, const Point3D<float> center, const float maxEdge, const unsigned int pointsCount)
{
	unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= pointsCount) return;
	for (int i = 0; i < DIMENSION; i++) {
		points[idx].point.coords[i] = (points[idx].point.coords[i] - center.coords[i]) * 1.0f / maxEdge;
	}

	// �����߷�����[-2^(maxDepth + 1), 2^(maxDepth + 1)]������䣬����[-1, 1]
	float3 PointNormal;
	PointNormal.x = points[idx].normal.coords[0];
	PointNormal.y = points[idx].normal.coords[1];
	PointNormal.z = points[idx].normal.coords[2];
	float len = device::Length(PointNormal);
	if (len > device::eps) {
		len = 1.0f / len;
	}
	len *= (2 << device::maxDepth);
	for (int i = 0; i < DIMENSION; i++) {
		points[idx].normal.coords[i] *= len;
	}
	//if (idx < 1000)	printf("index = %u, point = (%.5f, %.5f, %.5f), normal  = (%.5f, %.5f, %.5f)\n", idx, points[idx].point.coords[0], points[idx].point.coords[1], points[idx].point.coords[2], points[idx].normal.coords[0], points[idx].normal.coords[1], points[idx].normal.coords[2]);

}

__global__ void SparseSurfelFusion::device::generateCodeKernel(OrientedPoint3D<float>* pos, long long* keys, const unsigned int pointsNum)
{
	unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;	// unsigned int �� 2 * (2^31 - 1) = 42.95�� 
	if (idx >= pointsNum)	return;
	long long key = 0ll;	// ��ǰ���ܵ��key
	Point3D<float> myCenter = Point3D<float>(0.5f, 0.5f, 0.5f);
	float myWidth = 0.25f;
	// node�������Keyǰ32λ��x0y0z0 x1y1z1 ... xD-1 yD-1 zD-1  ->  D���10��
	for (int i = device::maxDepth - 1; i >= 0; i--) {	// ��0�㿪ʼ����
		if (pos[idx].point.coords[0] > myCenter.coords[0]) {	// x�����ĵ��ұߣ�����Ϊ1
			key |= 1ll << (3 * i + 34);		// ���ձ���˳��1�Ƶ���Ӧλ��
			myCenter.coords[0] += myWidth;
		}
		else {								// keyĬ��Ϊ0������������轫����Ϊ��Ϊ0
			myCenter.coords[0] -= myWidth;	
		}

		if (pos[idx].point.coords[1] > myCenter.coords[1]) {
			key |= 1ll << (3 * i + 33);
			myCenter.coords[1] += myWidth;
		}
		else {
			myCenter.coords[1] -= myWidth;
		}

		if (pos[idx].point.coords[2] > myCenter.coords[2]) {
			key |= 1ll << (3 * i + 32);
			myCenter.coords[2] += myWidth;
		}
		else {
			myCenter.coords[2] -= myWidth;
		}
		myWidth /= 2.0f;
	}
	// �ȼ�¼��node��λ�ã��ּ�¼��������ܵ���verticeArray�е�λ��, ����node�����ڸ�λ�������ʱ��idx��ȫ����Ӱ��node��˳����Ȼ����octree��˳��
	keys[idx] = key + idx;	
	//if (idx % 1000 == 0) {
	//	const int keyHigher32bits = int(keys[idx] >> 32);					// Octree����
	//	const int keyLower32bits = int(keys[idx] & ((1ll << 31) - 1));	// ��ԭʼ�����е�idx
	//	printf("idx = %d  High = %d  Low = %d\n", idx, keyHigher32bits, keyLower32bits);
	//}
}

__global__ void SparseSurfelFusion::device::updataLower32ForSortedDensePoints(const unsigned int sortedKeysCount, long long* sortedVerticesKey)
{
	const unsigned int idx = threadIdx.x + blockDim.x * blockIdx.x;
	if (idx >= sortedKeysCount)	return;
	if (idx == 0) {
		sortedVerticesKey[idx] &= ~((1ll << 32) - 1);	// ȡ��32λ�����ݣ���32λ��Ϊ0����Ϊ��ʱDensePoint�����Ѿ�������֮ǰ��32λ��������Ҫ���óɵ��µ�idx
	}
	else {
		sortedVerticesKey[idx] &= ~((1ll << 32) - 1);
		sortedVerticesKey[idx] += idx;						// ����������õ�32λ��idx��֮ǰ��idx��DensePoints����û�������idx������DensePoints�Ѿ������򣬱������
	}
}

__global__ void SparseSurfelFusion::device::labelSortedVerticesKeysKernel(const unsigned int sortedKeysCount, DeviceArrayView<long long> sortedVerticesKey, unsigned int* keyLabel)
{
	const unsigned int idx = threadIdx.x + blockDim.x * blockIdx.x;
	if (idx >= sortedKeysCount)	return;
	if (idx == 0) { 
		keyLabel[0] = 1; 
	}
	else {	// ��Ϊ�˴���Ҫ����ǰһ����idx�����������ͬһ���˺�������һ�����ڷô��ͻ
		// ֻ�Ƚ�Octree���벿�֣���ΪD��ڵ㣬ͬ���ڵ�Ĳ���ֻ������8��Ҷ�ӽڵ�
		const unsigned int Higher32Bits_Current = int(sortedVerticesKey[idx] >> 32);
		const unsigned int Higher32Bits_Previous = int(sortedVerticesKey[idx - 1] >> 32);
		if (Higher32Bits_Current != Higher32Bits_Previous) { keyLabel[idx] = 1; }
		else { keyLabel[idx] = 0; }
	}
}

__global__ void SparseSurfelFusion::device::compactedVoxelKeyKernel(const PtrSize<long long> sortedVoxelKey, const unsigned int* voxelKeyLabel, const unsigned int* prefixsumedLabel, long long* compactedKey, DeviceArrayHandle<int> compactedOffset)
{
	const auto idx = threadIdx.x + blockDim.x * blockIdx.x;
	if (idx >= sortedVoxelKey.size) return;
	if (voxelKeyLabel[idx] == 1) {										//�������ֵ��ǰһ����ͬʱ����ʱҲ�ǵ�һ����ͬ
		compactedKey[prefixsumedLabel[idx] - 1] = sortedVoxelKey[idx];	//����ʱ�ļ�ֵ��compactedKey
		compactedOffset[prefixsumedLabel[idx] - 1] = idx;				//�����ʱ���ƫ�Ƶ�idx��compactedOffset�洢
		//������֤��compactedKey��compactedOffset��Ӧ��compactedKey[i]������ǰһ����һ���ı����ֵ��compactedOffset[i]������������ֵ��voxelKeyLabel�еĵڼ���
	}
	if (idx == 0) {
		//printf("denseNodeNum = %d\n", sortedVoxelKey.size);
		// compactedVoxelKey �����ǣ�voxelsNum
		// compactedOffset   �����ǣ�voxelsNum + 1
		compactedOffset[compactedOffset.Size() - 1] = sortedVoxelKey.size;	// ���һ��ֵ��¼һ���ж��ٸ���Ч�����ص�(�����ظ�������)��
	}
}

__global__ void SparseSurfelFusion::device::initUniqueNodeKernel(OctNode* uniqueNode, const DeviceArrayView<long long> compactedKey, const unsigned int compactedNum, const int* compactedOffset)
{
	const unsigned int idx = threadIdx.x + blockDim.x * blockIdx.x;
	if (idx >= compactedNum) return;	// ѹ����ĸ���
	const int keyHigher32bits = int(compactedKey[idx] >> 32);					// Octree����
	const int keyLower32bits = int(compactedKey[idx] & ((1ll << 31) - 1));		// ��ԭʼ�����е�idx
	uniqueNode[idx].key = keyHigher32bits;										// ��¼�ڵ��key
	uniqueNode[idx].pidx = compactedOffset[idx];								// ��ǵ�һ���ڵ��λ��
	uniqueNode[idx].pnum = compactedOffset[idx + 1] - compactedOffset[idx];		// �����ͬkeyֵ�ڵ������
}

__global__ void SparseSurfelFusion::device::generateNodeNumsKernel(const DeviceArrayView<long long> uniqueCode, const unsigned int nodesCount, unsigned int* nodeNums)
{
	const unsigned int idx = threadIdx.x + blockDim.x * blockIdx.x;
	if (idx >= nodesCount)	return;
	if (idx == 0) {	// �׽ڵ�����Ϊ0���������ڵ㶨λ
		nodeNums[idx] = 0;
	}
	else if ((uniqueCode[idx - 1] >> 35) != (uniqueCode[idx] >> 35)) {		// 32 + 3 = 35  ->  ĩ�˽ڵ����һ��ڵ�  ->  ���ڵ���ͬ
		nodeNums[idx] = 8;
	}
	else {
		nodeNums[idx] = 0;
	}
}

__global__ void SparseSurfelFusion::device::buildNodeArrayDKernel(DeviceArrayView<OctNode> uniqueNode, DeviceArrayView<unsigned int> nodeAddress, DeviceArrayView<long long> compactedKey, const unsigned int nodesCount, int* Point2NodeArrayD, OctNode* NodeArrayD, unsigned int* NodeArrayDAddressFull)
{
	const unsigned int idx = threadIdx.x + blockDim.x * blockIdx.x;
	if (idx >= nodesCount)	return;
	int DLevelNodeIndex = nodeAddress[idx] + (uniqueNode[idx].key & 7);	// D��ڵ��λ��
	NodeArrayD[DLevelNodeIndex] = uniqueNode[idx];
	NodeArrayDAddressFull[DLevelNodeIndex] = nodeAddress[idx];
	const int keyLower32bits = int(compactedKey[idx] & ((1ll << 31) - 1));				// ��ԭʼ�����е�idx
	Point2NodeArrayD[keyLower32bits] = DLevelNodeIndex;									// �������ܵ㵽�ڵ��ӳ��
	//if (idx % 1000 == 0)	printf("nodeIndex = %d   keyLower32bits = %d\n", idx, keyLower32bits);
}

__global__ void SparseSurfelFusion::device::initNodeArrayDidxDnumKernel(OctNode* NodeArrayD, const unsigned int TotalNodeNums)
{
	const unsigned int idx = threadIdx.x + blockDim.x * blockIdx.x;
	if (idx >= TotalNodeNums)	return;
	NodeArrayD[idx].dnum = 1;
	NodeArrayD[idx].didx = idx;
}

__global__ void SparseSurfelFusion::device::processPoint2NodeArrayDKernel(int* Point2NodeArrayD, const unsigned int verticesCount)
{
	const unsigned int idx = threadIdx.x + blockDim.x * blockIdx.x;
	unsigned int nowIdx = idx;
	if (idx >= verticesCount)	return;
	int val = Point2NodeArrayD[idx];	// �ó��ܵ��Ӧ��NodeArrays��λ�ã����Point2NodeArrayD = -1��ô˵���õ�û����NodeArrays��
	while (val == -1) {					// ����õ㲢��������NodeArrays�У���ô����ǰ������ĳ�����NodeArrays�еĵ�
		nowIdx--;
		val = Point2NodeArrayD[nowIdx];
	}
	Point2NodeArrayD[idx] = val;
}

__global__ void SparseSurfelFusion::device::setPidxDidxInvalidValue(OctNode* uniqueNodeArrayPreviousLevel, const unsigned int TotalNodeNums)
{
	const unsigned int idx = threadIdx.x + blockDim.x * blockIdx.x;
	if (idx >= TotalNodeNums)	return;
	uniqueNodeArrayPreviousLevel[idx].pidx = device::maxIntValue;
	uniqueNodeArrayPreviousLevel[idx].didx = device::maxIntValue;
}

__global__ void SparseSurfelFusion::device::generateUniqueNodeArrayPreviousLevelKernel(DeviceArrayView<OctNode> NodeArrayD, DeviceArrayView<unsigned int> NodeAddressFull, const unsigned int TotalNodeNums, const unsigned int depth, OctNode* uniqueNodeArrayPreviousLevel)
{
	const unsigned int idx = threadIdx.x + blockDim.x * blockIdx.x;
	if (idx >= TotalNodeNums)	return;
	if (NodeArrayD[idx].pnum == 0) {							// ��ǰ�ڵ��ǿյģ���ǰ�ڵ�����Ч�ڵ㡾���ţ���Щ����Ч�ڵ���Щ����Ч�ڵ㡿
		unsigned int AFatherFirstSonNode = idx - idx % 8;		// �ҵ���ǰ���ڵ�ĵ�һ���ӽڵ�
		unsigned int AFatherValidNodeIdx = -1;					// ��ǰ���ڵ�ĵ�һ����Ч�Ľڵ�
		for (int i = 0; i < 8; i++) {							// ����������ڵ������ӽڵ㣬ֱ���ҵ���Ч���ӽڵ�
			AFatherValidNodeIdx = AFatherFirstSonNode + i;		// ���ڵ�ĵ�i���ӽڵ�idx
			if (NodeArrayD[AFatherValidNodeIdx].pnum != 0) {	// ��������Ч�Ľڵ������ѭ��
				break;
			}
		}
		const unsigned int fatherIdx = NodeAddressFull[AFatherValidNodeIdx] / 8;				// ��һ��(�����ڵ�)������0  0   0 8   8    ....  -->  00000000 11111111
		if (NodeArrayD[idx].dnum != 0) {	
			atomicAdd(&uniqueNodeArrayPreviousLevel[fatherIdx].dnum, NodeArrayD[idx].dnum);		// ԭ�Ӳ��������㵱ǰ�ڵ������ж��ٸ���
			atomicMin(&uniqueNodeArrayPreviousLevel[fatherIdx].didx, NodeArrayD[idx].didx);		// ԭ�Ӳ������ҳ���ǰ�ڵ�����idx��С�ĵ㣬����ǰ�ڵ���׸���
		}
	}
	else {	// �������Ч��
		const int fatherKey = NodeArrayD[idx].key & (~(7 << (3 * (device::maxDepth - depth))));	// ��ǰ��ĸ��ڵ�Key�������Ƿ�����Ч��
		const int fatherIdx = NodeAddressFull[idx] / 8;											// ��ǰ��ĸ��ڵ�
		const int sonKey = (NodeArrayD[idx].key >> (3 * (device::maxDepth - depth))) & 7;		// ���ӽڵ�ļ�����ʵ���ǵ�ǰ���������bitֵ
		uniqueNodeArrayPreviousLevel[fatherIdx].key = fatherKey;								// uniqueNodeArrayPreviousLevel��һ���NodeArrays��ʼ��¼���ڵ�
		//if (depth == device::maxDepth) printf("index = %d  fatherIdx = %d  NA_pnum = %d   Prev_pnum = %d\n", idx, fatherIdx, NodeArrayD[idx].pnum, uniqueNodeArrayPreviousLevel[fatherIdx].pnum);
		atomicAdd(&uniqueNodeArrayPreviousLevel[fatherIdx].pnum, NodeArrayD[idx].pnum);			// ��һ�㸸�ڵ������ǰ�ڵ������pnum�ĺͣ����뵱ǰ�㸸�ڵ���ͬһ��Key�ĳ��ܵ����(ֻ����Ч�����pnumֵ����Ч����0)
		atomicMin(&uniqueNodeArrayPreviousLevel[fatherIdx].pidx, NodeArrayD[idx].pidx);			// ��һ�㸸�ڵ��pidx���������ӽڵ���idx��С�ģ����뵱ǰ���ڵ���ͬһ��Key�ĳ��ܵ�����ǰ����Ǹ���idx
		atomicAdd(&uniqueNodeArrayPreviousLevel[fatherIdx].dnum, NodeArrayD[idx].dnum);			// ��һ�㸸�ڵ��dnum�ǰ������������ӽڵ��������������Ч�����Ч��
		atomicMin(&uniqueNodeArrayPreviousLevel[fatherIdx].didx, NodeArrayD[idx].didx);			// ��һ�㸸�ڵ���didx���ӽڵ��У���NodeArrayD���ǰ����һ���ڵ��idx��������Ч����Ч
		uniqueNodeArrayPreviousLevel[fatherIdx].children[sonKey] = idx;							// ����Ч���ʱ����������Ч�ĺ��ӽڵ㣬���ǵ�ǰ�����Чֵ��������һ�㸸�׵Ķ��ӣ���Ч�㲻����Ϊ����
	}
}

__global__ void SparseSurfelFusion::device::generateNodeNumsPreviousLevelKernel(DeviceArrayView<OctNode> uniqueNodePreviousLevel, const unsigned int uniqueCount, const unsigned int depth, unsigned int* NodeNumsPreviousLevel)
{
	const unsigned int idx = threadIdx.x + blockDim.x * blockIdx.x;
	if (idx >= uniqueCount)	return;
	if (idx == 0) {	// ��һ���ڵ㸸�ڵ���ǰһ���޷��Ƚϣ�Ĭ��Ϊ0
		NodeNumsPreviousLevel[idx] = 0;
	}
	else {
		// �鿴���ڵ��Ƿ���ǰһ����ͬ����ͬΪ8
		if ((uniqueNodePreviousLevel[idx - 1].key >> (3 * (device::maxDepth - depth + 1))) != (uniqueNodePreviousLevel[idx].key >> (3 * (device::maxDepth - depth + 1)))) {
			NodeNumsPreviousLevel[idx] = 8;
		}
		else {	// ��ͬΪ0
			NodeNumsPreviousLevel[idx] = 0;
		}
	}
}

__global__ void SparseSurfelFusion::device::generateNodeArrayPreviousLevelKernel(DeviceArrayView<OctNode> uniqueNodePreviousLevel, DeviceArrayView<unsigned int> nodeAddressPreviousLevel, const unsigned int uniqueCount, const unsigned int depth, OctNode* nodeArrayPreviousLevel, OctNode* nodeArrayD, unsigned int* NodeAddressFull)
{
	const unsigned int idx = threadIdx.x + blockDim.x * blockIdx.x;
	if (idx >= uniqueCount)	return;
	int index = nodeAddressPreviousLevel[idx] + ((uniqueNodePreviousLevel[idx].key >> (3 * (device::maxDepth - depth + 1))) & 7);	// ��һ�㸸�ڵ��idx
	nodeArrayPreviousLevel[index] = uniqueNodePreviousLevel[idx];
	NodeAddressFull[index] = nodeAddressPreviousLevel[idx];
	for (int i = 0; i < 8; i++) {	// children�ڵ����8��������ȫ������ֵ0�ˣ���ÿ�����ڵ����ȫ���ĺ��ӽڵ㣬ѡ����Ч�ĺ��ӽڵ㣬���Լ���index��ֵ������
		int nodeArrayDIndex = uniqueNodePreviousLevel[idx].children[i];	// ��¼��һ��ڵ�ĺ��ӽڵ㣬��nodeArrayD�Ľڵ�
		if (nodeArrayDIndex != 0) {	// ��Ч�ĺ��ӽڵ㣬����һ��bug��������ӽڵ���nodeArrayD�е�index�������0��Ҳ�ᱻ������Ч�㴦��
			nodeArrayD[nodeArrayDIndex].parent = index;	// ����ǰ�ڵ����Ϊ��Ч���ӽڵ�ĸ��ڵ�
		}
	}
}

__global__ void SparseSurfelFusion::device::updateNodeArrayParentAndChildrenKernel(const unsigned int totalNodeArrayLength, OctNode* NodeArray)
{
	const unsigned int idx = threadIdx.x + blockDim.x * blockIdx.x;
	if (idx >= totalNodeArrayLength)	return;
	if (NodeArray[idx].pnum == 0)	return;		// �������Ч����������º��Ӻ͸��׽ڵ�
	int depth = 0;
	for (depth = 0; depth < device::maxDepth; depth++) {				// Ѱ��idx����һ��
		if (device::BaseAddressArray_Device[depth] <= idx && idx < device::BaseAddressArray_Device[depth + 1]) {
			break;
		}
	}
	if (idx == 0) {
		NodeArray[idx].parent = -1;										// ����ǵ�0��ڵ�
#pragma unroll	// չ��ѭ������������
		for (int child = 0; child < 8; child++) {						// ����Children��NodeArray��λ��
			NodeArray[idx].children[child] += BaseAddressArray_Device[depth + 1];
		}
	}
	else {
		NodeArray[idx].parent += BaseAddressArray_Device[depth - 1];	// ����Parent��NodeArray�е�λ��

		if (depth < device::maxDepth) {										// ���һ��û��child�ڵ�
#pragma unroll	// չ��ѭ������������
			for (int child = 0; child < 8; child++) {						// ����Children��NodeArray��λ��
				if (NodeArray[idx].children[child] != 0) {					// ���ӽڵ�Ϊ��Ч�ڵ�
					NodeArray[idx].children[child] += BaseAddressArray_Device[depth + 1];
				}
			} 
		}

	}	
}

__global__ void SparseSurfelFusion::device::updateEmptyNodeInfo(const unsigned int totalNodeArrayLength, OctNode* NodeArray)
{
	const unsigned int idx = threadIdx.x + blockDim.x * blockIdx.x;
	if (idx >= totalNodeArrayLength || idx == 0)	return;
	if ((idx - 1) % 8 == 0) {	// �ų�0���һ���ڵ㣬ÿһ��ڵ㶼��8�ı������׽ڵ�
		int nowPIdx;			// �����Ч���pidx
		int nowDIdx;			// �����Ч���didx
		int validIdx;			// �����Ч���NodeArray��λ��
		int commonParent;		// �����Ч���parent
		for (validIdx = 0; validIdx < 8; validIdx++) {				// ����ÿ���ڵ㣬����ýڵ�pnum == 0����˵������Ч�ڵ㣬pnum != 0��˵����Ч
			if (NodeArray[idx + validIdx].pnum != 0) {				// �ҵ��׸���Ч�ڵ�
				nowPIdx = NodeArray[idx + validIdx].pidx;
				nowDIdx = NodeArray[idx + validIdx].didx;
				commonParent = NodeArray[idx + validIdx].parent;
				break;
			}
		}
		int depth = 0;
		for (depth = 0; depth < device::maxDepth; depth++) {				// Ѱ��idx����һ��
			if (device::BaseAddressArray_Device[depth] <= idx && idx < device::BaseAddressArray_Device[depth + 1]) {
				break;
			}
		}
		
		// ͨ������ȥ��ǰ��Ļ������룬����ǰ���ͬ���ױ���
		int baseKey = NodeArray[idx + validIdx].key - ((NodeArray[idx + validIdx].key) & (7 << (3 * (device::maxDepth - depth))));

		for (int j = 0; j < 8; j++) {									// �ٴα�����8����
			int index = idx + j;										// ����NodeArrayλ��
			if (NodeArray[index].pnum == 0) {							// ��Ч��	
				for (int k = 0; k < 8; k++) {							// ��Ч�ڵ�ĺ��ӽڵ����Ч
					NodeArray[index].children[k] = -1;
				}
			}
			else {														// ��Ч��
				int basePos;											// ��Ч��ĺ��ӽڵ��е�һ���ڵ���NodeArray�е�index
				for (int k = 0; k < 8; k++) {							// ���������Ч������к��ӽڵ�
					if (NodeArray[index].children[k] > 0) {				// ��������ĺ�������Ч��
						basePos = NodeArray[index].children[k] - k;		// �ҵ�������ӽڵ����������ĵ�һ���ڵ���NodeArray�е�index
						break;
					}
				}
				for (int k = 0; k < 8; k++) {							// ��Ч��ĺ��Ӿ�Ӧ�����ҵ���NodeArray�е�λ��
					if (depth != device::maxDepth) {					// �������һ��
						NodeArray[index].children[k] = basePos + k;		// ÿһ�����ӽڵ㶼�ҵ���NodeArray�е�λ��
					}
					else {
						NodeArray[index].children[k] = -1;				// ���һ��û�к��ӽڵ�
					}
				}
			}
			NodeArray[index].key = baseKey + (j << (3 * (device::maxDepth - depth)));	// ���µ�ǰ���key
			NodeArray[index].pidx = nowPIdx;		// �׸�key��ͬ���׸����ܵ��ڳ��ܵ������index
			nowPIdx += NodeArray[index].pnum;		// ���key��ͬ�ĳ��ܵ�

			if (depth != device::maxDepth) {		// ����������һ��D��
				NodeArray[index].didx = nowDIdx;	// ���D���׸���
				nowDIdx += NodeArray[index].dnum;	// ����ƫ��
			}
			NodeArray[index].parent = commonParent;	// ������Ч��Ч������ͬһ��parent
		}
	}
}

__global__ void SparseSurfelFusion::device::computeNodeNeighborKernel(const unsigned int left, const unsigned int thisLevelNodeCount, const unsigned int depth, OctNode* NodeArray)
{
	const unsigned int idx = threadIdx.x + blockDim.x * blockIdx.x;
	if (idx >= thisLevelNodeCount)	return;	// 0���ھ��Ѿ���ʼ��
	const unsigned int offset = idx + left;	// ��ǰ��Ľڵ���NodeArray�е�λ��
	for (int i = 0; i < 27; i++) {
		int sonKey = (NodeArray[offset].key >> (3 * (device::maxDepth - depth))) & 7;
		int parentIdx = NodeArray[offset].parent;
		int neighParent = NodeArray[parentIdx].neighs[device::LUTparent[sonKey][i]];
		if (neighParent != -1) {
			NodeArray[offset].neighs[i] = NodeArray[neighParent].children[LUTchild[sonKey][i]];
		}
		else {
			NodeArray[offset].neighs[i] = -1;
		}
	}
}

__global__ void SparseSurfelFusion::device::computeEncodedFunctionNodeIndexKernel(DeviceArrayView<unsigned int> depthBuffer, DeviceArrayView<OctNode> NodeArray, const unsigned int totalNodeCount, int* NodeIndexInFunction)
{
	const unsigned int idx = threadIdx.x + blockDim.x * blockIdx.x;
	if (idx >= totalNodeCount)	return;
	int depth = depthBuffer[idx];	// ��ǰidx������һ��
	getEncodedFunctionNodeIndex(NodeArray[idx].key, depth, NodeIndexInFunction[idx]);
}

__device__ void SparseSurfelFusion::device::getEncodedFunctionNodeIndex(const int& key, const int& CurrentDepth, int& index)
{
	/*
	 * Part_1 = (1 + (1 << (device::maxDepth + 1)) + (1 << (2 * (device::maxDepth + 1))))
	 * Code_1 = 00000000001 00000000001 00000000001	(device::maxDepth + 1Ϊһ�Σ���Ϊ3��)
	 * Part_2 = ((1 << CurrentDepth) - 1);		������CurrentDepth = 8��
	 * Code_2 = 00011111111
	 * Part_1 * Part2 <=> ��Code_2�ֱ�д��code����������[0, 11], [12, 21], [22, 31]
	 * index  = 00011111111 00011111111 00011111111
	*/ 
	index = ((1 << CurrentDepth) - 1) * (1 + (1 << (device::maxDepth + 1)) + (1 << (2 * (device::maxDepth + 1))));

	/*
	 * �����裺sonKey = 111, CurrentDepth = 8��
	 * Part_3 = 00011111111 00011111111 00011111111
	 * idx+P3 = 00100000000 00100000000 00100000000
	 * �����裺sonKey = 101, CurrentDepth = 8��
	 * Part_3 = 00011111111 00000000000 00011111111
	 * idx+P3 = 00100000000 00011111111 00100000000
	*/
	for (int depth = CurrentDepth; depth >= 1; depth--) {
		int sonKeyX = (key >> (3 * (device::maxDepth - depth) + 2)) & 1;
		int sonKeyY = (key >> (3 * (device::maxDepth - depth) + 1)) & 1;
		int sonKeyZ = (key >> (3 * (device::maxDepth - depth)    )) & 1;
		index += sonKeyX * (1 << (CurrentDepth - depth)) +
			     sonKeyY * (1 << (CurrentDepth - depth)) * (1 <<      (device::maxDepth + 1)) +
			     sonKeyZ * (1 << (CurrentDepth - depth)) * (1 << (2 * (device::maxDepth + 1)));
	}
}

__global__ void SparseSurfelFusion::device::ComputeDepthAndCenterKernel(DeviceArrayView<OctNode> NodeArray, const unsigned int NodeArraySize, unsigned int* DepthBuffer, Point3D<float>* CenterBuffer)
{
	const unsigned int idx = threadIdx.x + blockDim.x * blockIdx.x;
	if (idx >= NodeArraySize)	return;
	int depth = 0;
	for (depth = 0; depth < device::maxDepth; depth++) {				// Ѱ��idx����һ��
		if (device::BaseAddressArray_Device[depth] <= idx && idx < device::BaseAddressArray_Device[depth + 1]) {
			break;
		}
	}
	DepthBuffer[idx] = depth;
	Point3D<float> center;
	getNodeCenterAllDepth(NodeArray[idx].key, depth, center);
	CenterBuffer[idx] = center;
}

__device__ void SparseSurfelFusion::device::getNodeCenterAllDepth(const int& key, int currentDepth, Point3D<float>& Center)
{
	Center.coords[0] = float(0.5f);
	Center.coords[1] = float(0.5f);
	Center.coords[2] = float(0.5f);
	float Width = 0.25f;
	for (int i = device::maxDepth - 1; i >= (device::maxDepth - currentDepth); i--) {
		if ((key >> (3 * i + 2)) & 1) { Center.coords[0] += Width; }
		else { Center.coords[0] -= Width; }

		if ((key >> (3 * i + 1)) & 1) { Center.coords[1] += Width; }
		else { Center.coords[1] -= Width; }

		if ((key >> (3 * i)) & 1) { Center.coords[2] += Width; }
		else { Center.coords[2] -= Width; }

		Width /= 2;
	}
}

void SparseSurfelFusion::BuildOctree::getCoordinateAndNormal(DeviceArrayView<DepthSurfel> denseSurfel, cudaStream_t stream)
{
	unsigned int num = denseSurfel.Size();
	dim3 block(128);
	dim3 grid(divUp(denseSurfel.Size(), block.x));
	device::getCoordinateAndNormalKernel << <grid, block, 0, stream >> > (sampleOrientedPoints.Array().ptr(), denseSurfel, num);
}

void SparseSurfelFusion::BuildOctree::getBoundingBox(DeviceArrayView<OrientedPoint3D<float>> points, Point3D<float>& MaxPoint, Point3D<float>& MinPoint, cudaStream_t stream)
{
	/********************* ��Լ���ÿ��Block�����СPoint3D *********************/
	const unsigned int num = points.Size();
	//printf("pointsCount = %d\n", num);
	const unsigned int gridNum = divUp(num, device::MaxCudaThreadsPerBlock);
	dim3 blockReduce(device::MaxCudaThreadsPerBlock);
	dim3 gridReduce(gridNum);
	perBlockMaxPoint.ResizeArrayOrException(gridNum);
	perBlockMinPoint.ResizeArrayOrException(gridNum);
	Point3D<float>* maxArray = perBlockMaxPoint.DeviceArray().ptr();
	Point3D<float>* minArray = perBlockMinPoint.DeviceArray().ptr();
	device::reduceMaxMinKernel << <gridReduce, blockReduce, 0, stream >> > (maxArray, minArray, points, num);
	/********************* ��ÿ��Blockֵ���ص�Host��������е����ֵ *********************/
	perBlockMaxPoint.SynchronizeToHost(stream, true);
	perBlockMinPoint.SynchronizeToHost(stream, true);
	Point3D<float>* maxArrayHost = perBlockMaxPoint.HostArray().data();
	Point3D<float>* minArrayHost = perBlockMinPoint.HostArray().data();
	device::findMaxMinPoint(MaxPoint, MinPoint, maxArrayHost, minArrayHost, gridNum);
	//printf("MaxPoint(%.5f, %.5f, %.5f)   MinPoint(%.5f, %.5f, %.5f)\n", MaxPoint.coords[0], MaxPoint.coords[1], MaxPoint.coords[2], MinPoint.coords[0], MinPoint.coords[1], MinPoint.coords[2]);

}

void SparseSurfelFusion::BuildOctree::adjustPointsCoordinateAndNormal(DeviceBufferArray<OrientedPoint3D<float>>& points, const Point3D<float> MxPoint, const Point3D<float> MnPoint, float& MaxEdge, float ScaleFactor, Point3D<float>& Center, cudaStream_t stream)
{
	for (int i = 0; i < DIMENSION; i++) {
		// ������Ҫ��ȡBoundingBox���һ���߸�ֵ��MaxEdge
		if (i == 0 || MaxEdge < (MxPoint[i] - MnPoint[i])) {
			MaxEdge = float(MxPoint[i] - MnPoint[i]);
		}
		Center.coords[i] = float(MxPoint[i] + MnPoint[i]) / 2.0f;	// �е�λ��
	}
	MaxEdge *= ScaleFactor * 1.0f;	// ����BoundingBox�ĳߴ�
	for (int i = 0; i < DIMENSION; i++) {
		Center.coords[i] -= MaxEdge / 2.0f;	// �������ĵ��λ��
	}
	dim3 block(128);
	dim3 grid(divUp(points.ArrayView().Size(), block.x));
	device::adjustPointsCoordinateAndNormalKernel << <grid, block, 0, stream >> > (points.Array().ptr(), Center, MaxEdge, points.ArrayView().Size());
}

void SparseSurfelFusion::BuildOctree::generateCode(DeviceBufferArray<OrientedPoint3D<float>>& points, DeviceBufferArray<long long>& keys, size_t count, cudaStream_t stream)
{
	sortCode.ResizeArrayOrException(count);		// ����Array
	// �߳���Ӧ�ù���
	dim3 block(128);					// block �� [0, 1024]
	dim3 grid(divUp(count, block.x));	// grid  �� [0, 2^31 - 1]
	device::generateCodeKernel << <grid, block, 0, stream >> > (points.Array().ptr(), keys.Array().ptr(), count);
}

void SparseSurfelFusion::BuildOctree::sortAndCompactVerticesKeys(DeviceArray<OrientedPoint3D<float>>& points, cudaStream_t stream)
{
	const unsigned int VerticesKeysNum = sortCode.ArrayView().Size();
	pointKeySort.Sort(sortCode.Array(), points, stream);
	// ��point���׸���Ϊ���кõĳ��ܵ�
	CHECKCUDA(cudaMemcpyAsync(points.ptr(), pointKeySort.valid_sorted_value.ptr(), sizeof(OrientedPoint3D<float>) * VerticesKeysNum, cudaMemcpyDeviceToDevice, stream));
	CHECKCUDA(cudaMemcpyAsync(sortCode.Array().ptr(), pointKeySort.valid_sorted_key.ptr(), sizeof(long long) * VerticesKeysNum, cudaMemcpyDeviceToDevice, stream));

	//std::vector<long long> sortedCodeHost;
	//sortCode.ArrayView().Download(sortedCodeHost);
	//for (int i = 0; i < sortedCodeHost.size(); i++) {
	//	if (i % 1000 == 0) printf("index = %d   SortedCode = %lld\n", i, sortedCodeHost[i]);
	//}

	//printf("TotalSurfelCount = %d\n", VerticesKeysNum);

	/** ����õ�SortCodeû������ **/

	keyLabel.ResizeArrayOrException(VerticesKeysNum);
	dim3 block(128);
	dim3 grid(divUp(VerticesKeysNum, block.x));
	device::updataLower32ForSortedDensePoints << <grid, block, 0, stream >> > (VerticesKeysNum, sortCode.Array().ptr());
	device::labelSortedVerticesKeysKernel << <grid, block, 0, stream >> > (VerticesKeysNum, sortCode.ArrayView(), keyLabel.ArrayHandle());
	nodeNumsPrefixsum.InclusiveSum(keyLabel.ArrayView(), stream);

	//��ѯ������(CPU����������)
	unsigned int ValidKeysNum;	// ��Ч�ļ�������Octree�ӽڵ�Ϊͬһ���ĵ�Ӧ������Ч��
	//ǰ׺�͵�GPU��ַ��prefixsum_label
	const DeviceArray<unsigned int>& prefixsumLabel = nodeNumsPrefixsum.valid_prefixsum_array;

	//��ǰ׺�ʹ�GPU�п�����ValidKeysNum�У���ǵ����һ��������Ч��Key��������
	CHECKCUDA(cudaMemcpyAsync(&ValidKeysNum, prefixsumLabel.ptr() + prefixsumLabel.size() - 1, sizeof(unsigned int), cudaMemcpyDeviceToHost, stream));
	CHECKCUDA(cudaStreamSynchronize(stream));
	//printf("���ܵ�������%d   Unique������ = %d\n", VerticesKeysNum, ValidKeysNum);
	//����ѹ������
	uniqueCode.ResizeArrayOrException(ValidKeysNum);						//��compactedVoxelKey���ٿռ�
	compactedVerticesOffset.ResizeArrayOrException(ValidKeysNum + 1);		//��compactedVoxelOffset���ٿռ�
	device::compactedVoxelKeyKernel << <grid, block, 0, stream >> > (sortCode.Array(), keyLabel, prefixsumLabel, uniqueCode, compactedVerticesOffset.ArrayHandle());

	/** ��32λ������ **/
	
	//CHECKCUDA(cudaStreamSynchronize(stream));
	//std::vector<long long> uniqueCodeHost;
	//uniqueCode.ArrayView().Download(uniqueCodeHost);
	//for (int i = 0; i < uniqueCodeHost.size(); i++) {
	//	if (i % 1000 == 0) printf("index = %d  uniqueNodeCode = %lld\n", i, uniqueCodeHost[i]);
	//}
}

void SparseSurfelFusion::BuildOctree::initUniqueNode(DeviceBufferArray<OctNode>& uniqueNode, DeviceBufferArray<long long>& uniqueCode, cudaStream_t stream)
{
	const unsigned int nodesCount = uniqueCode.ArraySize();
	//printf("nodesCount = %u\n", nodesCount);

	//printf("ѹ����ĵ� = %d\n", nodesCount);
	uniqueNode.ResizeArrayOrException(nodesCount);
	dim3 block(128);
	dim3 grid(divUp(nodesCount, block.x));
	device::initUniqueNodeKernel << <grid, block, 0, stream >> > (uniqueNode, uniqueCode.ArrayView(), nodesCount, compactedVerticesOffset.Array().ptr());
}

void SparseSurfelFusion::BuildOctree::generateNodeNumsAndNodeAddress(DeviceBufferArray<long long>& uniqueCode, DeviceBufferArray<unsigned int>& NodeNums, DeviceBufferArray<unsigned int>& NodeAddress, cudaStream_t stream)
{
	const unsigned int nodesCount = uniqueCode.ArraySize();
	NodeNums.ResizeArrayOrException(nodesCount);
	dim3 block(128);
	dim3 grid(divUp(nodesCount, block.x));
	device::generateNodeNumsKernel << <grid, block, 0, stream >> > (uniqueCode.ArrayView(), nodesCount, NodeNums.Array().ptr());

	nodeNumsPrefixsum.InclusiveSum(NodeNums.ArrayView(), stream);

	//printf("nodesCount = %u\n", nodesCount);

	NodeAddress.ResizeArrayOrException(nodesCount);	// ���ٿռ�

	unsigned int* NodeAddressPtr = NodeAddress.Array().ptr();		// ��¶ָ��
	unsigned int* prefixsumNodeNumsPtr = nodeNumsPrefixsum.valid_prefixsum_array.ptr();	// ��¶ָ��
	// ��ǰ׺�����ݿ�����nodeAddress��
	CHECKCUDA(cudaMemcpyAsync(NodeAddressPtr, prefixsumNodeNumsPtr, sizeof(unsigned int) * nodesCount, cudaMemcpyDeviceToDevice, stream));
}

void SparseSurfelFusion::BuildOctree::buildNodeArrayD(DeviceArrayView<OrientedPoint3D<float>> denseVertices, DeviceArrayView<OctNode> uniqueNode, DeviceArrayView<long long> compactedKey, DeviceBufferArray<unsigned int>& NodeAddress, DeviceBufferArray<unsigned int>& NodeAddressFull, DeviceBufferArray<int>& Point2NodeArray, DeviceBufferArray<OctNode>& NodeArrayD, cudaStream_t stream)
{
	const unsigned int nodesCount = NodeAddress.ArrayView().Size();		// Unique��������ѹ�����
	//printf("nodesCount = %u\n", nodesCount);
	const unsigned int verticesCount = denseVertices.Size();
	unsigned int TotalNodeNums;		// �ܹ�D���ӽڵ�ĸ���
	const unsigned int* nodeAddressPtr = NodeAddress.Array().ptr();
	CHECKCUDA(cudaMemcpyAsync(&TotalNodeNums, nodeAddressPtr + nodesCount - 1, sizeof(unsigned int), cudaMemcpyDeviceToHost, stream));
	CHECKCUDA(cudaStreamSynchronize(stream));	// ��ͬ��
	TotalNodeNums += 8;	// �׽ڵ��Ƕ�Ӧ��8��û���ϣ�����Ӧ�ñ�ʾ�������һ��(D��)�Ľڵ�
	//printf("D���ܽڵ���Ŀ = %u\n", TotalNodeNums);
	NodeArrayD.AllocateBuffer(TotalNodeNums);
	NodeArrayD.ResizeArrayOrException(TotalNodeNums);
	NodeAddressFull.ResizeArrayOrException(TotalNodeNums);
	Point2NodeArray.ResizeArrayOrException(verticesCount);
	CHECKCUDA(cudaMemsetAsync(NodeArrayD.Array().ptr(), 0, sizeof(OctNode) * TotalNodeNums, stream));			// NodeArrayDĬ�϶�Ϊ0
	CHECKCUDA(cudaMemsetAsync(Point2NodeArray.Array().ptr(), -1, sizeof(int) * verticesCount, stream));			// Point2NodeArrayĬ�϶�Ϊ-1
	dim3 block1(128);
	dim3 grid1(divUp(nodesCount, block1.x));
	device::buildNodeArrayDKernel << <grid1, block1, 0, stream >> > (uniqueNode, NodeAddress.ArrayView(), compactedKey, nodesCount, Point2NodeArray.Array().ptr(), NodeArrayD.Array().ptr(), NodeAddressFull.Array().ptr());
	dim3 block2(128);
	dim3 grid2(divUp(TotalNodeNums, block2.x));
	device::initNodeArrayDidxDnumKernel << <grid2, block2, 0, stream >> > (NodeArrayD.Array().ptr(), TotalNodeNums);
	dim3 block3(128);
	dim3 grid3(divUp(verticesCount, block3.x));
	device::processPoint2NodeArrayDKernel << <grid3, block3, 0, stream >> > (Point2NodeArray.Array().ptr(), verticesCount);
}

void SparseSurfelFusion::BuildOctree::buildOtherDepthNodeArray(int* BaseAddressArray_Host, cudaStream_t stream)
{
	unsigned int allNodeNums_D = NodeArrays[Constants::maxDepth_Host].ArrayView().Size();		// ��¼D��ڵ�8�����ŵĽڵ�����
	//printf("�� D ��Octree�ڵ������ = %d\n", allNodeNums_D);
	unsigned int TotalNodeNumsPreviousLevel;													// ��һ�㸸�ڵ�ĸ���
	for (int depth = Constants::maxDepth_Host; depth >= 1; depth--) {
		NodeArrayCount_Host[depth] = allNodeNums_D;
		int UniqueCountPrevious = allNodeNums_D / 8;	// ��һ��Octree�ڵ�����������š�,ʵ������Unique
		OctNode* previousLevelUniqueNodePtr = uniqueNodePrevious.Array().ptr();	// ���ָ���Ա㸳��ֵ
		unsigned int* previousLevelnodeAddress = nodeAddressPrevious.Array().ptr();	// ���ָ���Ա㸳��ֵ
		// ��uniqueNodePreviousLevel���ٿռ䲢����ֵ
		uniqueNodePrevious.ResizeArrayOrException(UniqueCountPrevious);
		CHECKCUDA(cudaMemsetAsync(previousLevelUniqueNodePtr, 0, sizeof(OctNode) * UniqueCountPrevious, stream));
		// ��nodeAddressPreciousLevel���ٿռ䲢����ֵ
		nodeAddressPrevious.ResizeArrayOrException(UniqueCountPrevious);
		CHECKCUDA(cudaMemsetAsync(previousLevelnodeAddress, 0, sizeof(unsigned int) * UniqueCountPrevious, stream));

		dim3 block_PreLevel(128);
		dim3 grid_PreLevel(divUp(UniqueCountPrevious, block_PreLevel.x));
		device::setPidxDidxInvalidValue << <grid_PreLevel, block_PreLevel, 0, stream >> > (uniqueNodePrevious.Array().ptr(), UniqueCountPrevious);

		dim3 block_D(128);
		dim3 grid_D(divUp(allNodeNums_D, block_D.x));
		device::generateUniqueNodeArrayPreviousLevelKernel << <grid_D, block_D, 0, stream >> > (NodeArrays[depth].ArrayView(), NodeAddressFull.ArrayView(), allNodeNums_D, depth, uniqueNodePrevious.Array().ptr());
		
		// ÿһ���UniqueNodePrev��pidx��pnum��didx��dnum��û����
		nodeNums.ResizeArrayOrException(UniqueCountPrevious);	// ����nodeNumsΪ��һ���Unique��С
		device::generateNodeNumsPreviousLevelKernel << <grid_PreLevel, block_PreLevel, 0, stream >> > (uniqueNodePrevious.ArrayView(), UniqueCountPrevious, depth - 1, nodeNums.Array().ptr());
			
		CHECKCUDA(cudaStreamSynchronize(stream));	// ��ͬ��
		//printf("UniqueCountPrevious = %d\n", UniqueCountPrevious);

		// ��һ�㸸�ڵ������һ�����ڵ�ǰ��ڵ㣬���ֱ�ӷ���nodeNumsPrefixsum�������ٷ��仺��
		nodeNumsPrefixsum.InclusiveSum(nodeNums.ArrayView(), stream);
		void* prefixsumNodeNumsPtr = nodeNumsPrefixsum.valid_prefixsum_array.ptr();	// ��¶ָ��
		// ��ǰ׺�����ݿ�����nodeAddress��
		CHECKCUDA(cudaMemcpyAsync(previousLevelnodeAddress, prefixsumNodeNumsPtr, sizeof(unsigned int) * UniqueCountPrevious, cudaMemcpyDeviceToDevice, stream));
		
		if (depth > 1) {	// �ǵ�һ��
			CHECKCUDA(cudaMemcpyAsync(&TotalNodeNumsPreviousLevel, previousLevelnodeAddress + UniqueCountPrevious - 1, sizeof(unsigned int), cudaMemcpyDeviceToHost, stream));
			TotalNodeNumsPreviousLevel += 8;			// ��һ�㸸�ڵ���������׸����ڵ�Ϊ0������ۺͺ�Ӧ��+8
			NodeArrays[depth - 1].AllocateBuffer(TotalNodeNumsPreviousLevel);
			NodeArrays[depth - 1].ResizeArrayOrException(TotalNodeNumsPreviousLevel);
			NodeAddressFull.ResizeArrayOrException(TotalNodeNumsPreviousLevel);			// (depth - 1)�����Žڵ�NodeAddress��ȷ����һ��Unique��index
			void* previousLevelNodeArrayPtr = NodeArrays[depth - 1].Array().ptr();		// ���ָ���Ա㸳��ֵ
			CHECKCUDA(cudaMemsetAsync(previousLevelNodeArrayPtr, 0, sizeof(OctNode) * TotalNodeNumsPreviousLevel, stream));

			// ������һ�㸸�ڵ��nodeArray��������ǰ���nodeArray�е�parent��ֵ
			device::generateNodeArrayPreviousLevelKernel << <grid_PreLevel, block_PreLevel, 0, stream >> > (uniqueNodePrevious.ArrayView(), nodeAddressPrevious.ArrayView(), UniqueCountPrevious, depth, NodeArrays[depth - 1].Array().ptr(), NodeArrays[depth].Array().ptr(), NodeAddressFull.Array().ptr());
			
			//CHECKCUDA(cudaStreamSynchronize(stream));	// ��ͬ��
			//printf("TotalNodeNumsPreviousLevel = %d\n", TotalNodeNumsPreviousLevel);

		}
		else {
			TotalNodeNumsPreviousLevel = 1;
			NodeArrays[depth - 1].AllocateBuffer(TotalNodeNumsPreviousLevel);
			NodeArrays[depth - 1].ResizeArrayOrException(TotalNodeNumsPreviousLevel);
			OctNode* NodeArrays_0 = NodeArrays[depth - 1].Array().ptr();	// ��¶ָ�룬ֱ�Ӹ�ֵ
			CHECKCUDA(cudaMemcpyAsync(NodeArrays_0, uniqueNodePrevious.Array().ptr(), sizeof(OctNode) * 1, cudaMemcpyDeviceToDevice, stream));
		}
		allNodeNums_D = TotalNodeNumsPreviousLevel;
	}

	CHECKCUDA(cudaStreamSynchronize(stream));	// ��ͬ��

	for (int i = 0; i <= Constants::maxDepth_Host; i++) {
		if (i == 0) {
			NodeArrayCount_Host[i] = 1;
		}
		else {
			BaseAddressArray_Host[i] = BaseAddressArray_Host[i - 1] + NodeArrayCount_Host[i - 1];
		}
	}
	//for (int i = 0; i <= Constants::maxDepth_Host; i++) {
	//	printf("BaseAddressArray[%d] = %d    NodeArrayCount[%d] = %d\n", i, BaseAddressArray_Host[i], i, NodeArrayCount_Host[i]);
	//}
	// ��������Octree�Ѿ�cudaͬ��
}

void SparseSurfelFusion::BuildOctree::updateNodeInfo(int* BaseAddressArray_Host, DeviceBufferArray<OctNode>& NodeArray, cudaStream_t stream)
{
	BaseAddressArray_Device.ResizeArrayOrException(Constants::maxDepth_Host + 1);
	CHECKCUDA(cudaMemcpyAsync(BaseAddressArray_Device.Array().ptr(), BaseAddressArray_Host, sizeof(int) * (Constants::maxDepth_Host + 1), cudaMemcpyHostToDevice, stream));
	CHECKCUDA(cudaMemcpyToSymbolAsync(device::BaseAddressArray_Device, BaseAddressArray_Host, sizeof(int) * (Constants::maxDepth_Host + 1), 0, cudaMemcpyHostToDevice, stream));
	// D���׸��ڵ���NodeArrays��λ��ƫ�� + D��ڵ����� = ��NodeArray�Ľڵ�����
	const unsigned int totalNodeArrayLength = BaseAddressArray_Host[Constants::maxDepth_Host] + NodeArrayCount_Host[Constants::maxDepth_Host];
	//printf("NodeArray_sz = %d\n", totalNodeArrayLength);
	NodeArray.ResizeArrayOrException(totalNodeArrayLength);
	NodeArrayDepthIndex.ResizeArrayOrException(totalNodeArrayLength);
	NodeArrayNodeCenter.ResizeArrayOrException(totalNodeArrayLength);
	for (int i = 0; i <= Constants::maxDepth_Host; i++) {
		// ������������ֹδ������ɣ����������NodeArray�ͷ���
		CHECKCUDA(cudaMemcpyAsync(NodeArray.Array().ptr() + BaseAddressArray_Host[i], NodeArrays[i], sizeof(OctNode) * NodeArrayCount_Host[i], cudaMemcpyDeviceToDevice, stream));
		CHECKCUDA(cudaStreamSynchronize(stream));
		NodeArrays[i].ReleaseBuffer();
	}
	dim3 block(128);
	dim3 grid(divUp(totalNodeArrayLength, block.x));
	
	device::updateNodeArrayParentAndChildrenKernel << <grid, block, 0, stream >> > (totalNodeArrayLength, NodeArray.Array().ptr());

	device::updateEmptyNodeInfo << <grid, block, 0, stream >> > (totalNodeArrayLength, NodeArray.Array().ptr());
	device::ComputeDepthAndCenterKernel << <grid, block, 0, stream >> > (NodeArray.ArrayView(), totalNodeArrayLength, NodeArrayDepthIndex.Array().ptr(), NodeArrayNodeCenter.Array().ptr());

}

void SparseSurfelFusion::BuildOctree::computeNodeNeighbor(DeviceBufferArray<OctNode>& NodeArray, cudaStream_t stream)
{
	int Node_0_Neighs[27];	// ��0�㹹���ھ�
	for (int i = 0; i < 27; i++) {
		if (i == 13) {
			Node_0_Neighs[i] = 0;
		}
		else {
			Node_0_Neighs[i] = -1;
		}
	}
	CHECKCUDA(cudaMemcpyAsync(NodeArray[0].neighs, Node_0_Neighs, sizeof(int) * 27, cudaMemcpyHostToDevice, stream));

	for (int depth = 1; depth <= Constants::maxDepth_Host; depth++) {	// ˳�����ÿ���㣬�����ڵ��ھ�
		const unsigned int currentLevelNodeCount = NodeArrayCount_Host[depth];
		dim3 block(128);
		dim3 grid(divUp(currentLevelNodeCount, block.x));
		device::computeNodeNeighborKernel << <grid, block, 0, stream >> > (BaseAddressArray_Host[depth], currentLevelNodeCount, depth, NodeArray.Array().ptr());

	}
	CHECKCUDA(cudaStreamSynchronize(stream));	// ������Ҫ��ͬ���������㷨��Ϊ����������Ҫ�õ���ǰ������

}

void SparseSurfelFusion::BuildOctree::ComputeEncodedFunctionNodeIndex(cudaStream_t stream)
{
#ifdef CHECK_MESH_BUILD_TIME_COST
	auto start = std::chrono::high_resolution_clock::now();						// ��¼��ʼʱ���
#endif // CHECK_MESH_BUILD_TIME_COST


	const unsigned int totalNodeArrayCount = NodeArray.ArraySize(); 
	EncodedFunctionNodeIndex.ResizeArrayOrException(totalNodeArrayCount);
	dim3 block(128);
	dim3 grid(divUp(totalNodeArrayCount, block.x));
	device::computeEncodedFunctionNodeIndexKernel << <grid, block, 0, stream >> > (NodeArrayDepthIndex.ArrayView(), NodeArray.ArrayView(), totalNodeArrayCount, EncodedFunctionNodeIndex.Array().ptr());

#ifdef CHECK_MESH_BUILD_TIME_COST
	CHECKCUDA(cudaStreamSynchronize(stream));	// ��ͬ��
	auto end = std::chrono::high_resolution_clock::now();						// ��¼����ʱ���
	std::chrono::duration<double, std::milli> duration = end - start;			// ����ִ��ʱ�䣨��msΪ��λ��
	std::cout << "�������������ڵ�������ʱ��: " << duration.count() << " ms" << std::endl;		// ���
	std::cout << std::endl;
	std::cout << "-----------------------------------------------------" << std::endl;	// ���
	std::cout << std::endl;
#endif // CHECK_MESH_BUILD_TIME_COST
}


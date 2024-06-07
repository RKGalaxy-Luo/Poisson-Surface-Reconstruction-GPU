/*****************************************************************//**
 * \file   BuildOctree.cu
 * \brief  构造八叉树cuda方法实现
 * 
 * \author LUOJIAXUAN
 * \date   May 5th 2024
 *********************************************************************/
#include "BuildOctree.h"

namespace SparseSurfelFusion {
	namespace device {
		__device__ __constant__ int BaseAddressArray_Device[Constants::maxDepth_Host + 1] = { 0 };	// 记录每层节点首地址的位置

		__device__ __constant__ double eps = EPSILON;

		__device__ __constant__ int maxDepth = MAX_DEPTH_OCTREE;

		__device__ __constant__ int maxIntValue = 0x7fffffff;		// 最大int值

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
	if (idx < pointsCount) {	// 下方reduce的时候可能会访存出界(threadIdx.x + stride >= pointsCount)，如果上面直接返回的话，__shared__数组在某个线程上必然有没赋值的情况
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
	// 顺序寻址
	for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
		if (threadIdx.x < stride) {
			float lhs, rhs;
			/** 最大值 **/
			lhs = MaxPointX[threadIdx.x];
			rhs = MaxPointX[threadIdx.x + stride];
			MaxPointX[threadIdx.x] = lhs < rhs ? rhs : lhs;

			lhs = MaxPointY[threadIdx.x];
			rhs = MaxPointY[threadIdx.x + stride];
			MaxPointY[threadIdx.x] = lhs < rhs ? rhs : lhs;

			lhs = MaxPointZ[threadIdx.x];
			rhs = MaxPointZ[threadIdx.x + stride];
			MaxPointZ[threadIdx.x] = lhs < rhs ? rhs : lhs;

			/** 最小值 **/
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
	if (threadIdx.x == 0) {	// 该block的第一个线程
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
	// 通常不超过1000
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

	// 将法线放缩到[-2^(maxDepth + 1), 2^(maxDepth + 1)]这个区间，而非[-1, 1]
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
	unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;	// unsigned int 到 2 * (2^31 - 1) = 42.95亿 
	if (idx >= pointsNum)	return;
	long long key = 0ll;	// 当前稠密点的key
	Point3D<float> myCenter = Point3D<float>(0.5f, 0.5f, 0.5f);
	float myWidth = 0.25f;
	// node编码规则，Key前32位：x0y0z0 x1y1z1 ... xD-1 yD-1 zD-1  ->  D最多10层
	for (int i = device::maxDepth - 1; i >= 0; i--) {	// 从0层开始构建
		if (pos[idx].point.coords[0] > myCenter.coords[0]) {	// x在中心点右边，编码为1
			key |= 1ll << (3 * i + 34);		// 按照编码顺序将1移到对应位置
			myCenter.coords[0] += myWidth;
		}
		else {								// key默认为0，在左边则无需将编码为置为0
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
	// 既记录了node的位置，又记录了这个稠密点在verticeArray中的位置, 而且node编码在高位，排序的时候idx完全不会影响node的顺序，仍然还是octree的顺序
	keys[idx] = key + idx;	
	//if (idx % 1000 == 0) {
	//	const int keyHigher32bits = int(keys[idx] >> 32);					// Octree编码
	//	const int keyLower32bits = int(keys[idx] & ((1ll << 31) - 1));	// 在原始数组中的idx
	//	printf("idx = %d  High = %d  Low = %d\n", idx, keyHigher32bits, keyLower32bits);
	//}
}

__global__ void SparseSurfelFusion::device::updataLower32ForSortedDensePoints(const unsigned int sortedKeysCount, long long* sortedVerticesKey)
{
	const unsigned int idx = threadIdx.x + blockDim.x * blockIdx.x;
	if (idx >= sortedKeysCount)	return;
	if (idx == 0) {
		sortedVerticesKey[idx] &= ~((1ll << 32) - 1);	// 取高32位的数据，低32位置为0，因为此时DensePoint数组已经被排序，之前低32位的数据需要重置成当下的idx
	}
	else {
		sortedVerticesKey[idx] &= ~((1ll << 32) - 1);
		sortedVerticesKey[idx] += idx;						// 这里必须重置低32位的idx，之前的idx是DensePoints数组没有排序的idx，现在DensePoints已经被排序，必须更新
	}
}

__global__ void SparseSurfelFusion::device::labelSortedVerticesKeysKernel(const unsigned int sortedKeysCount, DeviceArrayView<long long> sortedVerticesKey, unsigned int* keyLabel)
{
	const unsigned int idx = threadIdx.x + blockDim.x * blockIdx.x;
	if (idx >= sortedKeysCount)	return;
	if (idx == 0) { 
		keyLabel[0] = 1; 
	}
	else {	// 因为此处需要访问前一个的idx，所以如果在同一个核函数更新一定存在访存冲突
		// 只比较Octree编码部分，因为D层节点，同父节点的部分只允许有8个叶子节点
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
	if (voxelKeyLabel[idx] == 1) {										//当编码键值与前一个不同时，此时也是第一个不同
		compactedKey[prefixsumedLabel[idx] - 1] = sortedVoxelKey[idx];	//将此时的键值给compactedKey
		compactedOffset[prefixsumedLabel[idx] - 1] = idx;				//将这个时候的偏移的idx给compactedOffset存储
		//上述保证了compactedKey与compactedOffset对应：compactedKey[i]储存与前一个不一样的编码键值，compactedOffset[i]储存这个编码键值在voxelKeyLabel中的第几个
	}
	if (idx == 0) {
		//printf("denseNodeNum = %d\n", sortedVoxelKey.size);
		// compactedVoxelKey 数量是：voxelsNum
		// compactedOffset   数量是：voxelsNum + 1
		compactedOffset[compactedOffset.Size() - 1] = sortedVoxelKey.size;	// 最后一个值记录一共有多少个有效的体素点(包括重复的体素)，
	}
}

__global__ void SparseSurfelFusion::device::initUniqueNodeKernel(OctNode* uniqueNode, const DeviceArrayView<long long> compactedKey, const unsigned int compactedNum, const int* compactedOffset)
{
	const unsigned int idx = threadIdx.x + blockDim.x * blockIdx.x;
	if (idx >= compactedNum) return;	// 压缩后的个数
	const int keyHigher32bits = int(compactedKey[idx] >> 32);					// Octree编码
	const int keyLower32bits = int(compactedKey[idx] & ((1ll << 31) - 1));		// 在原始数组中的idx
	uniqueNode[idx].key = keyHigher32bits;										// 记录节点的key
	uniqueNode[idx].pidx = compactedOffset[idx];								// 标记第一个节点的位置
	uniqueNode[idx].pnum = compactedOffset[idx + 1] - compactedOffset[idx];		// 标记相同key值节点的数量
}

__global__ void SparseSurfelFusion::device::generateNodeNumsKernel(const DeviceArrayView<long long> uniqueCode, const unsigned int nodesCount, unsigned int* nodeNums)
{
	const unsigned int idx = threadIdx.x + blockDim.x * blockIdx.x;
	if (idx >= nodesCount)	return;
	if (idx == 0) {	// 首节点设置为0，方便后面节点定位
		nodeNums[idx] = 0;
	}
	else if ((uniqueCode[idx - 1] >> 35) != (uniqueCode[idx] >> 35)) {		// 32 + 3 = 35  ->  末端节点的上一层节点  ->  父节点相同
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
	int DLevelNodeIndex = nodeAddress[idx] + (uniqueNode[idx].key & 7);	// D层节点的位置
	NodeArrayD[DLevelNodeIndex] = uniqueNode[idx];
	NodeArrayDAddressFull[DLevelNodeIndex] = nodeAddress[idx];
	const int keyLower32bits = int(compactedKey[idx] & ((1ll << 31) - 1));				// 在原始数组中的idx
	Point2NodeArrayD[keyLower32bits] = DLevelNodeIndex;									// 建立稠密点到节点的映射
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
	int val = Point2NodeArrayD[idx];	// 该稠密点对应的NodeArrays的位置，如果Point2NodeArrayD = -1那么说明该点没有在NodeArrays中
	while (val == -1) {					// 如果该点并不出现在NodeArrays中，那么就往前找最近的出现在NodeArrays中的点
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
	if (NodeArrayD[idx].pnum == 0) {							// 当前节点是空的，当前节点是无效节点【满排，有些是有效节点有些是无效节点】
		unsigned int AFatherFirstSonNode = idx - idx % 8;		// 找到当前父节点的第一个子节点
		unsigned int AFatherValidNodeIdx = -1;					// 当前父节点的第一个有效的节点
		for (int i = 0; i < 8; i++) {							// 遍历这个父节点所有子节点，直到找到有效的子节点
			AFatherValidNodeIdx = AFatherFirstSonNode + i;		// 父节点的第i个子节点idx
			if (NodeArrayD[AFatherValidNodeIdx].pnum != 0) {	// 找了了有效的节点就跳出循环
				break;
			}
		}
		const unsigned int fatherIdx = NodeAddressFull[AFatherValidNodeIdx] / 8;				// 上一层(即父节点)的排序：0  0   0 8   8    ....  -->  00000000 11111111
		if (NodeArrayD[idx].dnum != 0) {	
			atomicAdd(&uniqueNodeArrayPreviousLevel[fatherIdx].dnum, NodeArrayD[idx].dnum);		// 原子操作，计算当前节点下面有多少个点
			atomicMin(&uniqueNodeArrayPreviousLevel[fatherIdx].didx, NodeArrayD[idx].didx);		// 原子操作，找出当前节点下面idx最小的点，即当前节点的首个点
		}
	}
	else {	// 如果是有效点
		const int fatherKey = NodeArrayD[idx].key & (~(7 << (3 * (device::maxDepth - depth))));	// 当前点的父节点Key，不管是否是有效点
		const int fatherIdx = NodeAddressFull[idx] / 8;											// 当前点的父节点
		const int sonKey = (NodeArrayD[idx].key >> (3 * (device::maxDepth - depth))) & 7;		// 儿子节点的键，其实就是当前层的那三个bit值
		uniqueNodeArrayPreviousLevel[fatherIdx].key = fatherKey;								// uniqueNodeArrayPreviousLevel上一层的NodeArrays开始记录父节点
		//if (depth == device::maxDepth) printf("index = %d  fatherIdx = %d  NA_pnum = %d   Prev_pnum = %d\n", idx, fatherIdx, NodeArrayD[idx].pnum, uniqueNodeArrayPreviousLevel[fatherIdx].pnum);
		atomicAdd(&uniqueNodeArrayPreviousLevel[fatherIdx].pnum, NodeArrayD[idx].pnum);			// 上一层父节点包含当前节点的所有pnum的和，即与当前层父节点是同一个Key的稠密点个数(只有有效点才有pnum值，无效点是0)
		atomicMin(&uniqueNodeArrayPreviousLevel[fatherIdx].pidx, NodeArrayD[idx].pidx);			// 上一层父节点的pidx是其所有子节点中idx最小的，即与当前父节点是同一个Key的稠密点中最前面的那个点idx
		atomicAdd(&uniqueNodeArrayPreviousLevel[fatherIdx].dnum, NodeArrayD[idx].dnum);			// 上一层父节点的dnum是包含下面所有子节点的数量，包括有效点和无效点
		atomicMin(&uniqueNodeArrayPreviousLevel[fatherIdx].didx, NodeArrayD[idx].didx);			// 上一层父节点中didx是子节点中，在NodeArrayD中最靠前的那一个节点的idx，包括有效和无效
		uniqueNodeArrayPreviousLevel[fatherIdx].children[sonKey] = idx;							// 在有效点的时候设置其有效的孩子节点，就是当前层的有效值才能是上一层父亲的儿子，无效点不能作为儿子
	}
}

__global__ void SparseSurfelFusion::device::generateNodeNumsPreviousLevelKernel(DeviceArrayView<OctNode> uniqueNodePreviousLevel, const unsigned int uniqueCount, const unsigned int depth, unsigned int* NodeNumsPreviousLevel)
{
	const unsigned int idx = threadIdx.x + blockDim.x * blockIdx.x;
	if (idx >= uniqueCount)	return;
	if (idx == 0) {	// 第一个节点父节点与前一个无法比较，默认为0
		NodeNumsPreviousLevel[idx] = 0;
	}
	else {
		// 查看父节点是否与前一个相同，不同为8
		if ((uniqueNodePreviousLevel[idx - 1].key >> (3 * (device::maxDepth - depth + 1))) != (uniqueNodePreviousLevel[idx].key >> (3 * (device::maxDepth - depth + 1)))) {
			NodeNumsPreviousLevel[idx] = 8;
		}
		else {	// 相同为0
			NodeNumsPreviousLevel[idx] = 0;
		}
	}
}

__global__ void SparseSurfelFusion::device::generateNodeArrayPreviousLevelKernel(DeviceArrayView<OctNode> uniqueNodePreviousLevel, DeviceArrayView<unsigned int> nodeAddressPreviousLevel, const unsigned int uniqueCount, const unsigned int depth, OctNode* nodeArrayPreviousLevel, OctNode* nodeArrayD, unsigned int* NodeAddressFull)
{
	const unsigned int idx = threadIdx.x + blockDim.x * blockIdx.x;
	if (idx >= uniqueCount)	return;
	int index = nodeAddressPreviousLevel[idx] + ((uniqueNodePreviousLevel[idx].key >> (3 * (device::maxDepth - depth + 1))) & 7);	// 上一层父节点的idx
	nodeArrayPreviousLevel[index] = uniqueNodePreviousLevel[idx];
	NodeAddressFull[index] = nodeAddressPreviousLevel[idx];
	for (int i = 0; i < 8; i++) {	// children节点最多8个，并且全部赋初值0了，对每个父节点遍历全部的孩子节点，选择有效的孩子节点，将自己的index赋值给孩子
		int nodeArrayDIndex = uniqueNodePreviousLevel[idx].children[i];	// 记录上一层节点的孩子节点，即nodeArrayD的节点
		if (nodeArrayDIndex != 0) {	// 有效的孩子节点，存在一个bug：如果孩子节点在nodeArrayD中的index本身就是0，也会被当场无效点处理？
			nodeArrayD[nodeArrayDIndex].parent = index;	// 将当前节点的作为有效孩子节点的父节点
		}
	}
}

__global__ void SparseSurfelFusion::device::updateNodeArrayParentAndChildrenKernel(const unsigned int totalNodeArrayLength, OctNode* NodeArray)
{
	const unsigned int idx = threadIdx.x + blockDim.x * blockIdx.x;
	if (idx >= totalNodeArrayLength)	return;
	if (NodeArray[idx].pnum == 0)	return;		// 如果是无效点则无需更新孩子和父亲节点
	int depth = 0;
	for (depth = 0; depth < device::maxDepth; depth++) {				// 寻找idx在哪一层
		if (device::BaseAddressArray_Device[depth] <= idx && idx < device::BaseAddressArray_Device[depth + 1]) {
			break;
		}
	}
	if (idx == 0) {
		NodeArray[idx].parent = -1;										// 如果是第0层节点
#pragma unroll	// 展开循环，加速运算
		for (int child = 0; child < 8; child++) {						// 计算Children在NodeArray的位置
			NodeArray[idx].children[child] += BaseAddressArray_Device[depth + 1];
		}
	}
	else {
		NodeArray[idx].parent += BaseAddressArray_Device[depth - 1];	// 计算Parent在NodeArray中的位置

		if (depth < device::maxDepth) {										// 最后一层没有child节点
#pragma unroll	// 展开循环，加速运算
			for (int child = 0; child < 8; child++) {						// 计算Children在NodeArray的位置
				if (NodeArray[idx].children[child] != 0) {					// 孩子节点为有效节点
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
	if ((idx - 1) % 8 == 0) {	// 排除0层的一个节点，每一层节点都是8的倍数，首节点
		int nowPIdx;			// 获得有效点的pidx
		int nowDIdx;			// 获得有效点的didx
		int validIdx;			// 获得有效点的NodeArray的位置
		int commonParent;		// 获得有效点的parent
		for (validIdx = 0; validIdx < 8; validIdx++) {				// 遍历每个节点，如果该节点pnum == 0，则说明是无效节点，pnum != 0则说明有效
			if (NodeArray[idx + validIdx].pnum != 0) {				// 找到首个有效节点
				nowPIdx = NodeArray[idx + validIdx].pidx;
				nowDIdx = NodeArray[idx + validIdx].didx;
				commonParent = NodeArray[idx + validIdx].parent;
				break;
			}
		}
		int depth = 0;
		for (depth = 0; depth < device::maxDepth; depth++) {				// 寻找idx在哪一层
			if (device::BaseAddressArray_Device[depth] <= idx && idx < device::BaseAddressArray_Device[depth + 1]) {
				break;
			}
		}
		
		// 通过层数去当前层的基本编码，即当前层的同父亲编码
		int baseKey = NodeArray[idx + validIdx].key - ((NodeArray[idx + validIdx].key) & (7 << (3 * (device::maxDepth - depth))));

		for (int j = 0; j < 8; j++) {									// 再次遍历这8个点
			int index = idx + j;										// 点在NodeArray位置
			if (NodeArray[index].pnum == 0) {							// 无效点	
				for (int k = 0; k < 8; k++) {							// 无效节点的孩子节点均无效
					NodeArray[index].children[k] = -1;
				}
			}
			else {														// 有效点
				int basePos;											// 有效点的孩子节点中第一个节点在NodeArray中的index
				for (int k = 0; k < 8; k++) {							// 遍历这个有效点的所有孩子节点
					if (NodeArray[index].children[k] > 0) {				// 如果这个点的孩子是有效点
						basePos = NodeArray[index].children[k] - k;		// 找到这个孩子节点所在子树的第一个节点在NodeArray中的index
						break;
					}
				}
				for (int k = 0; k < 8; k++) {							// 有效点的孩子均应该能找到在NodeArray中的位置
					if (depth != device::maxDepth) {					// 不是最后一层
						NodeArray[index].children[k] = basePos + k;		// 每一个孩子节点都找到在NodeArray中的位置
					}
					else {
						NodeArray[index].children[k] = -1;				// 最后一层没有孩子节点
					}
				}
			}
			NodeArray[index].key = baseKey + (j << (3 * (device::maxDepth - depth)));	// 更新当前点的key
			NodeArray[index].pidx = nowPIdx;		// 首个key不同的首个稠密点在稠密点数组的index
			nowPIdx += NodeArray[index].pnum;		// 跨过key相同的稠密点

			if (depth != device::maxDepth) {		// 如果不是最后一层D层
				NodeArray[index].didx = nowDIdx;	// 深度D的首个点
				nowDIdx += NodeArray[index].dnum;	// 继续偏移
			}
			NodeArray[index].parent = commonParent;	// 不管有效无效均共享同一个parent
		}
	}
}

__global__ void SparseSurfelFusion::device::computeNodeNeighborKernel(const unsigned int left, const unsigned int thisLevelNodeCount, const unsigned int depth, OctNode* NodeArray)
{
	const unsigned int idx = threadIdx.x + blockDim.x * blockIdx.x;
	if (idx >= thisLevelNodeCount)	return;	// 0层邻居已经初始化
	const unsigned int offset = idx + left;	// 当前层的节点在NodeArray中的位置
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
	int depth = depthBuffer[idx];	// 当前idx是在哪一层
	getEncodedFunctionNodeIndex(NodeArray[idx].key, depth, NodeIndexInFunction[idx]);
}

__device__ void SparseSurfelFusion::device::getEncodedFunctionNodeIndex(const int& key, const int& CurrentDepth, int& index)
{
	/*
	 * Part_1 = (1 + (1 << (device::maxDepth + 1)) + (1 << (2 * (device::maxDepth + 1))))
	 * Code_1 = 00000000001 00000000001 00000000001	(device::maxDepth + 1为一段，分为3段)
	 * Part_2 = ((1 << CurrentDepth) - 1);		【假设CurrentDepth = 8】
	 * Code_2 = 00011111111
	 * Part_1 * Part2 <=> 将Code_2分别写到code的三个区间[0, 11], [12, 21], [22, 31]
	 * index  = 00011111111 00011111111 00011111111
	*/ 
	index = ((1 << CurrentDepth) - 1) * (1 + (1 << (device::maxDepth + 1)) + (1 << (2 * (device::maxDepth + 1))));

	/*
	 * 【假设：sonKey = 111, CurrentDepth = 8】
	 * Part_3 = 00011111111 00011111111 00011111111
	 * idx+P3 = 00100000000 00100000000 00100000000
	 * 【假设：sonKey = 101, CurrentDepth = 8】
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
	for (depth = 0; depth < device::maxDepth; depth++) {				// 寻找idx在哪一层
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
	/********************* 归约求解每个Block最大最小Point3D *********************/
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
	/********************* 将每个Block值下载到Host端求解所有点的最值 *********************/
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
		// 当下是要获取BoundingBox最长的一条边赋值给MaxEdge
		if (i == 0 || MaxEdge < (MxPoint[i] - MnPoint[i])) {
			MaxEdge = float(MxPoint[i] - MnPoint[i]);
		}
		Center.coords[i] = float(MxPoint[i] + MnPoint[i]) / 2.0f;	// 中点位置
	}
	MaxEdge *= ScaleFactor * 1.0f;	// 调整BoundingBox的尺寸
	for (int i = 0; i < DIMENSION; i++) {
		Center.coords[i] -= MaxEdge / 2.0f;	// 调整中心点的位置
	}
	dim3 block(128);
	dim3 grid(divUp(points.ArrayView().Size(), block.x));
	device::adjustPointsCoordinateAndNormalKernel << <grid, block, 0, stream >> > (points.Array().ptr(), Center, MaxEdge, points.ArrayView().Size());
}

void SparseSurfelFusion::BuildOctree::generateCode(DeviceBufferArray<OrientedPoint3D<float>>& points, DeviceBufferArray<long long>& keys, size_t count, cudaStream_t stream)
{
	sortCode.ResizeArrayOrException(count);		// 分配Array
	// 线程数应该够用
	dim3 block(128);					// block ∈ [0, 1024]
	dim3 grid(divUp(count, block.x));	// grid  ∈ [0, 2^31 - 1]
	device::generateCodeKernel << <grid, block, 0, stream >> > (points.Array().ptr(), keys.Array().ptr(), count);
}

void SparseSurfelFusion::BuildOctree::sortAndCompactVerticesKeys(DeviceArray<OrientedPoint3D<float>>& points, cudaStream_t stream)
{
	const unsigned int VerticesKeysNum = sortCode.ArrayView().Size();
	pointKeySort.Sort(sortCode.Array(), points, stream);
	// 将point彻底更换为排列好的稠密点
	CHECKCUDA(cudaMemcpyAsync(points.ptr(), pointKeySort.valid_sorted_value.ptr(), sizeof(OrientedPoint3D<float>) * VerticesKeysNum, cudaMemcpyDeviceToDevice, stream));
	CHECKCUDA(cudaMemcpyAsync(sortCode.Array().ptr(), pointKeySort.valid_sorted_key.ptr(), sizeof(long long) * VerticesKeysNum, cudaMemcpyDeviceToDevice, stream));

	//std::vector<long long> sortedCodeHost;
	//sortCode.ArrayView().Download(sortedCodeHost);
	//for (int i = 0; i < sortedCodeHost.size(); i++) {
	//	if (i % 1000 == 0) printf("index = %d   SortedCode = %lld\n", i, sortedCodeHost[i]);
	//}

	//printf("TotalSurfelCount = %d\n", VerticesKeysNum);

	/** 排序好的SortCode没有问题 **/

	keyLabel.ResizeArrayOrException(VerticesKeysNum);
	dim3 block(128);
	dim3 grid(divUp(VerticesKeysNum, block.x));
	device::updataLower32ForSortedDensePoints << <grid, block, 0, stream >> > (VerticesKeysNum, sortCode.Array().ptr());
	device::labelSortedVerticesKeysKernel << <grid, block, 0, stream >> > (VerticesKeysNum, sortCode.ArrayView(), keyLabel.ArrayHandle());
	nodeNumsPrefixsum.InclusiveSum(keyLabel.ArrayView(), stream);

	//查询体素数(CPU中声明的数)
	unsigned int ValidKeysNum;	// 有效的键数量，Octree子节点为同一个的点应该是无效的
	//前缀和的GPU地址给prefixsum_label
	const DeviceArray<unsigned int>& prefixsumLabel = nodeNumsPrefixsum.valid_prefixsum_array;

	//将前缀和从GPU中拷贝到ValidKeysNum中，标记到最后一个就是有效的Key的总数量
	CHECKCUDA(cudaMemcpyAsync(&ValidKeysNum, prefixsumLabel.ptr() + prefixsumLabel.size() - 1, sizeof(unsigned int), cudaMemcpyDeviceToHost, stream));
	CHECKCUDA(cudaStreamSynchronize(stream));
	//printf("稠密点数量：%d   Unique点数量 = %d\n", VerticesKeysNum, ValidKeysNum);
	//构造压缩数组
	uniqueCode.ResizeArrayOrException(ValidKeysNum);						//给compactedVoxelKey开辟空间
	compactedVerticesOffset.ResizeArrayOrException(ValidKeysNum + 1);		//给compactedVoxelOffset开辟空间
	device::compactedVoxelKeyKernel << <grid, block, 0, stream >> > (sortCode.Array(), keyLabel, prefixsumLabel, uniqueCode, compactedVerticesOffset.ArrayHandle());

	/** 低32位有问题 **/
	
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

	//printf("压缩后的点 = %d\n", nodesCount);
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

	NodeAddress.ResizeArrayOrException(nodesCount);	// 开辟空间

	unsigned int* NodeAddressPtr = NodeAddress.Array().ptr();		// 暴露指针
	unsigned int* prefixsumNodeNumsPtr = nodeNumsPrefixsum.valid_prefixsum_array.ptr();	// 暴露指针
	// 将前缀和数据拷贝到nodeAddress中
	CHECKCUDA(cudaMemcpyAsync(NodeAddressPtr, prefixsumNodeNumsPtr, sizeof(unsigned int) * nodesCount, cudaMemcpyDeviceToDevice, stream));
}

void SparseSurfelFusion::BuildOctree::buildNodeArrayD(DeviceArrayView<OrientedPoint3D<float>> denseVertices, DeviceArrayView<OctNode> uniqueNode, DeviceArrayView<long long> compactedKey, DeviceBufferArray<unsigned int>& NodeAddress, DeviceBufferArray<unsigned int>& NodeAddressFull, DeviceBufferArray<int>& Point2NodeArray, DeviceBufferArray<OctNode>& NodeArrayD, cudaStream_t stream)
{
	const unsigned int nodesCount = NodeAddress.ArrayView().Size();		// Unique的数量，压缩后的
	//printf("nodesCount = %u\n", nodesCount);
	const unsigned int verticesCount = denseVertices.Size();
	unsigned int TotalNodeNums;		// 总共D层子节点的个数
	const unsigned int* nodeAddressPtr = NodeAddress.Array().ptr();
	CHECKCUDA(cudaMemcpyAsync(&TotalNodeNums, nodeAddressPtr + nodesCount - 1, sizeof(unsigned int), cudaMemcpyDeviceToHost, stream));
	CHECKCUDA(cudaStreamSynchronize(stream));	// 流同步
	TotalNodeNums += 8;	// 首节点那对应的8个没算上，这里应该表示的是最后一层(D层)的节点
	//printf("D层总节点数目 = %u\n", TotalNodeNums);
	NodeArrayD.AllocateBuffer(TotalNodeNums);
	NodeArrayD.ResizeArrayOrException(TotalNodeNums);
	NodeAddressFull.ResizeArrayOrException(TotalNodeNums);
	Point2NodeArray.ResizeArrayOrException(verticesCount);
	CHECKCUDA(cudaMemsetAsync(NodeArrayD.Array().ptr(), 0, sizeof(OctNode) * TotalNodeNums, stream));			// NodeArrayD默认都为0
	CHECKCUDA(cudaMemsetAsync(Point2NodeArray.Array().ptr(), -1, sizeof(int) * verticesCount, stream));			// Point2NodeArray默认都为-1
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
	unsigned int allNodeNums_D = NodeArrays[Constants::maxDepth_Host].ArrayView().Size();		// 记录D层节点8个满排的节点数量
	//printf("第 D 层Octree节点的总数 = %d\n", allNodeNums_D);
	unsigned int TotalNodeNumsPreviousLevel;													// 上一层父节点的个数
	for (int depth = Constants::maxDepth_Host; depth >= 1; depth--) {
		NodeArrayCount_Host[depth] = allNodeNums_D;
		int UniqueCountPrevious = allNodeNums_D / 8;	// 上一层Octree节点的数量【满排】,实际上是Unique
		OctNode* previousLevelUniqueNodePtr = uniqueNodePrevious.Array().ptr();	// 获得指针以便赋初值
		unsigned int* previousLevelnodeAddress = nodeAddressPrevious.Array().ptr();	// 获得指针以便赋初值
		// 给uniqueNodePreviousLevel开辟空间并赋初值
		uniqueNodePrevious.ResizeArrayOrException(UniqueCountPrevious);
		CHECKCUDA(cudaMemsetAsync(previousLevelUniqueNodePtr, 0, sizeof(OctNode) * UniqueCountPrevious, stream));
		// 给nodeAddressPreciousLevel开辟空间并赋初值
		nodeAddressPrevious.ResizeArrayOrException(UniqueCountPrevious);
		CHECKCUDA(cudaMemsetAsync(previousLevelnodeAddress, 0, sizeof(unsigned int) * UniqueCountPrevious, stream));

		dim3 block_PreLevel(128);
		dim3 grid_PreLevel(divUp(UniqueCountPrevious, block_PreLevel.x));
		device::setPidxDidxInvalidValue << <grid_PreLevel, block_PreLevel, 0, stream >> > (uniqueNodePrevious.Array().ptr(), UniqueCountPrevious);

		dim3 block_D(128);
		dim3 grid_D(divUp(allNodeNums_D, block_D.x));
		device::generateUniqueNodeArrayPreviousLevelKernel << <grid_D, block_D, 0, stream >> > (NodeArrays[depth].ArrayView(), NodeAddressFull.ArrayView(), allNodeNums_D, depth, uniqueNodePrevious.Array().ptr());
		
		// 每一层的UniqueNodePrev的pidx，pnum，didx，dnum都没问题
		nodeNums.ResizeArrayOrException(UniqueCountPrevious);	// 调整nodeNums为上一层的Unique大小
		device::generateNodeNumsPreviousLevelKernel << <grid_PreLevel, block_PreLevel, 0, stream >> > (uniqueNodePrevious.ArrayView(), UniqueCountPrevious, depth - 1, nodeNums.Array().ptr());
			
		CHECKCUDA(cudaStreamSynchronize(stream));	// 流同步
		//printf("UniqueCountPrevious = %d\n", UniqueCountPrevious);

		// 上一层父节点的数量一定少于当前层节点，因此直接服用nodeNumsPrefixsum，无需再分配缓存
		nodeNumsPrefixsum.InclusiveSum(nodeNums.ArrayView(), stream);
		void* prefixsumNodeNumsPtr = nodeNumsPrefixsum.valid_prefixsum_array.ptr();	// 暴露指针
		// 将前缀和数据拷贝到nodeAddress中
		CHECKCUDA(cudaMemcpyAsync(previousLevelnodeAddress, prefixsumNodeNumsPtr, sizeof(unsigned int) * UniqueCountPrevious, cudaMemcpyDeviceToDevice, stream));
		
		if (depth > 1) {	// 非第一层
			CHECKCUDA(cudaMemcpyAsync(&TotalNodeNumsPreviousLevel, previousLevelnodeAddress + UniqueCountPrevious - 1, sizeof(unsigned int), cudaMemcpyDeviceToHost, stream));
			TotalNodeNumsPreviousLevel += 8;			// 上一层父节点的数量，首个父节点为0，因此累和后应该+8
			NodeArrays[depth - 1].AllocateBuffer(TotalNodeNumsPreviousLevel);
			NodeArrays[depth - 1].ResizeArrayOrException(TotalNodeNumsPreviousLevel);
			NodeAddressFull.ResizeArrayOrException(TotalNodeNumsPreviousLevel);			// (depth - 1)层满排节点NodeAddress表，确定上一层Unique的index
			void* previousLevelNodeArrayPtr = NodeArrays[depth - 1].Array().ptr();		// 获得指针以便赋初值
			CHECKCUDA(cudaMemsetAsync(previousLevelNodeArrayPtr, 0, sizeof(OctNode) * TotalNodeNumsPreviousLevel, stream));

			// 构建上一层父节点的nodeArray，并给当前层的nodeArray中的parent赋值
			device::generateNodeArrayPreviousLevelKernel << <grid_PreLevel, block_PreLevel, 0, stream >> > (uniqueNodePrevious.ArrayView(), nodeAddressPrevious.ArrayView(), UniqueCountPrevious, depth, NodeArrays[depth - 1].Array().ptr(), NodeArrays[depth].Array().ptr(), NodeAddressFull.Array().ptr());
			
			//CHECKCUDA(cudaStreamSynchronize(stream));	// 流同步
			//printf("TotalNodeNumsPreviousLevel = %d\n", TotalNodeNumsPreviousLevel);

		}
		else {
			TotalNodeNumsPreviousLevel = 1;
			NodeArrays[depth - 1].AllocateBuffer(TotalNodeNumsPreviousLevel);
			NodeArrays[depth - 1].ResizeArrayOrException(TotalNodeNumsPreviousLevel);
			OctNode* NodeArrays_0 = NodeArrays[depth - 1].Array().ptr();	// 暴露指针，直接赋值
			CHECKCUDA(cudaMemcpyAsync(NodeArrays_0, uniqueNodePrevious.Array().ptr(), sizeof(OctNode) * 1, cudaMemcpyDeviceToDevice, stream));
		}
		allNodeNums_D = TotalNodeNumsPreviousLevel;
	}

	CHECKCUDA(cudaStreamSynchronize(stream));	// 流同步

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
	// 上述构建Octree已经cuda同步
}

void SparseSurfelFusion::BuildOctree::updateNodeInfo(int* BaseAddressArray_Host, DeviceBufferArray<OctNode>& NodeArray, cudaStream_t stream)
{
	BaseAddressArray_Device.ResizeArrayOrException(Constants::maxDepth_Host + 1);
	CHECKCUDA(cudaMemcpyAsync(BaseAddressArray_Device.Array().ptr(), BaseAddressArray_Host, sizeof(int) * (Constants::maxDepth_Host + 1), cudaMemcpyHostToDevice, stream));
	CHECKCUDA(cudaMemcpyToSymbolAsync(device::BaseAddressArray_Device, BaseAddressArray_Host, sizeof(int) * (Constants::maxDepth_Host + 1), 0, cudaMemcpyHostToDevice, stream));
	// D层首个节点在NodeArrays的位置偏移 + D层节点数量 = 总NodeArray的节点数量
	const unsigned int totalNodeArrayLength = BaseAddressArray_Host[Constants::maxDepth_Host] + NodeArrayCount_Host[Constants::maxDepth_Host];
	//printf("NodeArray_sz = %d\n", totalNodeArrayLength);
	NodeArray.ResizeArrayOrException(totalNodeArrayLength);
	NodeArrayDepthIndex.ResizeArrayOrException(totalNodeArrayLength);
	NodeArrayNodeCenter.ResizeArrayOrException(totalNodeArrayLength);
	for (int i = 0; i <= Constants::maxDepth_Host; i++) {
		// 阻塞拷贝，防止未拷贝完成，但是下面的NodeArray释放了
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
	int Node_0_Neighs[27];	// 第0层构建邻居
	for (int i = 0; i < 27; i++) {
		if (i == 13) {
			Node_0_Neighs[i] = 0;
		}
		else {
			Node_0_Neighs[i] = -1;
		}
	}
	CHECKCUDA(cudaMemcpyAsync(NodeArray[0].neighs, Node_0_Neighs, sizeof(int) * 27, cudaMemcpyHostToDevice, stream));

	for (int depth = 1; depth <= Constants::maxDepth_Host; depth++) {	// 顺序遍历每个层，构建节点邻居
		const unsigned int currentLevelNodeCount = NodeArrayCount_Host[depth];
		dim3 block(128);
		dim3 grid(divUp(currentLevelNodeCount, block.x));
		device::computeNodeNeighborKernel << <grid, block, 0, stream >> > (BaseAddressArray_Host[depth], currentLevelNodeCount, depth, NodeArray.Array().ptr());

	}
	CHECKCUDA(cudaStreamSynchronize(stream));	// 这里需要流同步，后续算法分为两个流，均要用到当前计算结果

}

void SparseSurfelFusion::BuildOctree::ComputeEncodedFunctionNodeIndex(cudaStream_t stream)
{
#ifdef CHECK_MESH_BUILD_TIME_COST
	auto start = std::chrono::high_resolution_clock::now();						// 记录开始时间点
#endif // CHECK_MESH_BUILD_TIME_COST


	const unsigned int totalNodeArrayCount = NodeArray.ArraySize(); 
	EncodedFunctionNodeIndex.ResizeArrayOrException(totalNodeArrayCount);
	dim3 block(128);
	dim3 grid(divUp(totalNodeArrayCount, block.x));
	device::computeEncodedFunctionNodeIndexKernel << <grid, block, 0, stream >> > (NodeArrayDepthIndex.ArrayView(), NodeArray.ArrayView(), totalNodeArrayCount, EncodedFunctionNodeIndex.Array().ptr());

#ifdef CHECK_MESH_BUILD_TIME_COST
	CHECKCUDA(cudaStreamSynchronize(stream));	// 流同步
	auto end = std::chrono::high_resolution_clock::now();						// 记录结束时间点
	std::chrono::duration<double, std::milli> duration = end - start;			// 计算执行时间（以ms为单位）
	std::cout << "计算编码基函数节点索引的时间: " << duration.count() << " ms" << std::endl;		// 输出
	std::cout << std::endl;
	std::cout << "-----------------------------------------------------" << std::endl;	// 输出
	std::cout << std::endl;
#endif // CHECK_MESH_BUILD_TIME_COST
}


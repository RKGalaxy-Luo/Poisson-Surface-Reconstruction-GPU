/*****************************************************************//**
 * \file   LaplacianSolver.cu
 * \brief  拉普拉斯求解器cuda方法实现
 * 
 * \author LUOJIAXUAN
 * \date   May 26th 2024
 *********************************************************************/
#include "LaplacianSolver.h"
#if defined(__CUDACC__)		//如果由NVCC编译器编译
#include <cub/cub.cuh>
#endif

namespace SparseSurfelFusion {
	namespace device {
		__device__ __constant__ int res = RESOLUTION;
		
		__device__ __constant__ int maxDepth = MAX_DEPTH_OCTREE;
		
		__device__ __constant__ double eps = EPSILON;

		__device__ __constant__ int decodeOffset_1 = (1 << (MAX_DEPTH_OCTREE + 1));

		__device__ __constant__ int decodeOffset_2 = (1 << (2 * (MAX_DEPTH_OCTREE + 1)));
	}
}

__global__ void SparseSurfelFusion::device::GenerateSingleNodeLaplacian(const unsigned int depth, DeviceArrayView<double> dot_F_F, DeviceArrayView<double> dot_F_D2F, DeviceArrayView<int> encodeNodeIndexInFunction, DeviceArrayView<OctNode> NodeArray, const unsigned int begin, const unsigned int calculatedNodeNum, int* rowCount, int* colIndex, float* val)
{
	const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= calculatedNodeNum) return;
	const unsigned int offset = begin + idx;
	int count = 0;
	int colStart = idx * 27;
	int idxO_1[3];
	int encodeIndex = encodeNodeIndexInFunction[offset];
	idxO_1[0] = encodeIndex % device::decodeOffset_1;
	idxO_1[1] = (encodeIndex / device::decodeOffset_1) % device::decodeOffset_1;
	idxO_1[2] = encodeIndex / device::decodeOffset_2;

	//if (depth == 1) {
	//	printf("idx = %d   idxO_1 = (%d, %d, %d)\n", idx, idxO_1[0], idxO_1[1], idxO_1[2]);
	//}

	for (int i = 0; i < 27; i++) {
		int neighbor = NodeArray[offset].neighs[i];	// 节点的邻居节点
		if (neighbor == -1) continue;
		int colIdx = neighbor - begin;				// 相对于邻居的偏移
		int idxO_2[3];
		encodeIndex = encodeNodeIndexInFunction[neighbor];
		idxO_2[0] = encodeIndex % device::decodeOffset_1;
		idxO_2[1] = (encodeIndex / device::decodeOffset_1) % device::decodeOffset_1;
		idxO_2[2] = encodeIndex / device::decodeOffset_2;
		//if (depth == 1 && idx == 0) {
		//	printf("idx = %d   offset = %d   neighborIdx = %d   neighbor = %d   idxO_2 = (%d, %d, %d)\n", idx, offset, i, neighbor, idxO_2[0], idxO_2[1], idxO_2[2]);
		//}
		int scratch[3];
		scratch[0] = idxO_1[0] * device::res + idxO_2[0];
		scratch[1] = idxO_1[1] * device::res + idxO_2[1];
		scratch[2] = idxO_1[2] * device::res + idxO_2[2];

		double LaplacianEntryValue = GetLaplacianEntry(dot_F_F, dot_F_D2F, scratch);
		if (fabs(LaplacianEntryValue) > device::eps) {
			colIndex[colStart + count] = colIdx;
			val[colStart + count] = LaplacianEntryValue;
			count++;
		}
	}
	rowCount[idx] = count;
}

__device__ double SparseSurfelFusion::device::GetLaplacianEntry(DeviceArrayView<double> dot_F_F, DeviceArrayView<double> dot_F_D2F, const int* index)
{
	return double(dot_F_F[index[0]] * dot_F_F[index[1]] * dot_F_F[index[2]] * (dot_F_D2F[index[0]] + dot_F_D2F[index[1]] + dot_F_D2F[index[2]]));
}

__global__ void SparseSurfelFusion::device::markValidColIndex(const int* colIndex, const unsigned int nodeNum, bool* flag)
{
	const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= nodeNum) return;
	bool valid = false;
	if (colIndex[idx] >= 0) { valid = true;}
	flag[idx] = valid;
}

__global__ void SparseSurfelFusion::device::CalculatePointsImplicitFunctionValueKernel(DeviceArrayView<OrientedPoint3D<float>> DensePoints, DeviceArrayView<int> PointToNodeArrayDLevel, DeviceArrayView<OctNode> NodeArray, DeviceArrayView<int> encodeNodeIndexInFunction, DeviceArrayView<ConfirmedPPolynomial<CONVTIMES + 1, CONVTIMES + 2>> BaseFunctions, DeviceArrayView<float> dx, const unsigned int DLevelOffset, const unsigned int DenseVertexCount, float* pointsValue)
{
	const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= DenseVertexCount) return;
	int nowNode = DLevelOffset + PointToNodeArrayDLevel[idx];
	float val = 0.0f;
	Point3D<float> samplePoint = DensePoints[idx].point;
	while (nowNode != -1) {
		for (int i = 0; i < 27; i++) {
			int neighbor = NodeArray[nowNode].neighs[i];
			if (neighbor != -1) {
				int idxO[3];
				int encodeIndex = encodeNodeIndexInFunction[neighbor];
				idxO[0] = encodeIndex % decodeOffset_1;
				idxO[1] = (encodeIndex / decodeOffset_1) % decodeOffset_1;
				idxO[2] = encodeIndex / decodeOffset_2;

				ConfirmedPPolynomial<CONVTIMES + 1, CONVTIMES + 2> funcX = BaseFunctions[idxO[0]];
				ConfirmedPPolynomial<CONVTIMES + 1, CONVTIMES + 2> funcY = BaseFunctions[idxO[1]];
				ConfirmedPPolynomial<CONVTIMES + 1, CONVTIMES + 2> funcZ = BaseFunctions[idxO[2]];

				val += dx[neighbor] * value(funcX, samplePoint.coords[0]) * value(funcY, samplePoint.coords[1]) * value(funcZ, samplePoint.coords[2]);
			}
		}
		nowNode = NodeArray[nowNode].parent;
		//if (idx == 1000) {
		//	printf("NowNodeIndex = %d\n", nowNode);
		//}
	}
	pointsValue[idx] = val;
	//if (idx % 1000 == 0) {
	//	printf("index = %d   PointValue = %.9f\n", idx, val);
	//}
}

void SparseSurfelFusion::LaplacianSolver::LaplacianCGSolver(const int* BaseAddressArray, const int* NodeArrayCount, DeviceArrayView<int> encodeNodeIndexInFunction, DeviceArrayView<OctNode> NodeArray, float* Divergence, DeviceArrayView<double> dot_F_F, DeviceArrayView<double> dot_F_D2F, cudaStream_t stream)
{
#ifdef CHECK_MESH_BUILD_TIME_COST
	auto start = std::chrono::high_resolution_clock::now();						// 记录开始时间点
#endif // CHECK_MESH_BUILD_TIME_COST

	dx.ResizeArrayOrException(NodeArray.Size());

	for (int depth = 0; depth <= Constants::maxDepth_Host; depth++) {
		int CurrentLevelNodesNum = NodeArrayCount[depth];	// 当前层节点总数
		int CurrentLevelNodesNum_27 = CurrentLevelNodesNum * 27;
		//cudaStream_t stream = streams[depth];
		int* rowCount = NULL;	// 【中间变量，用完即释放，初始值为0】记录当前节点的邻居节点有多少个满足构成Laplace矩阵的元素 value ∈ [0, 26]
		CHECKCUDA(cudaMallocAsync(reinterpret_cast<void**>(&rowCount), sizeof(int) * (CurrentLevelNodesNum + 2), stream));
		CHECKCUDA(cudaMemsetAsync(rowCount, 0, sizeof(int) * (CurrentLevelNodesNum + 2), stream));

		int* colIndex = NULL;	// 【中间变量，用完即释放，初始值为-1】记录当前节点的邻居中满足"构成Laplace矩阵的元素"这一条件的节点，距离每层首节点的距离(neighbor - begin)
		CHECKCUDA(cudaMallocAsync(reinterpret_cast<void**>(&colIndex), sizeof(int) * CurrentLevelNodesNum_27, stream));
		CHECKCUDA(cudaMemsetAsync(colIndex, -1, sizeof(int) * CurrentLevelNodesNum_27, stream));

		float* val = NULL;		// 【中间变量，用完即释放】记录节点的所有邻居的Laplace元素的值，与colIndddex对应
		CHECKCUDA(cudaMallocAsync(reinterpret_cast<void**>(&val), sizeof(float) * CurrentLevelNodesNum_27, stream));

		dim3 block_1(128);
		dim3 grid_1(divUp(CurrentLevelNodesNum, block_1.x));
		device::GenerateSingleNodeLaplacian << <grid_1, block_1, 0, stream >> > (depth, dot_F_F, dot_F_D2F, encodeNodeIndexInFunction, NodeArray, BaseAddressArray[depth], NodeArrayCount[depth], rowCount + 1, colIndex, val);

		int* RowBaseAddress = NULL;	//【中间变量，用完即释放】记录当前节点之前的节点(及其邻居)，满足Laplace元素值的，总共有多少个(不计算当前节点，并且是从index=1开始，而非index=0)
		CHECKCUDA(cudaMallocAsync(reinterpret_cast<void**>(&RowBaseAddress), sizeof(int) * (CurrentLevelNodesNum + 2), stream));

		void* tempRowAddressStorage = NULL;	//【算法临时变量，用完即释放】排他前缀和的临时变量
		size_t tempRowAddressStorageBytes = 0;
		cub::DeviceScan::ExclusiveSum(tempRowAddressStorage, tempRowAddressStorageBytes, rowCount, RowBaseAddress, CurrentLevelNodesNum + 2, stream);
		CHECKCUDA(cudaMallocAsync(&tempRowAddressStorage, tempRowAddressStorageBytes, stream));
		cub::DeviceScan::ExclusiveSum(tempRowAddressStorage, tempRowAddressStorageBytes, rowCount, RowBaseAddress, CurrentLevelNodesNum + 2, stream);
		int valNums;				// 记录一共有多少个有效的Laplace元素，为了给MergedColIndex和MergedVal分配内存
		int lastRowNum;				// 记录最后一个节点的有效的Laplace元素
		CHECKCUDA(cudaMemcpyAsync(&valNums, RowBaseAddress + CurrentLevelNodesNum, sizeof(int), cudaMemcpyDeviceToHost, stream));
		CHECKCUDA(cudaMemcpyAsync(&lastRowNum, rowCount + CurrentLevelNodesNum, sizeof(int), cudaMemcpyDeviceToHost, stream));
		CHECKCUDA(cudaStreamSynchronize(stream));	// 阻塞Host线程，这里必须等待valNums计算完成，才能动态分配内存
		valNums += lastRowNum;		// 前面排他前缀和并未包含最后一个元素，第index = n + 1个应该是[0, n]个的和，这里rowCount的第0个一直是0，从1开始计算rowCount的
		CHECKCUDA(cudaMemcpyAsync(RowBaseAddress + CurrentLevelNodesNum + 1, &valNums, sizeof(int), cudaMemcpyHostToDevice, stream));	// 补齐index = n + 1个元素
		
		//CHECKCUDA(cudaStreamSynchronize(stream));	// 阻塞Host线程，这里必须等待valNums计算完成，才能动态分配内存
		//printf("depth = %d   valNums = %d\n", depth, valNums);

		int* MergedColIndex = NULL;	// 【中间变量，用完即释放】压缩的节点及邻居位置的有效值
		CHECKCUDA(cudaMallocAsync(reinterpret_cast<void**>(&MergedColIndex), sizeof(int) * valNums, stream));

		float* MergedVal = NULL;	// 【中间变量，用完即释放】压缩的节点及邻居有效的Laplace值
		CHECKCUDA(cudaMallocAsync(reinterpret_cast<void**>(&MergedVal), sizeof(float) * valNums, stream));

		bool* flag = NULL;			// 【中间变量，用完即释放】标记当前节点的邻居中满足"构成Laplace矩阵的元素"这一条件的节点的位置
		CHECKCUDA(cudaMallocAsync(reinterpret_cast<void**>(&flag), sizeof(bool) * CurrentLevelNodesNum_27, stream));

		dim3 block_2(128);
		dim3 grid_2(divUp(CurrentLevelNodesNum_27, block_2.x));
		device::markValidColIndex << <grid_2, block_2, 0, stream >> > (colIndex, CurrentLevelNodesNum_27, flag);

		int* MergedValSelectedNum = NULL;		// 【中间变量，用完即释放】获得压缩val后数量，理论上应该和valNums相等
		CHECKCUDA(cudaMallocAsync(reinterpret_cast<void**>(&MergedValSelectedNum), sizeof(int), stream));
		int* MergedColIndexSelectedNum = NULL;	// 【中间变量，用完即释放】获得压缩colIndex后的数量，理论上应该和valNums相等
		CHECKCUDA(cudaMallocAsync(reinterpret_cast<void**>(&MergedColIndexSelectedNum), sizeof(int), stream));

		void* d_temp_storage_1 = NULL;
		size_t temp_storage_bytes_1 = 0;
		CHECKCUDA(cub::DeviceSelect::Flagged(d_temp_storage_1, temp_storage_bytes_1, val, flag, MergedVal, MergedValSelectedNum, CurrentLevelNodesNum_27, stream, false));	// 确定临时设备存储需求
		CHECKCUDA(cudaMallocAsync(&d_temp_storage_1, temp_storage_bytes_1, stream));	
		CHECKCUDA(cub::DeviceSelect::Flagged(d_temp_storage_1, temp_storage_bytes_1, val, flag, MergedVal, MergedValSelectedNum, CurrentLevelNodesNum_27, stream, false));	// 筛选																				

		void* d_temp_storage_2 = NULL;
		size_t temp_storage_bytes_2 = 0;
		CHECKCUDA(cub::DeviceSelect::Flagged(d_temp_storage_2, temp_storage_bytes_2, colIndex, flag, MergedColIndex, MergedColIndexSelectedNum, CurrentLevelNodesNum_27, stream, false));
		CHECKCUDA(cudaMallocAsync(&d_temp_storage_2, temp_storage_bytes_2, stream)); 			 
		CHECKCUDA(cub::DeviceSelect::Flagged(d_temp_storage_2, temp_storage_bytes_2, colIndex, flag, MergedColIndex, MergedColIndexSelectedNum, CurrentLevelNodesNum_27, stream, false));

		int MergedValSelectedNum_Host;			// 获得压缩val后数量，理论上应该和valNums相等
		int MergedColIndexSelectedNum_Host;		// 获得压缩colIndex后的数量，理论上应该和valNums相等
		CHECKCUDA(cudaMemcpyAsync(&MergedValSelectedNum_Host, MergedValSelectedNum, sizeof(int), cudaMemcpyDeviceToHost, stream));
		CHECKCUDA(cudaMemcpyAsync(&MergedColIndexSelectedNum_Host, MergedColIndexSelectedNum, sizeof(int), cudaMemcpyDeviceToHost, stream));
		
		CHECKCUDA(cudaStreamSynchronize(stream));	// 流同步

		assert(MergedValSelectedNum_Host == valNums);		// 顺便Check一下压缩的是否正确
		assert(MergedColIndexSelectedNum_Host == valNums);	// 顺便Check一下压缩的是否正确

		//if (depth == 1) {
		//	std::vector<int> MergedColIndexHost;
		//	MergedColIndexHost.resize(valNums);
		//	std::vector<float> MergedValHost;
		//	MergedValHost.resize(valNums);
		//	CHECKCUDA(cudaMemcpyAsync(MergedColIndexHost.data(), MergedColIndex, sizeof(int) * valNums, cudaMemcpyDeviceToHost, stream));
		//	CHECKCUDA(cudaMemcpyAsync(MergedValHost.data(), MergedVal, sizeof(float) * valNums, cudaMemcpyDeviceToHost, stream));
		//	printf("valNum = %d\n", valNums);
		//	for (int i = 0; i < valNums; i++) {
		//		if ((i + 1) % 8 == 0) {
		//			printf("%10.5f\n",MergedValHost[i]);
		//		}
		//		else {
		//			printf("%10.5f   ",MergedValHost[i]);
		//		}
		//	}
		//}

#ifdef CHECK_MESH_BUILD_TIME_COST
		printf("第 %d 层节点的", depth);
#endif // CHECK_MESH_BUILD_TIME_COST
		solverCG_DeviceToDevice(CurrentLevelNodesNum, valNums, RowBaseAddress + 1, MergedColIndex, MergedVal, Divergence + BaseAddressArray[depth], dx.Ptr() + BaseAddressArray[depth], stream);
	
		cudaFreeAsync(tempRowAddressStorage, stream);	 // 临时空间
		cudaFreeAsync(d_temp_storage_1, stream);		 // 临时空间
		cudaFreeAsync(d_temp_storage_2, stream);		 // 临时空间

		cudaFreeAsync(rowCount, stream);
		cudaFreeAsync(colIndex, stream);
		cudaFreeAsync(val, stream);
		cudaFreeAsync(RowBaseAddress, stream);

		cudaFreeAsync(MergedColIndex, stream);
		cudaFreeAsync(MergedVal, stream);
		cudaFreeAsync(flag, stream); 

		cudaFreeAsync(MergedValSelectedNum, stream);
		cudaFreeAsync(MergedColIndexSelectedNum, stream);
	}



	//CHECKCUDA(cudaStreamSynchronize(stream));	// 流同步
	//std::vector<float>dxHost;
	//dx.ArrayView().Download(dxHost);
	//for (int i = 0; i < dxHost.size(); i++) {
	//	if (i < 1000 || i % 1000 == 0) {
	//		printf("index = %d   dx = %.9f\n", i, dxHost[i]);
	//	}
	//}


#ifdef CHECK_MESH_BUILD_TIME_COST
	CHECKCUDA(cudaStreamSynchronize(stream));	// 流同步
	auto end = std::chrono::high_resolution_clock::now();						// 记录结束时间点
	std::chrono::duration<double, std::milli> duration = end - start;			// 计算执行时间（以ms为单位）
	std::cout << "拉普拉斯求解的时间: " << duration.count() << " ms" << std::endl;		// 输出
	std::cout << std::endl;
	std::cout << "-----------------------------------------------------" << std::endl;	// 输出
	std::cout << std::endl;
#endif // CHECK_MESH_BUILD_TIME_COST
}

void SparseSurfelFusion::LaplacianSolver::CalculatePointsImplicitFunctionValue(DeviceArrayView<OrientedPoint3D<float>> DensePoints, DeviceArrayView<int> PointToNodeArrayDLevel, DeviceArrayView<OctNode> NodeArray, DeviceArrayView<int> encodeNodeIndexInFunction, DeviceArrayView<ConfirmedPPolynomial<CONVTIMES + 1, CONVTIMES + 2>> BaseFunctions, const unsigned int DLevelOffset, const unsigned int DenseVertexCount, cudaStream_t stream)
{
#ifdef CHECK_MESH_BUILD_TIME_COST
	auto start = std::chrono::high_resolution_clock::now();						// 记录开始时间点
#endif // CHECK_MESH_BUILD_TIME_COST

	dim3 block(128);
	dim3 grid(divUp(DenseVertexCount, block.x));
	device::CalculatePointsImplicitFunctionValueKernel << <grid, block, 0, stream >> > (DensePoints, PointToNodeArrayDLevel, NodeArray, encodeNodeIndexInFunction, BaseFunctions, dx.ArrayView(), DLevelOffset, DenseVertexCount, DensePointsImplicitFunctionValue.Array().ptr());

	// 规约加法
	float* isoValueDevice = NULL;
	CHECKCUDA(cudaMallocAsync(reinterpret_cast<void**>(&isoValueDevice), sizeof(float), stream));
	void* d_temp_storage = NULL;
	size_t temp_storage_bytes = 0;
	cub::DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, DensePointsImplicitFunctionValue.Array().ptr(), isoValueDevice, DenseVertexCount, stream);
	CHECKCUDA(cudaMallocAsync(&d_temp_storage, temp_storage_bytes, stream));
	cub::DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, DensePointsImplicitFunctionValue.Array().ptr(), isoValueDevice, DenseVertexCount, stream);
	CHECKCUDA(cudaMemcpyAsync(&isoValue, isoValueDevice, sizeof(float), cudaMemcpyDeviceToHost, stream));
	CHECKCUDA(cudaStreamSynchronize(stream));
	isoValue /= DenseVertexCount;

	CHECKCUDA(cudaFreeAsync(isoValueDevice, stream)); // 释放临时内存
	CHECKCUDA(cudaFreeAsync(d_temp_storage, stream)); // 释放临时内存

	//printf("isoValue = %.9f\n", isoValue);

#ifdef CHECK_MESH_BUILD_TIME_COST
	CHECKCUDA(cudaStreamSynchronize(stream));	// 流同步
	auto end = std::chrono::high_resolution_clock::now();						// 记录结束时间点
	std::chrono::duration<double, std::milli> duration = end - start;			// 计算执行时间（以ms为单位）
	std::cout << "计算等值(isoValue)的时间: " << duration.count() << " ms" << std::endl;		// 输出
	std::cout << std::endl;
	std::cout << "-----------------------------------------------------" << std::endl;	// 输出
	std::cout << std::endl;
#endif // CHECK_MESH_BUILD_TIME_COST
}

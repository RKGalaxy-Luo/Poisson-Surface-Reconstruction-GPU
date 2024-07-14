/*****************************************************************//**
 * \file   LaplacianSolver.cu
 * \brief  ������˹�����cuda����ʵ��
 * 
 * \author LUOJIAXUAN
 * \date   May 26th 2024
 *********************************************************************/
#include "LaplacianSolver.h"
#if defined(__CUDACC__)		//�����NVCC����������
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
		int neighbor = NodeArray[offset].neighs[i];	// �ڵ���ھӽڵ�
		if (neighbor == -1) continue;
		int colIdx = neighbor - begin;				// ������ھӵ�ƫ��
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
	auto start = std::chrono::high_resolution_clock::now();						// ��¼��ʼʱ���
#endif // CHECK_MESH_BUILD_TIME_COST

	dx.ResizeArrayOrException(NodeArray.Size());

	for (int depth = 0; depth <= Constants::maxDepth_Host; depth++) {
		int CurrentLevelNodesNum = NodeArrayCount[depth];	// ��ǰ��ڵ�����
		int CurrentLevelNodesNum_27 = CurrentLevelNodesNum * 27;
		//cudaStream_t stream = streams[depth];
		int* rowCount = NULL;	// ���м���������꼴�ͷţ���ʼֵΪ0����¼��ǰ�ڵ���ھӽڵ��ж��ٸ����㹹��Laplace�����Ԫ�� value �� [0, 26]
		CHECKCUDA(cudaMallocAsync(reinterpret_cast<void**>(&rowCount), sizeof(int) * (CurrentLevelNodesNum + 2), stream));
		CHECKCUDA(cudaMemsetAsync(rowCount, 0, sizeof(int) * (CurrentLevelNodesNum + 2), stream));

		int* colIndex = NULL;	// ���м���������꼴�ͷţ���ʼֵΪ-1����¼��ǰ�ڵ���ھ�������"����Laplace�����Ԫ��"��һ�����Ľڵ㣬����ÿ���׽ڵ�ľ���(neighbor - begin)
		CHECKCUDA(cudaMallocAsync(reinterpret_cast<void**>(&colIndex), sizeof(int) * CurrentLevelNodesNum_27, stream));
		CHECKCUDA(cudaMemsetAsync(colIndex, -1, sizeof(int) * CurrentLevelNodesNum_27, stream));

		float* val = NULL;		// ���м���������꼴�ͷš���¼�ڵ�������ھӵ�LaplaceԪ�ص�ֵ����colIndddex��Ӧ
		CHECKCUDA(cudaMallocAsync(reinterpret_cast<void**>(&val), sizeof(float) * CurrentLevelNodesNum_27, stream));

		dim3 block_1(128);
		dim3 grid_1(divUp(CurrentLevelNodesNum, block_1.x));
		device::GenerateSingleNodeLaplacian << <grid_1, block_1, 0, stream >> > (depth, dot_F_F, dot_F_D2F, encodeNodeIndexInFunction, NodeArray, BaseAddressArray[depth], NodeArrayCount[depth], rowCount + 1, colIndex, val);

		int* RowBaseAddress = NULL;	//���м���������꼴�ͷš���¼��ǰ�ڵ�֮ǰ�Ľڵ�(�����ھ�)������LaplaceԪ��ֵ�ģ��ܹ��ж��ٸ�(�����㵱ǰ�ڵ㣬�����Ǵ�index=1��ʼ������index=0)
		CHECKCUDA(cudaMallocAsync(reinterpret_cast<void**>(&RowBaseAddress), sizeof(int) * (CurrentLevelNodesNum + 2), stream));

		void* tempRowAddressStorage = NULL;	//���㷨��ʱ���������꼴�ͷš�����ǰ׺�͵���ʱ����
		size_t tempRowAddressStorageBytes = 0;
		cub::DeviceScan::ExclusiveSum(tempRowAddressStorage, tempRowAddressStorageBytes, rowCount, RowBaseAddress, CurrentLevelNodesNum + 2, stream);
		CHECKCUDA(cudaMallocAsync(&tempRowAddressStorage, tempRowAddressStorageBytes, stream));
		cub::DeviceScan::ExclusiveSum(tempRowAddressStorage, tempRowAddressStorageBytes, rowCount, RowBaseAddress, CurrentLevelNodesNum + 2, stream);
		int valNums;				// ��¼һ���ж��ٸ���Ч��LaplaceԪ�أ�Ϊ�˸�MergedColIndex��MergedVal�����ڴ�
		int lastRowNum;				// ��¼���һ���ڵ����Ч��LaplaceԪ��
		CHECKCUDA(cudaMemcpyAsync(&valNums, RowBaseAddress + CurrentLevelNodesNum, sizeof(int), cudaMemcpyDeviceToHost, stream));
		CHECKCUDA(cudaMemcpyAsync(&lastRowNum, rowCount + CurrentLevelNodesNum, sizeof(int), cudaMemcpyDeviceToHost, stream));
		CHECKCUDA(cudaStreamSynchronize(stream));	// ����Host�̣߳��������ȴ�valNums������ɣ����ܶ�̬�����ڴ�
		valNums += lastRowNum;		// ǰ������ǰ׺�Ͳ�δ�������һ��Ԫ�أ���index = n + 1��Ӧ����[0, n]���ĺͣ�����rowCount�ĵ�0��һֱ��0����1��ʼ����rowCount��
		CHECKCUDA(cudaMemcpyAsync(RowBaseAddress + CurrentLevelNodesNum + 1, &valNums, sizeof(int), cudaMemcpyHostToDevice, stream));	// ����index = n + 1��Ԫ��
		
		//CHECKCUDA(cudaStreamSynchronize(stream));	// ����Host�̣߳��������ȴ�valNums������ɣ����ܶ�̬�����ڴ�
		//printf("depth = %d   valNums = %d\n", depth, valNums);

		int* MergedColIndex = NULL;	// ���м���������꼴�ͷš�ѹ���Ľڵ㼰�ھ�λ�õ���Чֵ
		CHECKCUDA(cudaMallocAsync(reinterpret_cast<void**>(&MergedColIndex), sizeof(int) * valNums, stream));

		float* MergedVal = NULL;	// ���м���������꼴�ͷš�ѹ���Ľڵ㼰�ھ���Ч��Laplaceֵ
		CHECKCUDA(cudaMallocAsync(reinterpret_cast<void**>(&MergedVal), sizeof(float) * valNums, stream));

		bool* flag = NULL;			// ���м���������꼴�ͷš���ǵ�ǰ�ڵ���ھ�������"����Laplace�����Ԫ��"��һ�����Ľڵ��λ��
		CHECKCUDA(cudaMallocAsync(reinterpret_cast<void**>(&flag), sizeof(bool) * CurrentLevelNodesNum_27, stream));

		dim3 block_2(128);
		dim3 grid_2(divUp(CurrentLevelNodesNum_27, block_2.x));
		device::markValidColIndex << <grid_2, block_2, 0, stream >> > (colIndex, CurrentLevelNodesNum_27, flag);

		int* MergedValSelectedNum = NULL;		// ���м���������꼴�ͷš����ѹ��val��������������Ӧ�ú�valNums���
		CHECKCUDA(cudaMallocAsync(reinterpret_cast<void**>(&MergedValSelectedNum), sizeof(int), stream));
		int* MergedColIndexSelectedNum = NULL;	// ���м���������꼴�ͷš����ѹ��colIndex���������������Ӧ�ú�valNums���
		CHECKCUDA(cudaMallocAsync(reinterpret_cast<void**>(&MergedColIndexSelectedNum), sizeof(int), stream));

		void* d_temp_storage_1 = NULL;
		size_t temp_storage_bytes_1 = 0;
		CHECKCUDA(cub::DeviceSelect::Flagged(d_temp_storage_1, temp_storage_bytes_1, val, flag, MergedVal, MergedValSelectedNum, CurrentLevelNodesNum_27, stream, false));	// ȷ����ʱ�豸�洢����
		CHECKCUDA(cudaMallocAsync(&d_temp_storage_1, temp_storage_bytes_1, stream));	
		CHECKCUDA(cub::DeviceSelect::Flagged(d_temp_storage_1, temp_storage_bytes_1, val, flag, MergedVal, MergedValSelectedNum, CurrentLevelNodesNum_27, stream, false));	// ɸѡ																				

		void* d_temp_storage_2 = NULL;
		size_t temp_storage_bytes_2 = 0;
		CHECKCUDA(cub::DeviceSelect::Flagged(d_temp_storage_2, temp_storage_bytes_2, colIndex, flag, MergedColIndex, MergedColIndexSelectedNum, CurrentLevelNodesNum_27, stream, false));
		CHECKCUDA(cudaMallocAsync(&d_temp_storage_2, temp_storage_bytes_2, stream)); 			 
		CHECKCUDA(cub::DeviceSelect::Flagged(d_temp_storage_2, temp_storage_bytes_2, colIndex, flag, MergedColIndex, MergedColIndexSelectedNum, CurrentLevelNodesNum_27, stream, false));

		int MergedValSelectedNum_Host;			// ���ѹ��val��������������Ӧ�ú�valNums���
		int MergedColIndexSelectedNum_Host;		// ���ѹ��colIndex���������������Ӧ�ú�valNums���
		CHECKCUDA(cudaMemcpyAsync(&MergedValSelectedNum_Host, MergedValSelectedNum, sizeof(int), cudaMemcpyDeviceToHost, stream));
		CHECKCUDA(cudaMemcpyAsync(&MergedColIndexSelectedNum_Host, MergedColIndexSelectedNum, sizeof(int), cudaMemcpyDeviceToHost, stream));
		
		CHECKCUDA(cudaStreamSynchronize(stream));	// ��ͬ��

		assert(MergedValSelectedNum_Host == valNums);		// ˳��Checkһ��ѹ�����Ƿ���ȷ
		assert(MergedColIndexSelectedNum_Host == valNums);	// ˳��Checkһ��ѹ�����Ƿ���ȷ

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
		printf("�� %d ��ڵ��", depth);
#endif // CHECK_MESH_BUILD_TIME_COST
		solverCG_DeviceToDevice(CurrentLevelNodesNum, valNums, RowBaseAddress + 1, MergedColIndex, MergedVal, Divergence + BaseAddressArray[depth], dx.Ptr() + BaseAddressArray[depth], stream);
	
		cudaFreeAsync(tempRowAddressStorage, stream);	 // ��ʱ�ռ�
		cudaFreeAsync(d_temp_storage_1, stream);		 // ��ʱ�ռ�
		cudaFreeAsync(d_temp_storage_2, stream);		 // ��ʱ�ռ�

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



	//CHECKCUDA(cudaStreamSynchronize(stream));	// ��ͬ��
	//std::vector<float>dxHost;
	//dx.ArrayView().Download(dxHost);
	//for (int i = 0; i < dxHost.size(); i++) {
	//	if (i < 1000 || i % 1000 == 0) {
	//		printf("index = %d   dx = %.9f\n", i, dxHost[i]);
	//	}
	//}


#ifdef CHECK_MESH_BUILD_TIME_COST
	CHECKCUDA(cudaStreamSynchronize(stream));	// ��ͬ��
	auto end = std::chrono::high_resolution_clock::now();						// ��¼����ʱ���
	std::chrono::duration<double, std::milli> duration = end - start;			// ����ִ��ʱ�䣨��msΪ��λ��
	std::cout << "������˹����ʱ��: " << duration.count() << " ms" << std::endl;		// ���
	std::cout << std::endl;
	std::cout << "-----------------------------------------------------" << std::endl;	// ���
	std::cout << std::endl;
#endif // CHECK_MESH_BUILD_TIME_COST
}

void SparseSurfelFusion::LaplacianSolver::CalculatePointsImplicitFunctionValue(DeviceArrayView<OrientedPoint3D<float>> DensePoints, DeviceArrayView<int> PointToNodeArrayDLevel, DeviceArrayView<OctNode> NodeArray, DeviceArrayView<int> encodeNodeIndexInFunction, DeviceArrayView<ConfirmedPPolynomial<CONVTIMES + 1, CONVTIMES + 2>> BaseFunctions, const unsigned int DLevelOffset, const unsigned int DenseVertexCount, cudaStream_t stream)
{
#ifdef CHECK_MESH_BUILD_TIME_COST
	auto start = std::chrono::high_resolution_clock::now();						// ��¼��ʼʱ���
#endif // CHECK_MESH_BUILD_TIME_COST

	dim3 block(128);
	dim3 grid(divUp(DenseVertexCount, block.x));
	device::CalculatePointsImplicitFunctionValueKernel << <grid, block, 0, stream >> > (DensePoints, PointToNodeArrayDLevel, NodeArray, encodeNodeIndexInFunction, BaseFunctions, dx.ArrayView(), DLevelOffset, DenseVertexCount, DensePointsImplicitFunctionValue.Array().ptr());

	// ��Լ�ӷ�
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

	CHECKCUDA(cudaFreeAsync(isoValueDevice, stream)); // �ͷ���ʱ�ڴ�
	CHECKCUDA(cudaFreeAsync(d_temp_storage, stream)); // �ͷ���ʱ�ڴ�

	//printf("isoValue = %.9f\n", isoValue);

#ifdef CHECK_MESH_BUILD_TIME_COST
	CHECKCUDA(cudaStreamSynchronize(stream));	// ��ͬ��
	auto end = std::chrono::high_resolution_clock::now();						// ��¼����ʱ���
	std::chrono::duration<double, std::milli> duration = end - start;			// ����ִ��ʱ�䣨��msΪ��λ��
	std::cout << "�����ֵ(isoValue)��ʱ��: " << duration.count() << " ms" << std::endl;		// ���
	std::cout << std::endl;
	std::cout << "-----------------------------------------------------" << std::endl;	// ���
	std::cout << std::endl;
#endif // CHECK_MESH_BUILD_TIME_COST
}

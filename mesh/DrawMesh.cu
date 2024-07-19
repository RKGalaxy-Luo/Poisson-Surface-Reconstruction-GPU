/*****************************************************************//**
 * \file   DrawMesh.cu
 * \brief  OpenGL绘制渲染网格
 * 
 * \author LUOJIAXUAN
 * \date   June 5th 2024
 *********************************************************************/
#include "DrawMesh.h"

__host__ __device__ __forceinline__ void SparseSurfelFusion::device::KnnHeapDevice::update(unsigned int idx, float dist)
{
	if (dist < distance.x) {
		distance.x = dist;
		index.x = idx;

		if (distance.y < distance.z) {
			if (distance.x < distance.z) {
				swap(distance.x, distance.z);
				swap(index.x, index.z);
			}
		}
		else {
			if (distance.x < distance.y) {
				swap(distance.x, distance.y);
				swap(index.x, index.y);
				if (distance.y < distance.w) {
					swap(distance.y, distance.w);
					swap(index.y, index.w);
				}
			}
		}
	}
}

__device__ __forceinline__ void SparseSurfelFusion::device::bruteForceSearch4KNN(const float3& vertex, DeviceArrayView<OrientedPoint3D<float>> samplePoint, const unsigned int samplePointsCount, float4& distance, uint4& sampleIndex)
{
	KnnHeapDevice heap(distance, sampleIndex);	// 构建堆
	const unsigned int padded_node_num = ((samplePointsCount + 3) / 4) * 4;
	for (int k = 0; k < padded_node_num; k += 4) {
		// 计算到每一个采样点的距离
		const float tmp0 = vertex.x - samplePoint[k + 0].point.coords[0];
		const float tmp1 = vertex.y - samplePoint[k + 0].point.coords[1];
		const float tmp2 = vertex.z - samplePoint[k + 0].point.coords[2];

		const float tmp6 = vertex.x - samplePoint[k + 1].point.coords[0];
		const float tmp7 = vertex.y - samplePoint[k + 1].point.coords[1];
		const float tmp8 = vertex.z - samplePoint[k + 1].point.coords[2];

		const float tmp12 = vertex.x - samplePoint[k + 2].point.coords[0];
		const float tmp13 = vertex.y - samplePoint[k + 2].point.coords[1];
		const float tmp14 = vertex.z - samplePoint[k + 2].point.coords[2];

		const float tmp18 = vertex.x - samplePoint[k + 3].point.coords[0];
		const float tmp19 = vertex.y - samplePoint[k + 3].point.coords[1];
		const float tmp20 = vertex.z - samplePoint[k + 3].point.coords[2];

		const float tmp3 = __fmul_rn(tmp0, tmp0);
		const float tmp9 = __fmul_rn(tmp6, tmp6);
		const float tmp15 = __fmul_rn(tmp12, tmp12);
		const float tmp21 = __fmul_rn(tmp18, tmp18);

		const float tmp4 = __fmaf_rn(tmp1, tmp1, tmp3);
		const float tmp10 = __fmaf_rn(tmp7, tmp7, tmp9);
		const float tmp16 = __fmaf_rn(tmp13, tmp13, tmp15);
		const float tmp22 = __fmaf_rn(tmp19, tmp19, tmp21);

		const float dist_0 = __fmaf_rn(tmp2, tmp2, tmp4);
		const float dist_1 = __fmaf_rn(tmp8, tmp8, tmp10);
		const float dist_2 = __fmaf_rn(tmp14, tmp14, tmp16);
		const float dist_3 = __fmaf_rn(tmp20, tmp20, tmp22);

		// 更新索引
		heap.update(k + 0, dist_0);
		heap.update(k + 1, dist_1);
		heap.update(k + 2, dist_2);
		heap.update(k + 3, dist_3);
	}
}

__device__ float3 SparseSurfelFusion::device::VectorNormalize(const float3& normal)
{
	float3 result;
	float length = sqrtf(normal.x * normal.x + normal.y * normal.y + normal.z * normal.z);
	if (length == 0) { result.x = 0; result.y = 0; result.z = 0; }
	else {
		result.x = normal.x / length;
		result.y = normal.y / length;
		result.z = normal.z / length;
	}
	return result;
}

__device__ float3 SparseSurfelFusion::device::CrossProduct(const float3& Vector_OA, const float3& Vector_OB)
{
	float3 result;
	result.x = Vector_OA.y * Vector_OB.z - Vector_OA.z * Vector_OB.y;
	result.y = Vector_OA.z * Vector_OB.x - Vector_OA.x * Vector_OB.z;
	result.z = Vector_OA.x * Vector_OB.y - Vector_OA.y * Vector_OB.x;
	return result;
}

__global__ void SparseSurfelFusion::device::CalculateMeshNormalsKernel(const Point3D<float>* verticesArray, const TriangleIndex* indicesArray, const unsigned int meshCount, Point3D<float>* normalsArray)
{
	const unsigned int idx = threadIdx.x + blockDim.x * blockIdx.x;
	if (idx >= meshCount) return;
	Point3D<float> Point_O = verticesArray[indicesArray[idx].idx[0]];
	Point3D<float> Point_A = verticesArray[indicesArray[idx].idx[1]];
	Point3D<float> Point_B = verticesArray[indicesArray[idx].idx[2]];
	float3 Vector_OA, Vector_OB;

	Vector_OA.x = Point_A.coords[0] - Point_O.coords[0];
	Vector_OA.y = Point_A.coords[1] - Point_O.coords[1];
	Vector_OA.z = Point_A.coords[2] - Point_O.coords[2];

	Vector_OB.x = Point_B.coords[0] - Point_O.coords[0];
	Vector_OB.y = Point_B.coords[1] - Point_O.coords[1];
	Vector_OB.z = Point_B.coords[2] - Point_O.coords[2];

	float3 Normal = CrossProduct(Vector_OA, Vector_OB);
	float3 normalizedNormal = VectorNormalize(Normal);
	normalsArray[idx].coords[0] = normalizedNormal.x;
	normalsArray[idx].coords[1] = normalizedNormal.y;
	normalsArray[idx].coords[2] = normalizedNormal.z;
}

__global__ void SparseSurfelFusion::device::CountConnectedTriangleNumKernel(const TriangleIndex* indicesArray, const unsigned int meshCount, unsigned int* ConnectedTriangleNum)
{
	const unsigned int idx = threadIdx.x + blockDim.x * blockIdx.x;
	if (idx >= meshCount) return;
	atomicAdd(&ConnectedTriangleNum[indicesArray[idx].idx[0]], 1);
	atomicAdd(&ConnectedTriangleNum[indicesArray[idx].idx[1]], 1);
	atomicAdd(&ConnectedTriangleNum[indicesArray[idx].idx[2]], 1);
}

__global__ void SparseSurfelFusion::device::VerticesNormalsSumKernel(const Point3D<float>* meshNormals, const TriangleIndex* indicesArray, const unsigned int meshCount, Point3D<float>* VerticesNormalsSum)
{
	const unsigned int idx = threadIdx.x + blockDim.x * blockIdx.x;
	if (idx >= meshCount)	return;
	// 三角mesh的每一个顶点都应该加上其法线
	// 三角面元的第一个顶点
	atomicAdd(&VerticesNormalsSum[indicesArray[idx].idx[0]].coords[0], meshNormals[idx].coords[0]);
	atomicAdd(&VerticesNormalsSum[indicesArray[idx].idx[0]].coords[1], meshNormals[idx].coords[1]);
	atomicAdd(&VerticesNormalsSum[indicesArray[idx].idx[0]].coords[2], meshNormals[idx].coords[2]);
	// 三角面元的第二个顶点
	atomicAdd(&VerticesNormalsSum[indicesArray[idx].idx[1]].coords[0], meshNormals[idx].coords[0]);
	atomicAdd(&VerticesNormalsSum[indicesArray[idx].idx[1]].coords[1], meshNormals[idx].coords[1]);
	atomicAdd(&VerticesNormalsSum[indicesArray[idx].idx[1]].coords[2], meshNormals[idx].coords[2]);
	// 三角面元的第三个顶点
	atomicAdd(&VerticesNormalsSum[indicesArray[idx].idx[2]].coords[0], meshNormals[idx].coords[0]);
	atomicAdd(&VerticesNormalsSum[indicesArray[idx].idx[2]].coords[1], meshNormals[idx].coords[1]);
	atomicAdd(&VerticesNormalsSum[indicesArray[idx].idx[2]].coords[2], meshNormals[idx].coords[2]);
	//if (idx % 100 == 0) printf("ScaledVerticesNormalsSum[%d] = (%.10f, %.10f, %.10f)\n", idx, meshNormals[idx].coords[0], meshNormals[idx].coords[1], meshNormals[idx].coords[2]);
	//if (idx % 100 == 0) printf("ScaledVerticesNormalsSum[%d] = (%.10f, %.10f, %.10f)\n", idx, ScaledVerticesNormalsSum[idx].coords[0], ScaledVerticesNormalsSum[idx].coords[1], ScaledVerticesNormalsSum[idx].coords[2]);
}

__global__ void SparseSurfelFusion::device::CalculateVerticesAverageNormals(const unsigned int* ConnectedTriangleNum, const Point3D<float>* VerticesNormalsSum, const unsigned int verticesCount, Point3D<float>* VerticesAverageNormals)
{
	const unsigned int idx = threadIdx.x + blockDim.x * blockIdx.x;
	if (idx >= verticesCount)	return;
	int meshCount = ConnectedTriangleNum[idx];
	float AverageNormalX = VerticesNormalsSum[idx].coords[0] / meshCount;
	float AverageNormalY = VerticesNormalsSum[idx].coords[1] / meshCount;
	float AverageNormalZ = VerticesNormalsSum[idx].coords[2] / meshCount;
	//if (meshCount == 0) printf("Error = %d\n", idx);
	//if (idx % 1000 == 0) printf("meshCount[%d] = %d\n", idx, meshCount);
	//if (idx % 100 == 0) printf("ScaledVerticesNormalsSum[%d] = (%.10f, %.10f, %.10f)\n", idx, ScaledVerticesNormalsSum[idx].coords[0], ScaledVerticesNormalsSum[idx].coords[1], ScaledVerticesNormalsSum[idx].coords[2]);
	//if (idx % 1000 == 0) printf("AverageNormals[%d] = (%.10f, %.10f, %.10f)\n", idx, AverageNormalX, AverageNormalY, AverageNormalZ);
	
	float3 averageNormal;
	averageNormal.x = AverageNormalX;
	averageNormal.y = AverageNormalY;
	averageNormal.z = AverageNormalZ;

	float3 NormalizedAverageNormal = VectorNormalize(averageNormal);

	VerticesAverageNormals[idx].coords[0] = NormalizedAverageNormal.x;
	VerticesAverageNormals[idx].coords[1] = NormalizedAverageNormal.y;
	VerticesAverageNormals[idx].coords[2] = NormalizedAverageNormal.z;
	//if (idx % 1000 == 0) printf("NormalizedAverageNormals[%d] = (%.10f, %.10f, %.10f)\n", idx, NormalizedAverageNormal.x, NormalizedAverageNormal.y, NormalizedAverageNormal.z);

}

__global__ void SparseSurfelFusion::device::CalculateVerticesAverageColors(DeviceArrayView<Point3D<float>> meshVertices, DeviceArrayView<OrientedPoint3D<float>> samplePoints, const unsigned int verticesCount, const unsigned int samplePointsCount, Point3D<float>* VerticesAverageColors)
{
	const unsigned int idx = threadIdx.x + blockDim.x * blockIdx.x;
	if (idx >= verticesCount)	return;
	// 使用堆保持优先级队列
	float4 knnDistance = make_float4(1e6f, 1e6f, 1e6f, 1e6f);
	uint4 knnIndex = make_uint4(0, 0, 0, 0);
	float3 vertex = make_float3(meshVertices[idx].coords[0], meshVertices[idx].coords[1], meshVertices[idx].coords[2]);

	// knnIndex中是与vertex最近的四个稠密点邻居
	bruteForceSearch4KNN(vertex, samplePoints, samplePointsCount, knnDistance, knnIndex);

	// 获得最近的4个采样点
	float3 nearestSample_1 = make_float3(samplePoints[knnIndex.x].point.coords[0], samplePoints[knnIndex.x].point.coords[1], samplePoints[knnIndex.x].point.coords[2]);
	float3 nearestSample_2 = make_float3(samplePoints[knnIndex.y].point.coords[0], samplePoints[knnIndex.y].point.coords[1], samplePoints[knnIndex.y].point.coords[2]);
	float3 nearestSample_3 = make_float3(samplePoints[knnIndex.z].point.coords[0], samplePoints[knnIndex.z].point.coords[1], samplePoints[knnIndex.z].point.coords[2]);
	float3 nearestSample_4 = make_float3(samplePoints[knnIndex.w].point.coords[0], samplePoints[knnIndex.w].point.coords[1], samplePoints[knnIndex.w].point.coords[2]);

	// 计算四个点的距离
	float4 nearestDistance;
	nearestDistance.x = squared_norm(vertex - nearestSample_1);
	nearestDistance.y = squared_norm(vertex - nearestSample_2);
	nearestDistance.z = squared_norm(vertex - nearestSample_3);
	nearestDistance.w = squared_norm(vertex - nearestSample_4);

	const float distanceSumInverse = 1.0f / (nearestDistance.x + nearestDistance.y + nearestDistance.z + nearestDistance.w);
	float4 rgbWeight;
	rgbWeight.x = nearestDistance.x * distanceSumInverse;
	rgbWeight.y = nearestDistance.y * distanceSumInverse;
	rgbWeight.z = nearestDistance.z * distanceSumInverse;
	rgbWeight.w = nearestDistance.w * distanceSumInverse;

	VerticesAverageColors[idx].coords[0] = samplePoints[knnIndex.x].color[0] * rgbWeight.x + samplePoints[knnIndex.y].color[0] * rgbWeight.y + samplePoints[knnIndex.z].color[0] * rgbWeight.z + samplePoints[knnIndex.w].color[0] * rgbWeight.w;
	VerticesAverageColors[idx].coords[1] = samplePoints[knnIndex.x].color[1] * rgbWeight.x + samplePoints[knnIndex.y].color[1] * rgbWeight.y + samplePoints[knnIndex.z].color[1] * rgbWeight.z + samplePoints[knnIndex.w].color[1] * rgbWeight.w;
	VerticesAverageColors[idx].coords[2] = samplePoints[knnIndex.x].color[2] * rgbWeight.x + samplePoints[knnIndex.y].color[2] * rgbWeight.y + samplePoints[knnIndex.z].color[2] * rgbWeight.z + samplePoints[knnIndex.w].color[2] * rgbWeight.w;
}

void SparseSurfelFusion::DrawMesh::CalculateMeshNormals(DeviceArrayView<Point3D<float>> meshVertices, DeviceArrayView<TriangleIndex> meshTriangleIndices, cudaStream_t stream)
{

#ifdef CHECK_MESH_BUILD_TIME_COST
	auto time1 = std::chrono::high_resolution_clock::now();					// 记录开始时间点
#endif // CHECK_MESH_BUILD_TIME_COST

	CHECKCUDA(cudaMemcpyAsync(MeshVertices.Ptr(), meshVertices.RawPtr(), sizeof(Point3D<float>) * VerticesCount, cudaMemcpyDeviceToDevice, stream));
	CHECKCUDA(cudaMemcpyAsync(MeshTriangleIndices.Ptr(), meshTriangleIndices.RawPtr(), sizeof(TriangleIndex) * TranglesCount, cudaMemcpyDeviceToDevice, stream));

	Point3D<float>* MeshNormalsDevice = NULL;	// 记录计算得到的三角网格的法线
	CHECKCUDA(cudaMallocAsync(reinterpret_cast<void**>(&MeshNormalsDevice), sizeof(Point3D<float>) * TranglesCount, stream));

	dim3 block_Mesh(256);
	dim3 grid_Mesh(divUp(TranglesCount, block_Mesh.x));
	device::CalculateMeshNormalsKernel << <grid_Mesh, block_Mesh, 0, stream >> > (MeshVertices.ArrayView(), MeshTriangleIndices.ArrayView(), TranglesCount, MeshNormalsDevice);

	unsigned int* ConnectedTriangleNum = NULL;		// 记录一个顶点有多少邻接的三角形
	CHECKCUDA(cudaMallocAsync(reinterpret_cast<void**>(&ConnectedTriangleNum), sizeof(unsigned int) * VerticesCount, stream));
	CHECKCUDA(cudaMemsetAsync(ConnectedTriangleNum, 0, sizeof(unsigned int) * VerticesCount, stream));
	device::CountConnectedTriangleNumKernel << <grid_Mesh, block_Mesh, 0, stream >> > (MeshTriangleIndices.ArrayView(), TranglesCount, ConnectedTriangleNum);

	Point3D<float>* VerticesNormalsSum = NULL;		// 记录其邻接的三角Mesh的法线向量和
	CHECKCUDA(cudaMallocAsync(reinterpret_cast<void**>(&VerticesNormalsSum), sizeof(Point3D<float>) * VerticesCount, stream));
	CHECKCUDA(cudaMemsetAsync(VerticesNormalsSum, 0.0f, sizeof(Point3D<float>) * VerticesCount, stream));
	device::VerticesNormalsSumKernel << <grid_Mesh, block_Mesh, 0, stream >> > (MeshNormalsDevice, MeshTriangleIndices.ArrayView(), TranglesCount, VerticesNormalsSum);

	VerticesAverageNormals.ResizeArrayOrException(VerticesCount);

	dim3 block_vertex(256);
	dim3 grid_vertex(divUp(VerticesCount, block_vertex.x));
	device::CalculateVerticesAverageNormals << <grid_vertex, block_vertex, 0, stream >> > (ConnectedTriangleNum, VerticesNormalsSum, VerticesCount, VerticesAverageNormals.Ptr());

	CHECKCUDA(cudaFreeAsync(MeshNormalsDevice, stream));
	CHECKCUDA(cudaFreeAsync(ConnectedTriangleNum, stream));
	CHECKCUDA(cudaFreeAsync(VerticesNormalsSum, stream));

#ifdef CHECK_MESH_BUILD_TIME_COST
	CHECKCUDA(cudaStreamSynchronize(stream));
	auto time2 = std::chrono::high_resolution_clock::now();					// 记录结束时间点
	std::chrono::duration<double, std::milli> duration1 = time2 - time1;		// 计算执行时间（以ms为单位）
	std::cout << "计算Mesh法向量时间: " << duration1.count() << " ms" << std::endl;
#endif // CHECK_MESH_BUILD_TIME_COST
}

void SparseSurfelFusion::DrawMesh::CalculateMeshVerticesColor(DeviceArrayView<OrientedPoint3D<float>> sampleDensePoints, DeviceArrayView<Point3D<float>> meshVertices, cudaStream_t stream)
{
#ifdef CHECK_MESH_BUILD_TIME_COST
	auto time1 = std::chrono::high_resolution_clock::now();					// 记录开始时间点
#endif // CHECK_MESH_BUILD_TIME_COST

	dim3 block(256);
	dim3 grid(divUp(VerticesCount, block.x));
	device::CalculateVerticesAverageColors << <grid, block, 0, stream >> > (meshVertices, sampleDensePoints, VerticesCount, DensePointsCount, VerticesAverageColors.Ptr());

#ifdef CHECK_MESH_BUILD_TIME_COST
	CHECKCUDA(cudaStreamSynchronize(stream));
	auto time2 = std::chrono::high_resolution_clock::now();					// 记录结束时间点
	std::chrono::duration<double, std::milli> duration1 = time2 - time1;		// 计算执行时间（以ms为单位）
	std::cout << "计算Mesh顶点颜色的时间: " << duration1.count() << " ms" << std::endl;
#endif // CHECK_MESH_BUILD_TIME_COST
}
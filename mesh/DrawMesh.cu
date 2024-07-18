/*****************************************************************//**
 * \file   DrawMesh.cu
 * \brief  OpenGL绘制渲染网格
 * 
 * \author LUOJIAXUAN
 * \date   June 5th 2024
 *********************************************************************/
#include "DrawMesh.h"


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

void SparseSurfelFusion::DrawMesh::CalculateMeshNormals(DeviceArrayView<Point3D<float>> meshVertices, DeviceArrayView<TriangleIndex> meshTriangleIndices, cudaStream_t stream)
{

#ifdef CHECK_MESH_BUILD_TIME_COST
	auto time1 = std::chrono::high_resolution_clock::now();					// 记录开始时间点
#endif // CHECK_MESH_BUILD_TIME_COST

	TranglesCount = meshTriangleIndices.Size();
	VerticesCount = meshVertices.Size();
	MeshVertices.ResizeArrayOrException(VerticesCount);
	MeshTriangleIndices.ResizeArrayOrException(TranglesCount);

	CHECKCUDA(cudaMemcpyAsync(MeshVertices.Ptr(), meshVertices.RawPtr(), sizeof(Point3D<float>) * VerticesCount, cudaMemcpyDeviceToDevice, stream));
	CHECKCUDA(cudaMemcpyAsync(MeshTriangleIndices.Ptr(), meshTriangleIndices.RawPtr(), sizeof(TriangleIndex) * TranglesCount, cudaMemcpyDeviceToDevice, stream));


	//VerticesNormals.resize(VerticesCount);
	Point3D<float>* MeshNormalsDevice = NULL;	// 记录计算得到的三角网格的法线
	CHECKCUDA(cudaMallocAsync(reinterpret_cast<void**>(&MeshNormalsDevice), sizeof(Point3D<float>) * TranglesCount, stream));

	//std::vector<TriangleIndex> TriangleIndexHost;
	//mesh.GetTriangleIndices(TriangleIndexHost);
	//std::cout << "TriangleIndexHostCount = " << TriangleIndexHost.size() << std::endl;

	//TriangleIndex* TriangleIndexDevice = NULL;	// 三角网格索引
	//CHECKCUDA(cudaMallocAsync(reinterpret_cast<void**>(&TriangleIndexDevice), sizeof(TriangleIndex) * MeshCount, stream));
	//CHECKCUDA(cudaMemcpyAsync(TriangleIndexDevice, TriangleIndexHost.data(), sizeof(Point3D<float>) * MeshCount, cudaMemcpyHostToDevice, stream));

	//std::vector<Point3D<float>> VerticesArrayHost;
	//mesh.GetVertexArray(VerticesArrayHost);
	//std::cout << "VerticesArrayHostCount = " << VerticesArrayHost.size() << std::endl;

	//Point3D<float>* VerticesArrayDevice = NULL;		// 顶点数组
	//CHECKCUDA(cudaMallocAsync(reinterpret_cast<void**>(&VerticesArrayDevice), sizeof(Point3D<float>) * VerticesCount, stream));
	//CHECKCUDA(cudaMemcpyAsync(VerticesArrayDevice, VerticesArrayHost.data(), sizeof(Point3D<float>) * VerticesCount, cudaMemcpyHostToDevice, stream));
	//  << <grid_Mesh, block_Mesh, 0, stream >> > 
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

	//// << <grid_vertex, block_vertex, 0, stream >> > 
	//Point3D<float>* VerticesAverageNormals = NULL;	// 归一化的顶点平均法向量
	//CHECKCUDA(cudaMallocAsync(reinterpret_cast<void**>(&VerticesAverageNormals), sizeof(Point3D<float>) * VerticesCount, stream));
	dim3 block_vertex(256);
	dim3 grid_vertex(divUp(VerticesCount, block_vertex.x));
	device::CalculateVerticesAverageNormals << <grid_vertex, block_vertex, 0, stream >> > (ConnectedTriangleNum, VerticesNormalsSum, VerticesCount, VerticesAverageNormals.Ptr());

	//CHECKCUDA(cudaMemcpyAsync(VerticesNormals.data(), VerticesAverageNormals, sizeof(Point3D<float>) * VerticesCount, cudaMemcpyDeviceToHost, stream));

	CHECKCUDA(cudaFreeAsync(MeshNormalsDevice, stream));
	//CHECKCUDA(cudaFreeAsync(TriangleIndexDevice, stream));
	//CHECKCUDA(cudaFreeAsync(VerticesArrayDevice, stream));
	CHECKCUDA(cudaFreeAsync(ConnectedTriangleNum, stream));
	CHECKCUDA(cudaFreeAsync(VerticesNormalsSum, stream));

	CHECKCUDA(cudaStreamSynchronize(stream));

#ifdef CHECK_MESH_BUILD_TIME_COST
	CHECKCUDA(cudaStreamSynchronize(stream));
	auto time2 = std::chrono::high_resolution_clock::now();					// 记录结束时间点
	std::chrono::duration<double, std::milli> duration1 = time2 - time1;		// 计算执行时间（以ms为单位）
	std::cout << "计算Mesh法向量时间: " << duration1.count() << " ms" << std::endl;
#endif // CHECK_MESH_BUILD_TIME_COST
}

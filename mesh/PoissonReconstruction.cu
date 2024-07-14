/*****************************************************************//**
 * \file   PoissonReconstruction.cu
 * \brief  GPU求解泊松曲面重建cuda算法实现
 * 
 * \author LUOJIAXUAN
 * \date   May 4th 2024
 *********************************************************************/
#include "PoissonReconstruction.h"

__global__ void SparseSurfelFusion::device::buildDenseSurfelKernel(pcl::PointXYZ* coor, DepthSurfel* surfel, const unsigned int pointsNum)
{
	unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= pointsNum) return;
	surfel[idx].VertexAndConfidence.x = coor[idx].x;
	surfel[idx].VertexAndConfidence.y = coor[idx].y;
	surfel[idx].VertexAndConfidence.z = coor[idx].z;
	surfel[idx].VertexAndConfidence.w = 0;
}

__global__ void SparseSurfelFusion::device::buildOrientedDenseSurfelKernel(pcl::PointXYZ* coor, pcl::Normal* normal, DepthSurfel* surfel, const unsigned int pointsNum)
{
	unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= pointsNum) return;
	surfel[idx].VertexAndConfidence.x = coor[idx].x;
	surfel[idx].VertexAndConfidence.y = coor[idx].y;
	surfel[idx].VertexAndConfidence.z = coor[idx].z;
	surfel[idx].VertexAndConfidence.w = 0;

	surfel[idx].NormalAndRadius.x = normal[idx].normal_x;
	surfel[idx].NormalAndRadius.y = normal[idx].normal_y;
	surfel[idx].NormalAndRadius.z = normal[idx].normal_z;
	surfel[idx].NormalAndRadius.w = normal[idx].curvature;
}

void SparseSurfelFusion::PoissonReconstruction::buildDenseSurfel(DeviceBufferArray<pcl::PointXYZ>& PointCloudDevice, DeviceBufferArray<DepthSurfel>& DenseSurfel, cudaStream_t stream)
{
	const unsigned int PointsNum = PointCloudDevice.ArrayView().Size();
	dim3 block(128);
	dim3 grid(divUp(PointsNum, block.x));
	device::buildDenseSurfelKernel << <grid, block, 0, stream >> > (PointCloudDevice.Array().ptr(), DenseSurfel.Array().ptr(), PointsNum);
}

void SparseSurfelFusion::PoissonReconstruction::buildDenseSurfel(DeviceBufferArray<pcl::PointXYZ>& PointCloudDevice, DeviceBufferArray<pcl::Normal>& PointNormalDevice, DeviceBufferArray<DepthSurfel>& DenseSurfel, cudaStream_t stream)
{
	const unsigned int PointsNum = PointCloudDevice.ArrayView().Size();
	dim3 block(128);
	dim3 grid(divUp(PointsNum, block.x));
	device::buildOrientedDenseSurfelKernel << <grid, block, 0, stream >> > (PointCloudDevice.Array().ptr(), PointNormalDevice.Array().ptr(), DenseSurfel.Array().ptr(), PointsNum);
}
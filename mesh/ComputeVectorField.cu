/*****************************************************************//**
 * \file   ComputeVectorField.cu
 * \brief  计算向量场cuda方法实现
 * 
 * \author LUOJIAXUAN
 * \date   May 15th 2024
 *********************************************************************/
#include "ComputeVectorField.h"

namespace SparseSurfelFusion {
	namespace device {
		__device__ __constant__ int maxDepth = MAX_DEPTH_OCTREE;

		__device__ __constant__ int normalize = NORMALIZE;
	}
}

__device__ float SparseSurfelFusion::device::FCenterWidthPoint(int idx, int i, int j, const ConfirmedPPolynomial<CONVTIMES, CONVTIMES + 2>& BaseFunctionMaxDepth_d, const Point3D<float>& center, const float& width, const Point3D<float>& point)
{
	ConfirmedPPolynomial<CONVTIMES, CONVTIMES + 2> thisFunction_X = BaseFunctionMaxDepth_d.shift(center.coords[0]);
	ConfirmedPPolynomial<CONVTIMES, CONVTIMES + 2> thisFunction_Y = BaseFunctionMaxDepth_d.shift(center.coords[1]);
	ConfirmedPPolynomial<CONVTIMES, CONVTIMES + 2> thisFunction_Z = BaseFunctionMaxDepth_d.shift(center.coords[2]);
	float x = value(thisFunction_X, point.coords[0]);
	float y = value(thisFunction_Y, point.coords[1]);
	float z = value(thisFunction_Z, point.coords[2]);
	float ret = x * y * z;
	switch (device::normalize) {
	case 2:
		ret /= sqrt(1.0 / (1 << (device::maxDepth)));
		break;
	case 1:
		ret /= 1.0 / (1 << (device::maxDepth));
		break;
	}
	//if (/*1000 <= idx && idx < 1100*/idx == 1050) {
	//	printf("index = %d  (%d, %d)  point = (%.10f, %.10f, %.10f)  o_c = (%.5f, %.5f, %.5f)  ret = %.10f\n", idx, i, j, point.coords[0], point.coords[1], point.coords[2], center.coords[0], center.coords[1], center.coords[2], ret);
	//}
	return ret;
}

__device__ void SparseSurfelFusion::device::getFunctionIdxNode(const int& key, const int& maxDepth, int* index)
{
	// (假设device::maxDepth = 8)
	index[0] = (1 << device::maxDepth) - 1;	// 初值:00011111111 
	index[1] = index[0];
	index[2] = index[1];

	// (1 << (device::maxDepth - depth)) = 00011111111

	for (int depth = device::maxDepth; depth >= 1; depth--) {
		// 获得编码的x,y,z的分量
		int sonKeyX = (key >> (3 * (device::maxDepth - depth) + 2)) & 1;  // 获得孩子节点Key的X分量
		int sonKeyY = (key >> (3 * (device::maxDepth - depth) + 1)) & 1;  // 获得孩子节点Key的Y分量
		int sonKeyZ = (key >> (3 * (device::maxDepth - depth))) & 1;	  // 获得孩子节点Key的Z分量
		index[0] += sonKeyX * (1 << (device::maxDepth - depth));
		index[1] += sonKeyY * (1 << (device::maxDepth - depth));
		index[2] += sonKeyZ * (1 << (device::maxDepth - depth));
	}
}

__global__ void SparseSurfelFusion::device::CalculateVectorFieldKernel(ConfirmedPPolynomial<CONVTIMES, CONVTIMES + 2>* BaseFunctionMaxDepth_Device, DeviceArrayView<OrientedPoint3D<float>> DenseOrientedPoints, DeviceArrayView<OctNode> NodeArray, const unsigned int DLevelOffset, const unsigned int DLevelNodeNum, Point3D<float>* VectorField)
{
	const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= DLevelNodeNum) return;
	const unsigned int offset = DLevelOffset + idx;
	int index[3];
	float width;
	getFunctionIdxNode(NodeArray[offset].key, device::maxDepth, index);
	Point3D<float> o_c;
	BinaryNode<float>::CenterAndWidth(index[0], o_c.coords[0], width);
	BinaryNode<float>::CenterAndWidth(index[1], o_c.coords[1], width);
	BinaryNode<float>::CenterAndWidth(index[2], o_c.coords[2], width);

	//if (5000 <= idx && idx < 5100) {
	//	printf("index = %d   width = %.7f   o_c(%.7f, %.7f, %.7f)\n", idx, width, o_c.coords[0], o_c.coords[1], o_c.coords[2]);
	//}

	/** 检查index以及o_c,无误 **/

	Point3D<float> val;
	for (int i = 0; i < 27; i++) {
		int neighbor = NodeArray[offset].neighs[i];
		if (neighbor != -1) {
			for (int j = 0; j < NodeArray[neighbor].pnum; j++) {
				int pointIdx = NodeArray[neighbor].pidx + j;	// 在稠密点数组中的位置
				float weight = FCenterWidthPoint(idx, i, j, *BaseFunctionMaxDepth_Device, o_c, width, DenseOrientedPoints[pointIdx].point);
				val.coords[0] += weight * DenseOrientedPoints[pointIdx].normal.coords[0];
				val.coords[1] += weight * DenseOrientedPoints[pointIdx].normal.coords[1];
				val.coords[2] += weight * DenseOrientedPoints[pointIdx].normal.coords[2];
				//if (idx == 995) {
				//	printf("(%d, %d)  pnum = %d  pidx = %d  pointIdx[%d] = (%.10f, %.10f, %.10f, %.10f, %.10f, %.10f)\n", i, j, NodeArray[neighbor].pnum, NodeArray[neighbor].pidx, pointIdx, DenseOrientedPoints[pointIdx].point.coords[0], DenseOrientedPoints[pointIdx].point.coords[1], DenseOrientedPoints[pointIdx].point.coords[2], DenseOrientedPoints[pointIdx].normal.coords[0], DenseOrientedPoints[pointIdx].normal.coords[1], DenseOrientedPoints[pointIdx].normal.coords[2]);
				//}
			}
		}
	}

	VectorField[idx].coords[0] += val.coords[0];
	VectorField[idx].coords[1] += val.coords[1];
	VectorField[idx].coords[2] += val.coords[2];
}

void SparseSurfelFusion::ComputeVectorField::CalculateVectorField(ConfirmedPPolynomial<CONVTIMES, CONVTIMES + 2>* BaseFunctionMaxDepth_Device, DeviceArrayView<OrientedPoint3D<float>> DenseOrientedPoints, DeviceArrayView<OctNode> NodeArray, const unsigned int DLevelOffset, const unsigned int DLevelNodeNum, DeviceBufferArray<Point3D<float>>& VectorField, cudaStream_t stream)
{
	dim3 block(128);
	dim3 grid(divUp(DLevelNodeNum, block.x));
	device::CalculateVectorFieldKernel << <grid, block, 0, stream >> > (BaseFunctionMaxDepth_Device, DenseOrientedPoints, NodeArray, DLevelOffset, DLevelNodeNum, VectorField.Array().ptr());
}
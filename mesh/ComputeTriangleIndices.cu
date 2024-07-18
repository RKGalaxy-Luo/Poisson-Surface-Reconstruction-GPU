/*****************************************************************//**
 * \file   ComputeTriangleIndices.cu
 * \brief  插入修复三角网格，构建网格
 * 
 * \author LUOJIAXUAN
 * \date   June 3rd 2024
 *********************************************************************/
#include "ComputeTriangleIndices.h"
#if defined(__CUDACC__)		//如果由NVCC编译器编译
#include <cub/cub.cuh>
#endif
#include <thrust/device_ptr.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/copy.h>
namespace SparseSurfelFusion {

    struct ifSubdivide {
        __device__ bool operator()(const OctNode& x) {
            //return (x.children[0] == -1) && (x.hasTriangle || x.hasIntersection);
            return (x.children[0] == -1 && x.children[1] == -1 && x.children[2] == -1 && x.children[3] == -1 && x.children[4] == -1 && x.children[5] == -1 && x.children[6] == -1 && x.children[7] == -1) && (x.hasTriangle || x.hasIntersection);

        }
    };

	namespace device {
        __device__ __constant__ int maxDepth = MAX_DEPTH_OCTREE;	// Octree最大深度

		__device__ __constant__ int decodeOffset_1 = (1 << (MAX_DEPTH_OCTREE + 1));

		__device__ __constant__ int decodeOffset_2 = (1 << (2 * (MAX_DEPTH_OCTREE + 1)));

        __device__ __constant__ int childrenVertexKind[8] = { 0, 1, 3, 2, 4, 5, 7, 6 };

        __device__ __constant__ int maxIntValue = 0x7fffffff;		// 最大int值

        __device__ __constant__ float eps = EPSILON;

        __device__ __constant__ int edgeVertex[12][2] = { {0,1}, {2,3}, {4,5}, {6,7}, {0,3}, {1,2},
                                                          {4,7}, {5,6}, {0,4}, {1,5}, {3,7}, {2,6} };
        
        // 立方体8个顶点，2^8 = 256
        __constant__ int trianglesCount[256] = { 0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 2,
                                                 1, 2, 2, 3, 2, 3, 3, 4, 2, 3, 3, 4, 3, 4, 4, 3,
                                                 1, 2, 2, 3, 2, 3, 3, 4, 2, 3, 3, 4, 3, 4, 4, 3, 
                                                 2, 3, 3, 2, 3, 4, 4, 3, 3, 4, 4, 3, 4, 5, 5, 2,
                                                 1, 2, 2, 3, 2, 3, 3, 4, 2, 3, 3, 4, 3, 4, 4, 3,
                                                 2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 4,
                                                 2, 3, 3, 4, 3, 4, 2, 3, 3, 4, 4, 5, 4, 5, 3, 2,
                                                 3, 4, 4, 3, 4, 5, 3, 2, 4, 5, 5, 4, 5, 2, 4, 1,
                                                 1, 2, 2, 3, 2, 3, 3, 4, 2, 3, 3, 4, 3, 4, 4, 3,
                                                 2, 3, 3, 4, 3, 4, 4, 5, 3, 2, 4, 3, 4, 3, 5, 2,
                                                 2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 4,
                                                 3, 4, 4, 3, 4, 5, 5, 4, 4, 3, 5, 2, 5, 4, 2, 1,
                                                 2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 2, 3, 3, 2,
                                                 3, 4, 4, 5, 4, 5, 5, 2, 4, 3, 5, 4, 3, 2, 4, 1,
                                                 3, 4, 4, 5, 4, 5, 3, 4, 4, 5, 5, 2, 3, 4, 2, 1,
                                                 2, 3, 3, 2, 3, 4, 2, 1, 3, 2, 4, 1, 2, 1, 1, 0 };

        // Marching Cube三角形查找表
        __device__ __constant__ int triangles[256][16] = {
        {  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1},
        {   0,   4,   8,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1},
        {   5,   0,   9,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1},
        {   8,   9,   5,   8,   5,   4,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1},
        {   1,   5,  11,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1},
        {   0,   4,   8,   1,   5,  11,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1},
        {   9,  11,   1,   9,   1,   0,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1},
        {   8,   9,  11,   8,  11,   1,   8,   1,   4,  -1,  -1,  -1,  -1,  -1,  -1,  -1},
        {   4,   1,  10,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1},
        {  10,   8,   0,  10,   0,   1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1},
        {   5,   0,   9,   4,   1,  10,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1},
        {  10,   8,   9,  10,   9,   5,  10,   5,   1,  -1,  -1,  -1,  -1,  -1,  -1,  -1},
        {  11,  10,   4,  11,   4,   5,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1},
        {  11,  10,   8,  11,   8,   0,  11,   0,   5,  -1,  -1,  -1,  -1,  -1,  -1,  -1},
        {   9,  11,  10,   9,  10,   4,   9,   4,   0,  -1,  -1,  -1,  -1,  -1,  -1,  -1},
        {   8,   9,  11,   8,  11,  10,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1},
        {   8,   6,   2,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1},
        {   6,   2,   0,   4,   6,   0,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1},
        {   6,   2,   8,   5,   0,   9,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1},
        {   5,   4,   6,   9,   5,   6,   2,   9,   6,  -1,  -1,  -1,  -1,  -1,  -1,  -1},
        {   1,   5,  11,   8,   6,   2,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1},
        {   1,   5,  11,   6,   2,   0,   4,   6,   0,  -1,  -1,  -1,  -1,  -1,  -1,  -1},
        {   6,   2,   8,   9,  11,   1,   9,   1,   0,  -1,  -1,  -1,  -1,  -1,  -1,  -1},
        {   9,  11,   2,   2,  11,   1,   2,   1,   6,   6,   1,   4,  -1,  -1,  -1,  -1},
        {   1,  10,   4,   2,   8,   6,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1},
        {   2,   0,   1,   6,   2,   1,  10,   6,   1,  -1,  -1,  -1,  -1,  -1,  -1,  -1},
        {   5,   0,   9,   4,   1,  10,   8,   6,   2,  -1,  -1,  -1,  -1,  -1,  -1,  -1},
        {   5,   2,   9,   5,   6,   2,   5,   1,   6,   1,  10,   6,  -1,  -1,  -1,  -1},
        {   2,   8,   6,   4,   5,  11,   4,  11,  10,  -1,  -1,  -1,  -1,  -1,  -1,  -1},
        {   5,   2,   0,   6,   2,   5,  11,   6,   5,  10,   6,  11,  -1,  -1,  -1,  -1},
        {   9,  11,  10,   9,  10,   4,   9,   4,   0,   8,   6,   2,  -1,  -1,  -1,  -1},
        {   9,  11,   2,   2,  11,   6,  10,   6,  11,  -1,  -1,  -1,  -1,  -1,  -1,  -1},
        {   9,   2,   7,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1},
        {   7,   9,   2,   4,   8,   0,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1},
        {   0,   2,   7,   0,   7,   5,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1},
        {   7,   5,   4,   2,   7,   4,   8,   2,   4,  -1,  -1,  -1,  -1,  -1,  -1,  -1},
        {   7,   9,   2,   5,  11,   1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1},
        {   1,   5,  11,   0,   4,   8,   9,   2,   7,  -1,  -1,  -1,  -1,  -1,  -1,  -1},
        {   1,   0,   2,   1,   2,   7,   1,   7,  11,  -1,  -1,  -1,  -1,  -1,  -1,  -1},
        {   1,   7,  11,   1,   2,   7,   1,   4,   2,   4,   8,   2,  -1,  -1,  -1,  -1},
        {   4,   1,  10,   9,   2,   7,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1},
        {   7,   9,   2,   0,   1,  10,   0,  10,   8,  -1,  -1,  -1,  -1,  -1,  -1,  -1},
        {   4,   1,  10,   2,   7,   5,   0,   2,   5,  -1,  -1,  -1,  -1,  -1,  -1,  -1},
        {   2,  10,   8,   1,  10,   2,   7,   1,   2,   5,   1,   7,  -1,  -1,  -1,  -1},
        {   7,   9,   2,  10,   4,   5,  11,  10,   5,  -1,  -1,  -1,  -1,  -1,  -1,  -1},
        {  11,  10,   8,  11,   8,   0,  11,   0,   5,   9,   2,   7,  -1,  -1,  -1,  -1},
        {  11,  10,   7,   7,  10,   4,   7,   4,   2,   2,   4,   0,  -1,  -1,  -1,  -1},
        {  11,  10,   7,   7,  10,   2,   8,   2,  10,  -1,  -1,  -1,  -1,  -1,  -1,  -1},
        {   7,   9,   8,   6,   7,   8,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1},
        {   4,   6,   7,   0,   4,   7,   9,   0,   7,  -1,  -1,  -1,  -1,  -1,  -1,  -1},
        {   6,   7,   5,   8,   6,   5,   0,   8,   5,  -1,  -1,  -1,  -1,  -1,  -1,  -1},
        {   4,   6,   7,   5,   4,   7,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1},
        {   5,  11,   1,   8,   6,   7,   9,   8,   7,  -1,  -1,  -1,  -1,  -1,  -1,  -1},
        {   4,   6,   7,   0,   4,   7,   9,   0,   7,  11,   1,   5,  -1,  -1,  -1,  -1},
        {   8,   1,   0,  11,   1,   8,   6,  11,   8,   7,  11,   6,  -1,  -1,  -1,  -1},
        {  11,   6,   7,   1,   6,  11,   6,   1,   4,  -1,  -1,  -1,  -1,  -1,  -1,  -1},
        {   1,  10,   4,   6,   7,   9,   6,   9,   8,  -1,  -1,  -1,  -1,  -1,  -1,  -1},
        {   0,   1,   9,   9,   1,  10,   9,  10,   7,   7,  10,   6,  -1,  -1,  -1,  -1},
        {   6,   7,   5,   8,   6,   5,   0,   8,   5,   1,  10,   4,  -1,  -1,  -1,  -1},
        {   1,   7,   5,  10,   7,   1,   7,  10,   6,  -1,  -1,  -1,  -1,  -1,  -1,  -1},
        {  11,  10,   4,  11,   4,   5,   7,   9,   8,   6,   7,   8,  -1,  -1,  -1,  -1},
        {   0,   6,   9,   9,   6,   7,   6,   0,   5,   5,  11,  10,   5,  10,   6,  -1},
        {   8,   7,   0,   6,   7,   8,   4,   0,   7,  11,  10,   4,   7,  11,   4,  -1},
        {  11,  10,   6,  11,   6,   7,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1},
        {  11,   7,   3,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1},
        {   0,   4,   8,  11,   7,   3,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1},
        {   9,   5,   0,  11,   7,   3,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1},
        {  11,   7,   3,   4,   8,   9,   5,   4,   9,  -1,  -1,  -1,  -1,  -1,  -1,  -1},
        {   3,   1,   5,   3,   5,   7,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1},
        {   0,   4,   8,   7,   3,   1,   5,   7,   1,  -1,  -1,  -1,  -1,  -1,  -1,  -1},
        {   3,   1,   0,   3,   0,   9,   3,   9,   7,  -1,  -1,  -1,  -1,  -1,  -1,  -1},
        {   7,   8,   9,   4,   8,   7,   3,   4,   7,   1,   4,   3,  -1,  -1,  -1,  -1},
        {   1,  10,   4,   3,  11,   7,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1},
        {   3,  11,   7,   8,   0,   1,  10,   8,   1,  -1,  -1,  -1,  -1,  -1,  -1,  -1},
        {   4,   1,  10,   5,   0,   9,  11,   7,   3,  -1,  -1,  -1,  -1,  -1,  -1,  -1},
        {  10,   8,   9,  10,   9,   5,  10,   5,   1,  11,   7,   3,  -1,  -1,  -1,  -1},
        {   4,   5,   7,   4,   7,   3,   4,   3,  10,  -1,  -1,  -1,  -1,  -1,  -1,  -1},
        {  10,   8,   3,   3,   8,   0,   3,   0,   7,   7,   0,   5,  -1,  -1,  -1,  -1},
        {   4,   3,  10,   4,   7,   3,   4,   0,   7,   0,   9,   7,  -1,  -1,  -1,  -1},
        {  10,   8,   3,   3,   8,   7,   9,   7,   8,  -1,  -1,  -1,  -1,  -1,  -1,  -1},
        {  11,   7,   3,   8,   6,   2,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1},
        {  11,   7,   3,   2,   0,   4,   2,   4,   6,  -1,  -1,  -1,  -1,  -1,  -1,  -1},
        {  11,   7,   3,   8,   6,   2,   5,   0,   9,  -1,  -1,  -1,  -1,  -1,  -1,  -1},
        {   5,   4,   6,   9,   5,   6,   2,   9,   6,   3,  11,   7,  -1,  -1,  -1,  -1},
        {   8,   6,   2,   3,   1,   5,   3,   5,   7,  -1,  -1,  -1,  -1,  -1,  -1,  -1},
        {   3,   1,   5,   3,   5,   7,   6,   2,   0,   4,   6,   0,  -1,  -1,  -1,  -1},
        {   3,   1,   0,   3,   0,   9,   3,   9,   7,   2,   8,   6,  -1,  -1,  -1,  -1},
        {   9,   4,   2,   2,   4,   6,   4,   9,   7,   7,   3,   1,   7,   1,   4,  -1},
        {   8,   6,   2,  11,   7,   3,   4,   1,  10,  -1,  -1,  -1,  -1,  -1,  -1,  -1},
        {   2,   0,   1,   6,   2,   1,  10,   6,   1,  11,   7,   3,  -1,  -1,  -1,  -1},
        {   5,   0,   9,   4,   1,  10,   8,   6,   2,  11,   7,   3,  -1,  -1,  -1,  -1},
        {  11,   7,   3,   5,   2,   9,   5,   6,   2,   5,   1,   6,   1,  10,   6,  -1},
        {   4,   5,   7,   4,   7,   3,   4,   3,  10,   6,   2,   8,  -1,  -1,  -1,  -1},
        {  10,   5,   3,   3,   5,   7,   5,  10,   6,   6,   2,   0,   6,   0,   5,  -1},
        {   8,   6,   2,   4,   3,  10,   4,   7,   3,   4,   0,   7,   0,   9,   7,  -1},
        {   9,   7,  10,  10,   7,   3,  10,   6,   9,   6,   2,   9,  -1,  -1,  -1,  -1},
        {   3,  11,   9,   2,   3,   9,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1},
        {   4,   8,   0,   2,   3,  11,   2,  11,   9,  -1,  -1,  -1,  -1,  -1,  -1,  -1},
        {   0,   2,   3,   0,   3,  11,   0,  11,   5,  -1,  -1,  -1,  -1,  -1,  -1,  -1},
        {   2,   3,   8,   8,   3,  11,   8,  11,   4,   4,  11,   5,  -1,  -1,  -1,  -1},
        {   2,   3,   1,   2,   1,   5,   2,   5,   9,  -1,  -1,  -1,  -1,  -1,  -1,  -1},
        {   2,   3,   1,   2,   1,   5,   2,   5,   9,   0,   4,   8,  -1,  -1,  -1,  -1},
        {   0,   2,   3,   0,   3,   1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1},
        {   2,   3,   8,   8,   3,   4,   1,   4,   3,  -1,  -1,  -1,  -1,  -1,  -1,  -1},
        {   1,  10,   4,   9,   2,   3,  11,   9,   3,  -1,  -1,  -1,  -1,  -1,  -1,  -1},
        {  10,   8,   0,  10,   0,   1,   3,  11,   9,   2,   3,   9,  -1,  -1,  -1,  -1},
        {   0,   2,   3,   0,   3,  11,   0,  11,   5,   1,  10,   4,  -1,  -1,  -1,  -1},
        {   5,   2,  11,  11,   2,   3,   2,   5,   1,   1,  10,   8,   1,   8,   2,  -1},
        {  10,   2,   3,   9,   2,  10,   4,   9,  10,   5,   9,   4,  -1,  -1,  -1,  -1},
        {   5,  10,   0,   0,  10,   8,  10,   5,   9,   9,   2,   3,   9,   3,  10,  -1},
        {   0,   2,   4,   4,   2,  10,   3,  10,   2,  -1,  -1,  -1,  -1,  -1,  -1,  -1},
        {  10,   8,   2,  10,   2,   3,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1},
        {  11,   9,   8,   3,  11,   8,   6,   3,   8,  -1,  -1,  -1,  -1,  -1,  -1,  -1},
        {   0,  11,   9,   3,  11,   0,   4,   3,   0,   6,   3,   4,  -1,  -1,  -1,  -1},
        {  11,   5,   3,   5,   0,   3,   0,   6,   3,   0,   8,   6,  -1,  -1,  -1,  -1},
        {   3,   4,   6,  11,   4,   3,   4,  11,   5,  -1,  -1,  -1,  -1,  -1,  -1,  -1},
        {   3,   1,   6,   6,   1,   5,   6,   5,   8,   8,   5,   9,  -1,  -1,  -1,  -1},
        {   0,   6,   9,   4,   6,   0,   5,   9,   6,   3,   1,   5,   6,   3,   5,  -1},
        {   3,   1,   6,   6,   1,   8,   0,   8,   1,  -1,  -1,  -1,  -1,  -1,  -1,  -1},
        {   3,   1,   4,   3,   4,   6,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1},
        {  11,   9,   8,   3,  11,   8,   6,   3,   8,   4,   1,  10,  -1,  -1,  -1,  -1},
        {   3,   9,   6,  11,   9,   3,  10,   6,   9,   0,   1,  10,   9,   0,  10,  -1},
        {   4,   1,  10,  11,   5,   3,   5,   0,   3,   0,   6,   3,   0,   8,   6,  -1},
        {   5,  10,   6,   1,  10,   5,   6,  11,   5,   6,   3,  11,  -1,  -1,  -1,  -1},
        {  10,   5,   3,   4,   5,  10,   6,   3,   5,   9,   8,   6,   5,   9,   6,  -1},
        {   6,   3,  10,   9,   0,   5,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1},
        {   3,  10,   0,   0,  10,   4,   0,   8,   3,   8,   6,   3,  -1,  -1,  -1,  -1},
        {   6,   3,  10,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1},
        {  10,   3,   6,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1},
        {   3,   6,  10,   0,   4,   8,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1},
        {   5,   0,   9,  10,   3,   6,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1},
        {   3,   6,  10,   8,   9,   5,   8,   5,   4,  -1,  -1,  -1,  -1,  -1,  -1,  -1},
        {  11,   1,   5,  10,   3,   6,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1},
        {   0,   4,   8,   1,   5,  11,  10,   3,   6,  -1,  -1,  -1,  -1,  -1,  -1,  -1},
        {  10,   3,   6,   0,   9,  11,   1,   0,  11,  -1,  -1,  -1,  -1,  -1,  -1,  -1},
        {   8,   9,  11,   8,  11,   1,   8,   1,   4,  10,   3,   6,  -1,  -1,  -1,  -1},
        {   4,   1,   3,   6,   4,   3,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1},
        {   0,   1,   3,   8,   0,   3,   6,   8,   3,  -1,  -1,  -1,  -1,  -1,  -1,  -1},
        {   5,   0,   9,   3,   6,   4,   1,   3,   4,  -1,  -1,  -1,  -1,  -1,  -1,  -1},
        {   8,   9,   6,   6,   9,   5,   6,   5,   3,   3,   5,   1,  -1,  -1,  -1,  -1},
        {   6,   4,   5,   6,   5,  11,   6,  11,   3,  -1,  -1,  -1,  -1,  -1,  -1,  -1},
        {   0,   6,   8,   0,   3,   6,   0,   5,   3,   5,  11,   3,  -1,  -1,  -1,  -1},
        {   3,   9,  11,   0,   9,   3,   6,   0,   3,   4,   0,   6,  -1,  -1,  -1,  -1},
        {   8,   9,   6,   6,   9,   3,  11,   3,   9,  -1,  -1,  -1,  -1,  -1,  -1,  -1},
        {   2,   8,  10,   3,   2,  10,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1},
        {   3,   2,   0,  10,   3,   0,   4,  10,   0,  -1,  -1,  -1,  -1,  -1,  -1,  -1},
        {   5,   0,   9,   8,  10,   3,   8,   3,   2,  -1,  -1,  -1,  -1,  -1,  -1,  -1},
        {   9,   3,   2,  10,   3,   9,   5,  10,   9,   4,  10,   5,  -1,  -1,  -1,  -1},
        {  11,   1,   5,   2,   8,  10,   3,   2,  10,  -1,  -1,  -1,  -1,  -1,  -1,  -1},
        {   3,   2,   0,  10,   3,   0,   4,  10,   0,   5,  11,   1,  -1,  -1,  -1,  -1},
        {   9,  11,   1,   9,   1,   0,   2,   8,  10,   3,   2,  10,  -1,  -1,  -1,  -1},
        {  10,   2,   4,   3,   2,  10,   1,   4,   2,   9,  11,   1,   2,   9,   1,  -1},
        {   1,   3,   2,   4,   1,   2,   8,   4,   2,  -1,  -1,  -1,  -1,  -1,  -1,  -1},
        {   0,   1,   3,   2,   0,   3,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1},
        {   1,   3,   2,   4,   1,   2,   8,   4,   2,   9,   5,   0,  -1,  -1,  -1,  -1},
        {   9,   3,   2,   5,   3,   9,   3,   5,   1,  -1,  -1,  -1,  -1,  -1,  -1,  -1},
        {   3,   2,  11,  11,   2,   8,  11,   8,   5,   5,   8,   4,  -1,  -1,  -1,  -1},
        {   5,   2,   0,  11,   2,   5,   2,  11,   3,  -1,  -1,  -1,  -1,  -1,  -1,  -1},
        {   4,   3,   8,   8,   3,   2,   3,   4,   0,   0,   9,  11,   0,  11,   3,  -1},
        {   9,  11,   3,   9,   3,   2,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1},
        {  10,   3,   6,   9,   2,   7,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1},
        {   9,   2,   7,  10,   3,   6,   0,   4,   8,  -1,  -1,  -1,  -1,  -1,  -1,  -1},
        {  10,   3,   6,   7,   5,   0,   7,   0,   2,  -1,  -1,  -1,  -1,  -1,  -1,  -1},
        {   7,   5,   4,   2,   7,   4,   8,   2,   4,  10,   3,   6,  -1,  -1,  -1,  -1},
        {  10,   3,   6,   9,   2,   7,   1,   5,  11,  -1,  -1,  -1,  -1,  -1,  -1,  -1},
        {  10,   3,   6,   9,   2,   7,   1,   5,  11,   0,   4,   8,  -1,  -1,  -1,  -1},
        {   1,   0,   2,   1,   2,   7,   1,   7,  11,   3,   6,  10,  -1,  -1,  -1,  -1},
        {  10,   3,   6,   1,   7,  11,   1,   2,   7,   1,   4,   2,   4,   8,   2,  -1},
        {   9,   2,   7,   6,   4,   1,   6,   1,   3,  -1,  -1,  -1,  -1,  -1,  -1,  -1},
        {   0,   1,   3,   8,   0,   3,   6,   8,   3,   7,   9,   2,  -1,  -1,  -1,  -1},
        {   0,   2,   7,   0,   7,   5,   4,   1,   3,   6,   4,   3,  -1,  -1,  -1,  -1},
        {   2,   5,   8,   7,   5,   2,   6,   8,   5,   1,   3,   6,   5,   1,   6,  -1},
        {   6,   4,   5,   6,   5,  11,   6,  11,   3,   7,   9,   2,  -1,  -1,  -1,  -1},
        {   9,   2,   7,   0,   6,   8,   0,   3,   6,   0,   5,   3,   5,  11,   3,  -1},
        {   3,   4,  11,   6,   4,   3,   7,  11,   4,   0,   2,   7,   4,   0,   7,  -1},
        {  11,   3,   8,   8,   3,   6,   8,   2,  11,   2,   7,  11,  -1,  -1,  -1,  -1},
        {   9,   8,  10,   7,   9,  10,   3,   7,  10,  -1,  -1,  -1,  -1,  -1,  -1,  -1},
        {   9,   0,   7,   0,   4,   7,   4,   3,   7,   4,  10,   3,  -1,  -1,  -1,  -1},
        {   8,  10,   0,   0,  10,   3,   0,   3,   5,   5,   3,   7,  -1,  -1,  -1,  -1},
        {  10,   5,   4,   3,   5,  10,   5,   3,   7,  -1,  -1,  -1,  -1,  -1,  -1,  -1},
        {   9,   8,  10,   7,   9,  10,   3,   7,  10,   1,   5,  11,  -1,  -1,  -1,  -1},
        {   1,   5,  11,   9,   0,   7,   0,   4,   7,   4,   3,   7,   4,  10,   3,  -1},
        {  11,   0,   7,   1,   0,  11,   3,   7,   0,   8,  10,   3,   0,   8,   3,  -1},
        {   7,   1,   4,  11,   1,   7,   4,   3,   7,   4,  10,   3,  -1,  -1,  -1,  -1},
        {   4,   9,   8,   7,   9,   4,   1,   7,   4,   3,   7,   1,  -1,  -1,  -1,  -1},
        {   7,   1,   3,   9,   1,   7,   1,   9,   0,  -1,  -1,  -1,  -1,  -1,  -1,  -1},
        {   8,   7,   0,   0,   7,   5,   7,   8,   4,   4,   1,   3,   4,   3,   7,  -1},
        {   5,   1,   3,   7,   5,   3,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1},
        {   3,   4,  11,  11,   4,   5,   4,   3,   7,   7,   9,   8,   7,   8,   4,  -1},
        {   3,   9,   0,   7,   9,   3,   0,  11,   3,   0,   5,  11,  -1,  -1,  -1,  -1},
        {   3,   7,  11,   8,   4,   0,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1},
        {   3,   7,  11,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1},
        {   6,  10,  11,   7,   6,  11,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1},
        {   0,   4,   8,  10,  11,   7,  10,   7,   6,  -1,  -1,  -1,  -1,  -1,  -1,  -1},
        {   9,   5,   0,   6,  10,  11,   7,   6,  11,  -1,  -1,  -1,  -1,  -1,  -1,  -1},
        {   8,   9,   5,   8,   5,   4,   6,  10,  11,   7,   6,  11,  -1,  -1,  -1,  -1},
        {   5,   7,   6,   5,   6,  10,   5,  10,   1,  -1,  -1,  -1,  -1,  -1,  -1,  -1},
        {   5,   7,   6,   5,   6,  10,   5,  10,   1,   4,   8,   0,  -1,  -1,  -1,  -1},
        {   1,   0,  10,  10,   0,   9,  10,   9,   6,   6,   9,   7,  -1,  -1,  -1,  -1},
        {   1,   7,  10,  10,   7,   6,   7,   1,   4,   4,   8,   9,   4,   9,   7,  -1},
        {   7,   6,   4,   7,   4,   1,   7,   1,  11,  -1,  -1,  -1,  -1,  -1,  -1,  -1},
        {  11,   0,   1,   8,   0,  11,   7,   8,  11,   6,   8,   7,  -1,  -1,  -1,  -1},
        {   7,   6,   4,   7,   4,   1,   7,   1,  11,   5,   0,   9,  -1,  -1,  -1,  -1},
        {  11,   6,   1,   7,   6,  11,   5,   1,   6,   8,   9,   5,   6,   8,   5,  -1},
        {   4,   5,   7,   4,   7,   6,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1},
        {   5,   7,   0,   0,   7,   8,   6,   8,   7,  -1,  -1,  -1,  -1,  -1,  -1,  -1},
        {   7,   6,   9,   9,   6,   0,   4,   0,   6,  -1,  -1,  -1,  -1,  -1,  -1,  -1},
        {   8,   9,   7,   8,   7,   6,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1},
        {   8,  10,  11,   2,   8,  11,   7,   2,  11,  -1,  -1,  -1,  -1,  -1,  -1,  -1},
        {  10,  11,   4,   4,  11,   7,   4,   7,   0,   0,   7,   2,  -1,  -1,  -1,  -1},
        {   8,  10,  11,   2,   8,  11,   7,   2,  11,   5,   0,   9,  -1,  -1,  -1,  -1},
        {   9,   4,   2,   5,   4,   9,   7,   2,   4,  10,  11,   7,   4,  10,   7,  -1},
        {   1,   8,  10,   2,   8,   1,   5,   2,   1,   7,   2,   5,  -1,  -1,  -1,  -1},
        {   1,   7,  10,   5,   7,   1,   4,  10,   7,   2,   0,   4,   7,   2,   4,  -1},
        {   7,   1,   9,   9,   1,   0,   1,   7,   2,   2,   8,  10,   2,  10,   1,  -1},
        {   7,   2,   9,  10,   1,   4,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1},
        {   8,   4,   2,   4,   1,   2,   1,   7,   2,   1,  11,   7,  -1,  -1,  -1,  -1},
        {  11,   0,   1,   7,   0,  11,   0,   7,   2,  -1,  -1,  -1,  -1,  -1,  -1,  -1},
        {   5,   0,   9,   8,   4,   2,   4,   1,   2,   1,   7,   2,   1,  11,   7,  -1},
        {   2,   5,   1,   9,   5,   2,   1,   7,   2,   1,  11,   7,  -1,  -1,  -1,  -1},
        {   4,   5,   8,   8,   5,   2,   7,   2,   5,  -1,  -1,  -1,  -1,  -1,  -1,  -1},
        {   7,   2,   0,   5,   7,   0,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1},
        {   7,   2,   4,   4,   2,   8,   4,   0,   7,   0,   9,   7,  -1,  -1,  -1,  -1},
        {   7,   2,   9,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1},
        {  10,  11,   9,   6,  10,   9,   2,   6,   9,  -1,  -1,  -1,  -1,  -1,  -1,  -1},
        {  10,  11,   9,   6,  10,   9,   2,   6,   9,   0,   4,   8,  -1,  -1,  -1,  -1},
        {   5,  10,  11,   6,  10,   5,   0,   6,   5,   2,   6,   0,  -1,  -1,  -1,  -1},
        {   2,   5,   8,   8,   5,   4,   5,   2,   6,   6,  10,  11,   6,  11,   5,  -1},
        {  10,   1,   6,   1,   5,   6,   5,   2,   6,   5,   9,   2,  -1,  -1,  -1,  -1},
        {   0,   4,   8,  10,   1,   6,   1,   5,   6,   5,   2,   6,   5,   9,   2,  -1},
        {   1,   0,  10,  10,   0,   6,   2,   6,   0,  -1,  -1,  -1,  -1,  -1,  -1,  -1},
        {   2,   6,   1,   1,   6,  10,   1,   4,   2,   4,   8,   2,  -1,  -1,  -1,  -1},
        {  11,   9,   1,   1,   9,   2,   1,   2,   4,   4,   2,   6,  -1,  -1,  -1,  -1},
        {   8,   1,   6,   0,   1,   8,   2,   6,   1,  11,   9,   2,   1,  11,   2,  -1},
        {  11,   6,   1,   1,   6,   4,   6,  11,   5,   5,   0,   2,   5,   2,   6,  -1},
        {   2,   6,   8,  11,   5,   1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1},
        {   6,   4,   2,   2,   4,   9,   5,   9,   4,  -1,  -1,  -1,  -1,  -1,  -1,  -1},
        {   5,   9,   6,   6,   9,   2,   6,   8,   5,   8,   0,   5,  -1,  -1,  -1,  -1},
        {   0,   2,   6,   0,   6,   4,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1},
        {   2,   6,   8,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1},
        {   8,  10,  11,   9,   8,  11,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1},
        {   0,  11,   9,   4,  11,   0,  11,   4,  10,  -1,  -1,  -1,  -1,  -1,  -1,  -1},
        {   5,  10,  11,   0,  10,   5,  10,   0,   8,  -1,  -1,  -1,  -1,  -1,  -1,  -1},
        {   4,  10,  11,   5,   4,  11,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1},
        {   1,   8,  10,   5,   8,   1,   8,   5,   9,  -1,  -1,  -1,  -1,  -1,  -1,  -1},
        {   9,   4,  10,   0,   4,   9,  10,   5,   9,  10,   1,   5,  -1,  -1,  -1,  -1},
        {   0,   8,  10,   1,   0,  10,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1},
        {  10,   1,   4,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1},
        {   4,   9,   8,   1,   9,   4,   9,   1,  11,  -1,  -1,  -1,  -1,  -1,  -1,  -1},
        {   1,  11,   9,   0,   1,   9,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1},
        {  11,   0,   8,   5,   0,  11,   8,   1,  11,   8,   4,   1,  -1,  -1,  -1,  -1},
        {  11,   5,   1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1},
        {   5,   9,   8,   4,   5,   8,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1},
        {   9,   0,   5,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1},
        {   8,   4,   0,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1},
        {  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1} };
        
        __device__ __constant__ int faceEdges[6][4] = { {4,  6,  8,  10},
                                                        {5,  7,  9,  11},
                                                        {0,  2,  8,  9 },
                                                        {1,  3,  10, 11},
                                                        {0,  1,  4,  5 },
                                                        {2,  3,  6,  7 } };

        __device__ __constant__ int LUTparent[8][27] = {
        {0,1,1,3,4,4,3,4,4,9,10,10,12,13,13,12,13,13,9,10,10,12,13,13,12,13,13},
        {1,1,2,4,4,5,4,4,5,10,10,11,13,13,14,13,13,14,10,10,11,13,13,14,13,13,14},
        {3,4,4,3,4,4,6,7,7,12,13,13,12,13,13,15,16,16,12,13,13,12,13,13,15,16,16},
        {4,4,5,4,4,5,7,7,8,13,13,14,13,13,14,16,16,17,13,13,14,13,13,14,16,16,17},
        {9,10,10,12,13,13,12,13,13,9,10,10,12,13,13,12,13,13,18,19,19,21,22,22,21,22,22},
        {10,10,11,13,13,14,13,13,14,10,10,11,13,13,14,13,13,14,19,19,20,22,22,23,22,22,23},
        {12,13,13,12,13,13,15,16,16,12,13,13,12,13,13,15,16,16,21,22,22,21,22,22,24,25,25},
        {13,13,14,13,13,14,16,16,17,13,13,14,13,13,14,16,16,17,22,22,23,22,22,23,25,25,26}
        };
        __device__ __constant__ int LUTchild[8][27] = {
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
__global__ void SparseSurfelFusion::device::ComputeVertexImplicitFunctionValueKernel(DeviceArrayView<VertexNode> VertexArray, DeviceArrayView<OctNode> NodeArray, DeviceArrayView<ConfirmedPPolynomial<CONVTIMES + 1, CONVTIMES + 2>> BaseFunctions, DeviceArrayView<float> dx, DeviceArrayView<int> encodeNodeIndexInFunction, const unsigned int VertexArraySize, const float isoValue, float* vvalue)
{
    const unsigned int idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (idx >= VertexArraySize)	return;
    VertexNode nowVertex = VertexArray[idx];
    int depth = nowVertex.depth;
    float val = 0.0f;
    int exceedChildrenId = childrenVertexKind[nowVertex.vertexKind];
    int nowNode = nowVertex.ownerNodeIdx;
    if (nowNode > 0) {
        while (nowNode != -1) {
            for (int i = 0; i < 27; i++) {
                int neighbor = NodeArray[nowNode].neighs[i];
                if (neighbor != -1) {
                    int idxO[3];
                    int encode_idx = encodeNodeIndexInFunction[neighbor];
                    idxO[0] = encode_idx % decodeOffset_1;
                    idxO[1] = (encode_idx / decodeOffset_1) % decodeOffset_1;
                    idxO[2] = encode_idx / decodeOffset_2;

                    ConfirmedPPolynomial<CONVTIMES + 1, CONVTIMES + 2> funcX = BaseFunctions[idxO[0]];
                    ConfirmedPPolynomial<CONVTIMES + 1, CONVTIMES + 2> funcY = BaseFunctions[idxO[1]];
                    ConfirmedPPolynomial<CONVTIMES + 1, CONVTIMES + 2> funcZ = BaseFunctions[idxO[2]];

                    val += dx[neighbor] * value(funcX, nowVertex.pos.coords[0]) * value(funcY, nowVertex.pos.coords[1]) * value(funcZ, nowVertex.pos.coords[2]);
                }
            }
            nowNode = NodeArray[nowNode].parent;
        }
        nowNode = nowVertex.ownerNodeIdx;
        while (depth < device::maxDepth) {
            depth++;
            nowNode = NodeArray[nowNode].children[exceedChildrenId];
            if (nowNode == -1) break;
            for (int i = 0; i < 27; i++) {
                int neighbor = NodeArray[nowNode].neighs[i];
                if (neighbor != -1) {
                    int idxO[3];
                    int encode_idx = encodeNodeIndexInFunction[neighbor];
                    idxO[0] = encode_idx % decodeOffset_1;
                    idxO[1] = (encode_idx / decodeOffset_1) % decodeOffset_1;
                    idxO[2] = encode_idx / decodeOffset_2;

                    ConfirmedPPolynomial<CONVTIMES + 1, CONVTIMES + 2> funcX = BaseFunctions[idxO[0]];
                    ConfirmedPPolynomial<CONVTIMES + 1, CONVTIMES + 2> funcY = BaseFunctions[idxO[1]];
                    ConfirmedPPolynomial<CONVTIMES + 1, CONVTIMES + 2> funcZ = BaseFunctions[idxO[2]];

                    val += dx[neighbor] * value(funcX, nowVertex.pos.coords[0]) * value(funcY, nowVertex.pos.coords[1]) * value(funcZ, nowVertex.pos.coords[2]);
                }
            }
        }
    }
    vvalue[idx] = val - isoValue;
}

__global__ void SparseSurfelFusion::device::generateVertexNumsKernel(DeviceArrayView<EdgeNode> EdgeArray, DeviceArrayView<OctNode> NodeArray, DeviceArrayView<float> vvalue, const unsigned int EdgeArraySize, int* vexNums, bool* markValidVertex)
{
    const unsigned int idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (idx >= EdgeArraySize)	return;
    EdgeNode nowEdge = EdgeArray[idx];
    int owner = nowEdge.ownerNodeIdx;
    int kind = nowEdge.edgeKind;
    int index[2];
    index[0] = device::edgeVertex[kind][0];
    index[1] = device::edgeVertex[kind][1];

    int v1 = NodeArray[owner].vertices[index[0]] - 1;
    int v2 = NodeArray[owner].vertices[index[1]] - 1;
    if (vvalue[v1] * vvalue[v2] <= 0) {
        vexNums[idx] = 1;
        markValidVertex[idx] = true;
    }
    else {
        vexNums[idx] = 0;
        markValidVertex[idx] = false;
    }
}

__global__ void SparseSurfelFusion::device::generateTriangleNumsKernel(DeviceArrayView<OctNode> NodeArray, DeviceArrayView<float> vvalue, const unsigned int DLevelOffset, const unsigned int DLevelNodeCount, int* triNums, int* cubeCatagory)
{
    const unsigned int idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (idx >= DLevelNodeCount)	return;
    const unsigned int offset = DLevelOffset + idx;
    OctNode currentNode = NodeArray[offset];    // 当前最大层节点
    int currentCubeCatagory = 0;                // 立方体类型
    for (int i = 0; i < 8; i++) {
        if (vvalue[currentNode.vertices[i] - 1] < 0) {
            currentCubeCatagory |= 1 << i;
        }
    }
    triNums[idx] = device::trianglesCount[currentCubeCatagory];
    cubeCatagory[idx] = currentCubeCatagory;
}

__global__ void SparseSurfelFusion::device::generateIntersectionPoint(DeviceArrayView<OctNode> NodeArray, DeviceArrayView<VertexNode> VertexArray, DeviceArrayView<float> vvalue, const EdgeNode* validEdgeArray, const int* validVexAddress, const unsigned int validEdgeArraySize, Point3D<float>* VertexBuffer)
{
    const unsigned int idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (idx >= validEdgeArraySize)	return;
    int owner = validEdgeArray[idx].ownerNodeIdx;
    int kind = validEdgeArray[idx].edgeKind;
    int orientation = kind >> 2;

    int index[2];

    index[0] = edgeVertex[kind][0];
    index[1] = edgeVertex[kind][1];

    int v1 = NodeArray[owner].vertices[index[0]] - 1;
    int v2 = NodeArray[owner].vertices[index[1]] - 1;
    Point3D<float> p1 = VertexArray[v1].pos; 
    Point3D<float> p2 = VertexArray[v2].pos;
    float f1 = vvalue[v1];
    float f2 = vvalue[v2];
    Point3D<float> isoPoint;
    interpolatePoint(p1, p2, orientation, f1, f2, isoPoint);
    VertexBuffer[validVexAddress[idx]] = isoPoint;
}

__device__ void SparseSurfelFusion::device::interpolatePoint(const Point3D<float>& p1, const Point3D<float>& p2, const int& dim, const float& v1, const float& v2, Point3D<float>& out)
{
    for (int i = 0; i < 3; i++) {
        if (i != dim) {
            out.coords[i] = p1.coords[i];
        }
    }
    float pivot = v1 / (v1 - v2);
    float anotherPivot = 1 - pivot;
    out.coords[dim] = p2.coords[dim] * pivot + p1.coords[dim] * anotherPivot;
}

__global__ void SparseSurfelFusion::device::generateTrianglePos(DeviceArrayView<OctNode> NodeArray, DeviceArrayView<FaceNode> FaceArray, DeviceArrayView<int> triNums, DeviceArrayView<int> cubeCatagory, DeviceArrayView<int> vexAddress, DeviceArrayView<int> triAddress, const unsigned int DLevelOffset, const unsigned int DLevelNodeCount, TriangleIndex* TriangleBuffer, int* hasSurfaceIntersection)
{
    const unsigned int idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (idx >= DLevelNodeCount)	return;
    const unsigned int offset = DLevelOffset + idx;
    OctNode currentNode = NodeArray[offset];
    int currentTriNum = triNums[idx];
    int currentCubeCatagory = cubeCatagory[idx];
    int currentTriangleBufferStart = triAddress[idx];
    int edgeHasVertex[12] = { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 };

    //if (idx % 1000 == 0) printf("index = %d   TriNum = %d\n", idx, currentTriNum);

    for (int i = 0; i < currentTriNum; i++) {
        int edgeIdx[3];
        edgeIdx[0] = device::triangles[currentCubeCatagory][3 * i];
        edgeIdx[1] = device::triangles[currentCubeCatagory][3 * i + 1];
        edgeIdx[2] = device::triangles[currentCubeCatagory][3 * i + 2];

        edgeHasVertex[edgeIdx[0]] = 1;
        edgeHasVertex[edgeIdx[1]] = 1;
        edgeHasVertex[edgeIdx[2]] = 1;

        int vertexIdx[3];
        vertexIdx[0] = vexAddress[currentNode.edges[edgeIdx[0]] - 1];
        vertexIdx[1] = vexAddress[currentNode.edges[edgeIdx[1]] - 1];
        vertexIdx[2] = vexAddress[currentNode.edges[edgeIdx[2]] - 1];

        TriangleBuffer[currentTriangleBufferStart + i].idx[0] = vertexIdx[0];
        TriangleBuffer[currentTriangleBufferStart + i].idx[1] = vertexIdx[1];
        TriangleBuffer[currentTriangleBufferStart + i].idx[2] = vertexIdx[2];
    }
    int currentFace;
    int parentNodeIndex;
    for (int i = 0; i < 6; i++) {
        int mark = 0;              // 记录是否存在Surface-Edge Intersections(面边相交)
        for (int j = 0; j < 4; j++) {
            mark |= edgeHasVertex[device::faceEdges[i][j]];
        }
        if (mark == 1) {
            parentNodeIndex = NodeArray[offset].parent;
            currentFace = currentNode.faces[i] - 1;
            hasSurfaceIntersection[currentFace] = 1;
            while (FaceArray[currentFace].hasParentFace != -1) {
                currentFace = NodeArray[parentNodeIndex].faces[i] - 1;
                parentNodeIndex = NodeArray[parentNodeIndex].parent;
                hasSurfaceIntersection[currentFace] = 1;
            }
        }
    }
}

__global__ void SparseSurfelFusion::device::generateSubdivideTrianglePos(const EasyOctNode* SubdivideArray, const unsigned int DLevelOffset, const unsigned int DLevelNodeCount, const int* SubdivideTriNums, const int* SubdivideCubeCatagory, const int* SubdivideVexAddress, const int* SubdivideTriAddress, TriangleIndex* SubdivideTriangleBuffer)
{
    const unsigned int idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (idx >= DLevelNodeCount)	return;
    const unsigned int offset = DLevelOffset + idx;
    int nowTriNum = SubdivideTriNums[idx];
    int nowCubeCatagory = SubdivideCubeCatagory[idx];
    int nowTriangleBufferStart = SubdivideTriAddress[idx];
    //if (flag == true && nowTriNum != 0) printf("idx = %d   nowTriNum = %d   nowTriangleBufferStart = %d\n", idx, nowTriNum, nowTriangleBufferStart);
    for (int i = 0; i < nowTriNum; i++) {
        int edgeIdx[3];
        edgeIdx[0] = triangles[nowCubeCatagory][3 * i];
        edgeIdx[1] = triangles[nowCubeCatagory][3 * i + 1];
        edgeIdx[2] = triangles[nowCubeCatagory][3 * i + 2];

        int vertexIdx[3];
        vertexIdx[0] = SubdivideVexAddress[SubdivideArray[offset].edges[edgeIdx[0]] - 1];
        vertexIdx[1] = SubdivideVexAddress[SubdivideArray[offset].edges[edgeIdx[1]] - 1];
        vertexIdx[2] = SubdivideVexAddress[SubdivideArray[offset].edges[edgeIdx[2]] - 1];
        SubdivideTriangleBuffer[nowTriangleBufferStart + i].idx[0] = vertexIdx[0];
        SubdivideTriangleBuffer[nowTriangleBufferStart + i].idx[1] = vertexIdx[1];
        SubdivideTriangleBuffer[nowTriangleBufferStart + i].idx[2] = vertexIdx[2];
        //if (flag == true)    printf("idx = %d   nowTriangleBufferStart = %d   vertexIdx[%d] = (%d, %d, %d)\n", idx, nowTriangleBufferStart, nowTriangleBufferStart + i, vertexIdx[0], vertexIdx[1], vertexIdx[2]);

    }
}

__global__ void SparseSurfelFusion::device::ProcessLeafNodesAtOtherDepth(DeviceArrayView<VertexNode> VertexArray, DeviceArrayView<float> vvalue, const unsigned int OtherDepthNodeCount, const int* hasSurfaceIntersection, OctNode* NodeArray, bool* markValidSubdividedNode)
{
    const unsigned int idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (idx >= OtherDepthNodeCount)	return;
    OctNode currentNode = NodeArray[idx];
    int hasTri = 0;
    int sign = (vvalue[currentNode.vertices[0] - 1] < 0) ? -1 : 1;
    for (int i = 1; i < 8; i++) {
        if (sign * vvalue[currentNode.vertices[i] - 1] < 0) {
            hasTri = 1;
            break;
        }
    }

    NodeArray[idx].hasTriangle = hasTri;

    int hasIntersection = 0;
    for (int i = 0; i < 6; i++) {
        if (hasSurfaceIntersection[currentNode.faces[i] - 1]) {
            hasIntersection = 1;
            break;
        }
    }
    NodeArray[idx].hasIntersection = hasIntersection;


    if ((NodeArray[idx].children[0] == -1) && (hasTri || hasIntersection)) {
        markValidSubdividedNode[idx] = true;
    }
    else {
        markValidSubdividedNode[idx] = false;
    }
}

__global__ void SparseSurfelFusion::device::precomputeSubdivideDepth(DeviceArrayView<OctNode> SubdivideNode, DeviceArrayView<unsigned int> DepthBuffer, const int SubdivideNum, int* SubdivideDepthBuffer, int* SubdivideDepthNum)
{
    const unsigned int idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (idx >= SubdivideNum)	return;
    int nodeIndex = SubdivideNode[idx].neighs[13];
    int depth = DepthBuffer[nodeIndex];
    SubdivideDepthBuffer[idx] = depth;
    SubdivideDepthNum[idx + depth * SubdivideNum] = 1;
}

__global__ void SparseSurfelFusion::device::singleRebuildArray(DeviceArrayView<OctNode> SubdivideNode, DeviceArrayView<int> SubdivideDepthBuffer, const unsigned int iterRound, const unsigned int NodeArraySize, const unsigned int SubdivideArraySize, EasyOctNode* SubdivideArray, int* SubdivideArrayDepthBuffer, Point3D<float>* SubdivideArrayCenterBuffer)
{
    const unsigned int idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (idx >= SubdivideArraySize)	return;
    int rootId = SubdivideNode[iterRound].neighs[13];
    int rootDepth = SubdivideDepthBuffer[iterRound];
    int rootKey = SubdivideNode[iterRound].key;
    int thisNodeDepth = getSubdivideDepth(rootDepth, idx);
    int relativeDepth = thisNodeDepth - rootDepth;
    int idxOffset = idx - (powf(8, relativeDepth) - 1) / 7;

    if (thisNodeDepth < maxDepth) {
        int nextDepthAddress = (powf(8, relativeDepth + 1) - 1) / 7;
        for (int k = 0; k < 8; ++k) {
            SubdivideArray[idx].children[k] = NodeArraySize + nextDepthAddress + (idxOffset << 3) + k;
        }
    }
    else {
        for (int k = 0; k < 8; ++k) {
            SubdivideArray[idx].children[k] = -1;
        }
    }

    if (idx != 0) {
        int parentDepthAddress = (powf(8, relativeDepth - 1) - 1) / 7;
        SubdivideArray[idx].parent = NodeArraySize + parentDepthAddress + (idxOffset >> 3);
    }

    int thisKey = rootKey;
    thisKey |= (idxOffset) << (3 * (maxDepth - thisNodeDepth));
    SubdivideArray[idx].key = thisKey;

    SubdivideArrayDepthBuffer[idx] = thisNodeDepth;
    Point3D<float> thisNodeCenter;
    getNodeCenterAllDepth(thisKey, thisNodeDepth, thisNodeCenter);
    SubdivideArrayCenterBuffer[idx] = thisNodeCenter;
}

__device__ int SparseSurfelFusion::device::getSubdivideDepth(const int& rootDepth, const int& idx)
{
    int up = idx * 7 + 1;
    int base = 8;
    int relativeDepth = 0;
    while (base <= up) {
        relativeDepth++;
        base <<= 3;
    }
    return rootDepth + relativeDepth;
}

__device__ void SparseSurfelFusion::device::getNodeCenterAllDepth(const int& key, const int& currentDepth, Point3D<float>& center)
{
    center.coords[0] = float(0.5);
    center.coords[1] = float(0.5);
    center.coords[2] = float(0.5);
    float Width = 0.25f;
    for (int i = device::maxDepth - 1; i >= (device::maxDepth - currentDepth); --i) {
        if ((key >> (3 * i + 2)) & 1) center.coords[0] += Width;
        else center.coords[0] -= Width;

        if ((key >> (3 * i + 1)) & 1) center.coords[1] += Width;
        else center.coords[1] -= Width;

        if ((key >> (3 * i)) & 1) center.coords[2] += Width;
        else center.coords[2] -= Width;

        Width /= 2;
    }
}

__global__ void SparseSurfelFusion::device::computeRebuildNeighbor(DeviceArrayView<OctNode> NodeArray, const unsigned int currentLevelOffset, const unsigned int currentLevelNodesCount, const unsigned int NodeArraySize, const unsigned int depth, EasyOctNode* SubdivideArray)
{
    const unsigned int idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (idx >= currentLevelNodesCount)	return;
    const unsigned int offset = currentLevelOffset + idx;
    for (int i = 0; i < 27; i++) {
        int sonKey = (SubdivideArray[offset].key >> (3 * (device::maxDepth - depth))) & 7;
        int parentIdx = SubdivideArray[offset].parent;
        int neighParent;
        if (parentIdx < NodeArraySize) {
            neighParent = NodeArray[parentIdx].neighs[device::LUTparent[sonKey][i]];
        }
        else {
            neighParent = SubdivideArray[parentIdx - NodeArraySize].neighs[device::LUTparent[sonKey][i]];
        }
        if (neighParent != -1) {
            if (neighParent < NodeArraySize) {
                SubdivideArray[offset].neighs[i] = NodeArray[neighParent].children[device::LUTchild[sonKey][i]];
            }
            else {
                SubdivideArray[offset].neighs[i] = SubdivideArray[neighParent - NodeArraySize].children[device::LUTchild[sonKey][i]];
            }
        }
        else {
            SubdivideArray[offset].neighs[i] = -1;
        }
    }
}

__global__ void SparseSurfelFusion::device::initSubdivideVertexOwner(const EasyOctNode* SubdivideArray, const Point3D<float>* SubdivideArrayCenterBuffer, const unsigned int currentLevelOffset, const unsigned int currentLevelNodesCount, const unsigned int NodeArraySize, VertexNode* SubdividePreVertexArray, bool* markValidSubdivideVertex)
{
    const unsigned int idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (idx >= currentLevelNodesCount)	return;
    const unsigned int offset = currentLevelOffset + idx;
    int NodeOwnerKey[8] = { device::maxIntValue,device::maxIntValue, device::maxIntValue, device::maxIntValue,
                            device::maxIntValue, device::maxIntValue, device::maxIntValue, device::maxIntValue };
    int NodeOwnerIdx[8] = { -1, -1, -1, -1, -1, -1, -1, -1 };
    int depth = maxDepth;
    float halfWidth = 1.0f / (1 << (depth + 1));
    float Width = 1.0f / (1 << depth);
    float Widthsq = Width * Width;
    Point3D<float> neighCenter[27];
    int neigh[27];
#pragma unroll
    for (int i = 0; i < 27; i++) {
        neigh[i] = SubdivideArray[offset].neighs[i];
        if (neigh[i] != -1 && neigh[i] >= NodeArraySize) {
            neighCenter[i] = SubdivideArrayCenterBuffer[neigh[i] - NodeArraySize];
        }
    }
    const Point3D<float>& nodeCenter = neighCenter[13];

    Point3D<float> vertexPos[8];
#pragma unroll
    for (int i = 0; i < 8; i++) {
        vertexPos[i].coords[0] = nodeCenter.coords[0] + (2 * (i & 1) - 1) * halfWidth;
        vertexPos[i].coords[1] = nodeCenter.coords[1] + (2 * ((i & 2) >> 1) - 1) * halfWidth;
        vertexPos[i].coords[2] = nodeCenter.coords[2] + (2 * ((i & 4) >> 2) - 1) * halfWidth;
    }

#pragma unroll

    for (int i = 0; i < 8; i++) {
        for (int j = 0; j < 27; j++) {
            if (neigh[j] != -1 && SquareDistance(vertexPos[i], neighCenter[j]) < Widthsq) {
                int neighKey;
                if (neigh[j] < NodeArraySize) continue;
                else
                    neighKey = SubdivideArray[neigh[j] - NodeArraySize].key;
                if (NodeOwnerKey[i] > neighKey) {
                    NodeOwnerKey[i] = neighKey;
                    NodeOwnerIdx[i] = neigh[j];
                }
            }
        }
    }
#pragma unroll
    for (int i = 0; i < 8; i++) {
        int vertexIdx = 8 * idx + i;
        if (NodeOwnerIdx[i] == NodeArraySize + offset) {
            SubdividePreVertexArray[vertexIdx].ownerNodeIdx = NodeOwnerIdx[i];
            SubdividePreVertexArray[vertexIdx].pos.coords[0] = vertexPos[i].coords[0];
            SubdividePreVertexArray[vertexIdx].pos.coords[1] = vertexPos[i].coords[1];
            SubdividePreVertexArray[vertexIdx].pos.coords[2] = vertexPos[i].coords[2];
            SubdividePreVertexArray[vertexIdx].vertexKind = i;
            SubdividePreVertexArray[vertexIdx].depth = depth;
            markValidSubdivideVertex[vertexIdx] = true;
            //printf("idx = %d   vertexIdx = %d\n", idx, vertexIdx);
        }
        else {
            markValidSubdivideVertex[vertexIdx] = false;
        }
    }

}

__global__ void SparseSurfelFusion::device::maintainSubdivideVertexNodePointer(DeviceArrayView<Point3D<float>> CenterBuffer, const unsigned int VertexArraySize, const unsigned int NodeArraySize, const Point3D<float>* SubdivideArrayCenterBuffer, VertexNode* VertexArray, EasyOctNode* SubdivideArray)
{
    const unsigned int idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (idx >= VertexArraySize)	return;
    int owner = VertexArray[idx].ownerNodeIdx;
    float Width = 1.0f / (1 << device::maxDepth);
    float WidthSquare = Width * Width;
    Point3D<float> neighCenter[27];
    Point3D<float> vertexPos = VertexArray[idx].pos;

    int neigh[27];
    for (int i = 0; i < 27; i++) {
        neigh[i] = SubdivideArray[owner - NodeArraySize].neighs[i];
    }
    for (int i = 0; i < 27; i++) {
        if (neigh[i] != -1) {
            if (neigh[i] < NodeArraySize) {
                neighCenter[i] = CenterBuffer[neigh[i]];
            }
            else {
                neighCenter[i] = SubdivideArrayCenterBuffer[neigh[i] - NodeArraySize];
            }
        }
    }
    int count = 0;
    for (int i = 0; i < 27; i++) {
        if (neigh[i] != -1 && SquareDistance(vertexPos, neighCenter[i]) < WidthSquare) {
            VertexArray[idx].nodes[count] = neigh[i];
            count++;
            int index = 0;
            if (neighCenter[i].coords[0] - vertexPos.coords[0] < 0) index |= 1;
            if (neighCenter[i].coords[2] - vertexPos.coords[2] < 0) index |= 4;
            if (neighCenter[i].coords[1] - vertexPos.coords[1] < 0) {
                if (index & 1) {
                    index += 1;
                }
                else {
                    index += 3;
                }
            }
            if (neigh[i] >= NodeArraySize) {
                SubdivideArray[neigh[i] - NodeArraySize].vertices[index] = idx + 1;
            }
        }
    }
}

__global__ void SparseSurfelFusion::device::initSubdivideEdgeArray(const EasyOctNode* SubdivideArray, const Point3D<float>* SubdivideArrayCenterBuffer, const unsigned int NodeArraySize, const unsigned int DLevelOffset, const unsigned int DLevelNodeCount, EdgeNode* SubdividePreEdgeArray, bool* markValidSubdivideEdge)
{
    const unsigned int idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (idx >= DLevelNodeCount)	return;
    const unsigned int offset = DLevelOffset + idx;
    int NodeOwnerKey[12] = { device::maxIntValue, device::maxIntValue, device::maxIntValue,
                             device::maxIntValue, device::maxIntValue, device::maxIntValue,
                             device::maxIntValue, device::maxIntValue, device::maxIntValue,
                             device::maxIntValue, device::maxIntValue, device::maxIntValue };
    int NodeOwnerIdx[12] = { -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 };
    float halfWidth = 1.0f / (1 << (device::maxDepth + 1));
    float Width = 1.0f / (1 << device::maxDepth);
    float WidthSquare = Width * Width;
    Point3D<float> neighCenter[27];
    int neigh[27];
#pragma unroll
    for (int i = 0; i < 27; i++) {
        neigh[i] = SubdivideArray[offset].neighs[i];
        if (neigh[i] != -1 && neigh[i] >= NodeArraySize) {
            neighCenter[i] = SubdivideArrayCenterBuffer[neigh[i] - NodeArraySize];
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
            if (neigh[j] != -1 && SquareDistance(edgeCenterPos[i], neighCenter[j]) < WidthSquare) {
                int neighKey;
                if (neigh[j] < NodeArraySize) continue;
                else
                    neighKey = SubdivideArray[neigh[j] - NodeArraySize].key;
                if (NodeOwnerKey[i] > neighKey) {
                    NodeOwnerKey[i] = neighKey;
                    NodeOwnerIdx[i] = neigh[j];
                }
            }
        }
    }
#pragma unroll
    for (int i = 0; i < 12; i++) {
        int edgeIdx = 12 * idx + i;
        if (NodeOwnerIdx[i] == offset + NodeArraySize) {
            SubdividePreEdgeArray[edgeIdx].ownerNodeIdx = NodeOwnerIdx[i];
            SubdividePreEdgeArray[edgeIdx].edgeKind = i;
            markValidSubdivideEdge[edgeIdx] = true;
        }
        else {
            markValidSubdivideEdge[edgeIdx] = false;
        }
    }
}

__global__ void SparseSurfelFusion::device::maintainSubdivideEdgeNodePointer(DeviceArrayView<Point3D<float>> CenterBuffer, const Point3D<float>* SubdivideArrayCenterBuffer, const unsigned int EdgeArraySize, const unsigned int NodeArraySize, EasyOctNode* SubdivideArray, EdgeNode* EdgeArray)
{
    const unsigned int idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (idx >= EdgeArraySize)	return;
    int owner = EdgeArray[idx].ownerNodeIdx;

    int depth = device::maxDepth;
    float halfWidth = 1.0f / (1 << (depth + 1));
    float Width = 1.0f / (1 << depth);
    float WidthSquare = Width * Width;

    Point3D<float> neighCenter[27];
    int neighbor[27];
    for (int i = 0; i < 27; i++) {
        neighbor[i] = SubdivideArray[owner - NodeArraySize].neighs[i];
        if (neighbor[i] != -1) {
            if (neighbor[i] < NodeArraySize)
                neighCenter[i] = CenterBuffer[neighbor[i]];
            else
                neighCenter[i] = SubdivideArrayCenterBuffer[neighbor[i] - NodeArraySize];
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
    for (int i = 0; i < 27; i++) {
        if (neighbor[i] != -1 && SquareDistance(edgeCenterPos, neighCenter[i]) < WidthSquare) {
            EdgeArray[idx].nodes[count] = neighbor[i];
            count++;
            int index = orientation << 2;
            int dim = 0;
            for (int j = 0; j < 3; j++) {
                if (orientation != j) {
                    if (neighCenter[i].coords[j] - edgeCenterPos.coords[j] < 0) index |= (1 << dim);
                    dim++;
                }
            }
            if (neighbor[i] >= NodeArraySize)
                SubdivideArray[neighbor[i] - NodeArraySize].edges[index] = idx + 1;
        }
    }
}

__forceinline__ __device__ double SparseSurfelFusion::device::SquareDistance(const Point3D<float>& p1, const Point3D<float>& p2)
{
    return (p1.coords[0] - p2.coords[0]) * (p1.coords[0] - p2.coords[0]) + (p1.coords[1] - p2.coords[1]) * (p1.coords[1] - p2.coords[1]) + (p1.coords[2] - p2.coords[2]) * (p1.coords[2] - p2.coords[2]);
}

__global__ void SparseSurfelFusion::device::computeSubdivideVertexImplicitFunctionValue(const VertexNode* SubdivideVertexArray, const EasyOctNode* SubdivideArray, DeviceArrayView<OctNode> NodeArray, DeviceArrayView<float> dx, DeviceArrayView<int> EncodedNodeIdxInFunction, DeviceArrayView<ConfirmedPPolynomial<CONVTIMES + 1, CONVTIMES + 2>> baseFunctions, const unsigned int NodeArraySize, const unsigned int rootId, const unsigned int SubdivideVertexArraySize, const float isoValue, float* SubdivideVvalue)
{
    const unsigned int idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (idx >= SubdivideVertexArraySize)	return;
    VertexNode nowVertex = SubdivideVertexArray[idx];
    int depth = device::maxDepth;
    float val = 0.0f;
    int nowNode = nowVertex.ownerNodeIdx;
    if (nowNode > 0) {
        while (nowNode != -1) {
            for (int k = 0; k < 27; ++k) {
                int neigh;
                if (nowNode < NodeArraySize)
                    neigh = NodeArray[nowNode].neighs[k];
                else
                    neigh = SubdivideArray[nowNode - NodeArraySize].neighs[k];
                if (neigh != -1) {
                    if (neigh == NodeArraySize)
                        neigh = rootId;
                    int idxO[3];
                    int encode_idx;
                    if (neigh < NodeArraySize)
                        encode_idx = EncodedNodeIdxInFunction[neigh];
                    else continue;  // d_x = 0 in Subdivide space
                    idxO[0] = encode_idx % decodeOffset_1;
                    idxO[1] = (encode_idx / decodeOffset_1) % decodeOffset_1;
                    idxO[2] = encode_idx / decodeOffset_2;

                    ConfirmedPPolynomial<CONVTIMES + 1, CONVTIMES + 2> funcX = baseFunctions[idxO[0]];
                    ConfirmedPPolynomial<CONVTIMES + 1, CONVTIMES + 2> funcY = baseFunctions[idxO[1]];
                    ConfirmedPPolynomial<CONVTIMES + 1, CONVTIMES + 2> funcZ = baseFunctions[idxO[2]];

                    val += dx[neigh] * value(funcX, nowVertex.pos.coords[0]) * value(funcY, nowVertex.pos.coords[1]) * value(funcZ, nowVertex.pos.coords[2]);
                }
            }
            if (nowNode < NodeArraySize)
                nowNode = NodeArray[nowNode].parent;
            else
                nowNode = SubdivideArray[nowNode - NodeArraySize].parent;
        }
    }
    SubdivideVvalue[idx] = val - isoValue;
}

__global__ void SparseSurfelFusion::device::computeSubdivideVertexImplicitFunctionValue(const VertexNode* SubdivideVertexArray, const EasyOctNode* SubdivideArray, DeviceArrayView<OctNode> NodeArray, DeviceArrayView<float> dx, DeviceArrayView<int> EncodedNodeIdxInFunction, DeviceArrayView<ConfirmedPPolynomial<CONVTIMES + 1, CONVTIMES + 2>> baseFunctions, const unsigned int NodeArraySize, const int* ReplacedNodeId, const int* IsRoot, const unsigned int SubdivideVertexArraySize, const float isoValue, float* SubdivideVvalue)
{
    const unsigned int idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (idx >= SubdivideVertexArraySize)	return;
    VertexNode nowVertex = SubdivideVertexArray[idx];
    float val = 0.0f;
    int nowNode = nowVertex.ownerNodeIdx;
    if (nowNode > 0) {
        while (nowNode != -1) {
            for (int k = 0; k < 27; ++k) {
                int neigh;
                if (nowNode < NodeArraySize)
                    neigh = NodeArray[nowNode].neighs[k];
                else
                    neigh = SubdivideArray[nowNode - NodeArraySize].neighs[k];
                if (neigh != -1) {
                    if (neigh >= NodeArraySize && IsRoot[neigh - NodeArraySize])
                        neigh = ReplacedNodeId[neigh - NodeArraySize];
                    int idxO[3];
                    int encode_idx;
                    if (neigh < NodeArraySize)
                        encode_idx = EncodedNodeIdxInFunction[neigh];
                    else continue;  // d_x = 0 in Subdivide space
                    idxO[0] = encode_idx % decodeOffset_1;
                    idxO[1] = (encode_idx / decodeOffset_1) % decodeOffset_1;
                    idxO[2] = encode_idx / decodeOffset_2;

                    ConfirmedPPolynomial<CONVTIMES + 1, CONVTIMES + 2> funcX = baseFunctions[idxO[0]];
                    ConfirmedPPolynomial<CONVTIMES + 1, CONVTIMES + 2> funcY = baseFunctions[idxO[1]];
                    ConfirmedPPolynomial<CONVTIMES + 1, CONVTIMES + 2> funcZ = baseFunctions[idxO[2]];

                    val += dx[neigh] * value(funcX, nowVertex.pos.coords[0]) * value(funcY, nowVertex.pos.coords[1]) * value(funcZ, nowVertex.pos.coords[2]);
                }
            }
            if (nowNode < NodeArraySize) nowNode = NodeArray[nowNode].parent;
            else nowNode = SubdivideArray[nowNode - NodeArraySize].parent;
        }
    }
    SubdivideVvalue[idx] = val - isoValue;
}

__global__ void SparseSurfelFusion::device::generateSubdivideVexNums(const EdgeNode* SubdivideEdgeArray, const EasyOctNode* SubdivideArray, const unsigned int SubdivideEdgeArraySize, const unsigned int NodeArraySize, const float* SubdivideVvalue, int* SubdivideVexNums, bool* markValidSubdivedeVexNum)
{
    const unsigned int idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (idx >= SubdivideEdgeArraySize)	return;
    int owner = SubdivideEdgeArray[idx].ownerNodeIdx - NodeArraySize;   // 当前边的Owner
    int kind = SubdivideEdgeArray[idx].edgeKind;                        // 当前边的类型
    int index[2];
    index[0] = edgeVertex[kind][0];
    index[1] = edgeVertex[kind][1];
    int v1 = SubdivideArray[owner].vertices[index[0]] - 1;
    int v2 = SubdivideArray[owner].vertices[index[1]] - 1;
    if (SubdivideVvalue[v1] * SubdivideVvalue[v2] <= 0) {
        SubdivideVexNums[idx] = 1;
        markValidSubdivedeVexNum[idx] = true;
    }
    else {
        markValidSubdivedeVexNum[idx] = false;
    }

}

__global__ void SparseSurfelFusion::device::generateTriNums(const EasyOctNode* SubdivideNodeArray, const unsigned int DLevelOffset, const unsigned int DLevelNodeCount, const float* vvalue, int* triNums, int* cubeCatagory)
{
    const unsigned int idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (idx >= DLevelNodeCount)	return;
    const unsigned int offset = DLevelOffset + idx;
    int currentCubeCatagory = 0;
    for (int i = 0; i < 8; i++) {
        if (vvalue[SubdivideNodeArray[offset].vertices[i] - 1] < 0) {
            currentCubeCatagory |= 1 << i;
        }
    }
    triNums[idx] = trianglesCount[currentCubeCatagory];
    cubeCatagory[idx] = currentCubeCatagory;
}

__global__ void SparseSurfelFusion::device::generateSubdivideIntersectionPoint(const EdgeNode* SubdivideValidEdgeArray, const VertexNode* SubdivideVertexArray, const EasyOctNode* SubdivideArray, const int* SubdivideValidVexAddress, const float* SubdivideVvalue, const unsigned int SubdivideValidEdgeArraySize, const unsigned int NodeArraySize, Point3D<float>* SubdivideVertexBuffer)
{
    const unsigned int idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (idx >= SubdivideValidEdgeArraySize)	return;
    int owner = SubdivideValidEdgeArray[idx].ownerNodeIdx - NodeArraySize;
    int kind = SubdivideValidEdgeArray[idx].edgeKind;
    int orientation = kind >> 2;
    int index[2];

    index[0] = edgeVertex[kind][0];
    index[1] = edgeVertex[kind][1];

    int v1 = SubdivideArray[owner].vertices[index[0]] - 1;
    int v2 = SubdivideArray[owner].vertices[index[1]] - 1;
    Point3D<float> p1 = SubdivideVertexArray[v1].pos, p2 = SubdivideVertexArray[v2].pos;
    float f1 = SubdivideVvalue[v1];
    float f2 = SubdivideVvalue[v2];
    Point3D<float> isoPoint;
    device::interpolatePoint(p1, p2, orientation, f1, f2, isoPoint);
    SubdivideVertexBuffer[SubdivideValidVexAddress[idx]] = isoPoint;
}

__global__ void SparseSurfelFusion::device::initFixedDepthNums(DeviceArrayView<OctNode> SubdivideNode, DeviceArrayView<int> SubdivideDepthBuffer, const unsigned int DepthOffset, const unsigned int DepthNodeCount, int* fixedDepthNums)
{
    const unsigned int idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (idx >= DepthNodeCount)	return;
    const unsigned int offset = DepthOffset + idx;
    int nodeNum = 1;
    for (int depth = SubdivideDepthBuffer[offset]; depth <= device::maxDepth; ++depth) {
        fixedDepthNums[(depth - 1) * DepthNodeCount + idx] = nodeNum;
        nodeNum <<= 3;  // 乘8
    }
}

__global__ void SparseSurfelFusion::device::wholeRebuildArray(DeviceArrayView<OctNode> SubdivideNode, const unsigned int finerDepthStart, const unsigned int finerSubdivideNum, const unsigned int NodeArraySize, const int* SubdivideDepthBuffer, const int* depthNodeAddress_Device, const int* fixedDepthAddress, EasyOctNode* RebuildArray, int* RebuildDepthBuffer, Point3D<float>* RebuildCenterBuffer, int* ReplaceNodeId, int* IsRoot, OctNode* NodeArray)
{
    const unsigned int idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (idx >= finerSubdivideNum)	return;
    int depthNodeAddress[MAX_DEPTH_OCTREE + 1];
    for (int i = 0; i <= maxDepth; i++) {                   // 获得数据更快：L1缓存
        depthNodeAddress[i] = depthNodeAddress_Device[i];   // depthNodeAddress_Device是Global内存，后续需要循环，时间开销大
    }
    const unsigned int offset = finerDepthStart + idx;
    int nowDepth = SubdivideDepthBuffer[offset];
    int fixedDepthOffset = fixedDepthAddress[(nowDepth - 1) * finerSubdivideNum + idx];
    int nowIdx = depthNodeAddress[nowDepth] + fixedDepthOffset;
    OctNode rootNode = SubdivideNode[offset];
    int replacedId = rootNode.neighs[13];
    rootNode.neighs[13] = NodeArraySize + nowIdx;
    RebuildArray[nowIdx] = rootNode;

    ReplaceNodeId[nowIdx] = replacedId;

    RebuildDepthBuffer[nowIdx] = nowDepth;

    IsRoot[nowIdx] = 1;
    Point3D<float> thisNodeCenter;
    getNodeCenterAllDepth(rootNode.key, nowDepth, thisNodeCenter);
    RebuildCenterBuffer[nowIdx] = thisNodeCenter;

    int sonKey = (rootNode.key >> (3 * (device::maxDepth - nowDepth))) & 7;
    NodeArray[rootNode.parent].children[sonKey] = NodeArraySize + nowIdx;
    int parentNodeIdx;
    int childrenNums = 8;
    while (nowDepth < device::maxDepth) {
        nowDepth++;
        fixedDepthOffset = fixedDepthAddress[(nowDepth - 1) * finerSubdivideNum + idx];
        nowIdx = depthNodeAddress[nowDepth] + fixedDepthOffset;
        for (int j = 0; j < childrenNums; j += 8) {
            int fatherFixedDepthOffset = fixedDepthAddress[(nowDepth - 2) * finerSubdivideNum + idx];
            parentNodeIdx = depthNodeAddress[nowDepth - 1] + fatherFixedDepthOffset + j / 8;
            int parentGlobalIdx = RebuildArray[parentNodeIdx].neighs[13];
            int parentKey = RebuildArray[parentNodeIdx].key;
            for (int k = 0; k < 8; k++) {
                int thisRoundIdx = nowIdx + j + k;
                int nowKey = parentKey | (k << (3 * (device::maxDepth - nowDepth)));
                RebuildArray[thisRoundIdx].parent = parentGlobalIdx;
                RebuildArray[thisRoundIdx].key = nowKey;
                RebuildArray[thisRoundIdx].neighs[13] = NodeArraySize + thisRoundIdx;

                ReplaceNodeId[thisRoundIdx] = replacedId;

                RebuildDepthBuffer[thisRoundIdx] = nowDepth;

                getNodeCenterAllDepth(nowKey, nowDepth, thisNodeCenter);
                RebuildCenterBuffer[thisRoundIdx] = thisNodeCenter;

                RebuildArray[parentNodeIdx].children[k] = NodeArraySize + thisRoundIdx;
            }
        }
        childrenNums <<= 3;
    }
}

__global__ void SparseSurfelFusion::device::markValidMeshVertexIndex(const Point3D<float>* VertexBuffer, const unsigned int verticesNum, bool* markValidVertices)
{
    const unsigned int idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (idx >= verticesNum) return;
    if (fabsf(VertexBuffer[idx].coords[0]) < device::eps) {
        printf("构建顶点发生错误！VertexBuffer[%d] = (%.5f, %.5f, %.5f)", idx, VertexBuffer[idx].coords[0], VertexBuffer[idx].coords[1], VertexBuffer[idx].coords[2]);
        markValidVertices[idx] = false;
    }
    else {
        markValidVertices[idx] = true;
    }
}

__global__ void SparseSurfelFusion::device::markValidMeshTriangleIndex(TriangleIndex* TriangleBuffer, const unsigned int previousVertexOffset, const unsigned int allTriNums, const unsigned int verticesNum, bool* markValidTriangleIndex)
{
    const unsigned int idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (idx >= allTriNums) return;
    bool triValid = true;
    TriangleIndex tri;
    for (int i = 0; i < 3; i++) {
        tri.idx[i] = TriangleBuffer[idx].idx[i] + previousVertexOffset;
        if (tri.idx[i] < 0 || tri.idx[i] >= verticesNum + previousVertexOffset) {
            triValid = false;
        }
    }

    TriangleBuffer[idx] = tri;
    if (triValid) {
        markValidTriangleIndex[idx] = true;
    }
    else {
        printf("三角索引构建错误！ index = %d   PreOffset = %d   verticesNum = %d   TriangleBuffer = (%d, %d, %d)\n", idx, previousVertexOffset, verticesNum, tri.idx[0], tri.idx[1], tri.idx[2]);
        markValidTriangleIndex[idx] = false;
    }
}



void SparseSurfelFusion::ComputeTriangleIndices::ComputeVertexImplicitFunctionValue(DeviceArrayView<VertexNode> VertexArray, DeviceArrayView<OctNode> NodeArray, DeviceArrayView<ConfirmedPPolynomial<CONVTIMES + 1, CONVTIMES + 2>> BaseFunction, DeviceArrayView<float> dx, DeviceArrayView<int> encodeNodeIndexInFunction, const float isoValue, cudaStream_t stream)
{
    const unsigned int VertexArraySize = VertexArray.Size();
    dim3 block(128);
    dim3 grid(divUp(VertexArraySize, block.x));

    device::ComputeVertexImplicitFunctionValueKernel << <grid, block, 0, stream >> > (VertexArray, NodeArray, BaseFunction, dx, encodeNodeIndexInFunction, VertexArraySize, isoValue, vvalue.Array().ptr());
}

void SparseSurfelFusion::ComputeTriangleIndices::insertTriangle(const Point3D<float>* VertexBufferHost, const int& allVexNums, const int* TriangleBufferHost, const int& allTriNums, CoredVectorMeshData& mesh)
{
    int previousVertex = mesh.inCorePoints.size();

    for (int i = 0; i < allVexNums; i++) {
        if (abs(VertexBufferHost[i].coords[0]) < EPSILON) {
            printf("error\n");
        }
        mesh.inCorePoints.push_back(VertexBufferHost[i]);
    }

    int inCoreFlag = 0; // 判断是第几个节点，三角形绘制有顺序
    for (int i = 0; i < 3; i++) {
        inCoreFlag |= CoredMeshData::IN_CORE_FLAG[i];
    }

    for (int i = 0; i < allTriNums; i++) {
        TriangleIndex tri;
        for (int j = 0; j < 3; j++) {
            tri.idx[j] = TriangleBufferHost[3 * i + j] + previousVertex;
            //if (i % 100 == 0) printf("depth = %d   idx = %d   TriangleBufferHost[%d] = %d\n" ,depth, i, 3 * i + j, TriangleBufferHost[3 * i + j]);
            if (tri.idx[j] < 0 || tri.idx[j] >= allVexNums + previousVertex) {
                printf("%d %d\n", tri.idx[j] - previousVertex, allVexNums);
                printf("tri error\n");
            }
        }
        mesh.addTriangle(tri, inCoreFlag);
    }
}

void SparseSurfelFusion::ComputeTriangleIndices::insertTriangle(Point3D<float>* VertexBuffer, const int allVexNums, TriangleIndex* TriangleBuffer, const int allTriNums, cudaStream_t stream)
{
    dim3 block_vex(128);
    dim3 grid_vex(divUp(allVexNums, block_vex.x));
    device::markValidMeshVertexIndex << <grid_vex, block_vex, 0, stream >> > (VertexBuffer, allVexNums, markValidTriangleVertex.Ptr());

    unsigned int* validVerticesCount = NULL;    // 有效的顶点
    unsigned int validVerticesCountHost = 0;
    CHECKCUDA(cudaMallocAsync(reinterpret_cast<void**>(&validVerticesCount), sizeof(unsigned int), stream));

    void* d_temp_storage_1 = NULL;    // 中间变量，用完即可释放
    size_t temp_storage_bytes_1 = 0;  // 中间变量
    CHECKCUDA(cub::DeviceSelect::Flagged(d_temp_storage_1, temp_storage_bytes_1, VertexBuffer, markValidTriangleVertex.Ptr(), MeshTriangleVertex.Ptr() + MeshTriangleVertex.ArraySize(), validVerticesCount, allVexNums, stream, false));	// 确定临时设备存储需求
    CHECKCUDA(cudaMallocAsync(&d_temp_storage_1, temp_storage_bytes_1, stream));
    CHECKCUDA(cub::DeviceSelect::Flagged(d_temp_storage_1, temp_storage_bytes_1, VertexBuffer, markValidTriangleVertex.Ptr(), MeshTriangleVertex.Ptr() + MeshTriangleVertex.ArraySize(), validVerticesCount, allVexNums, stream, false));	// 筛选	
    CHECKCUDA(cudaMemcpyAsync(&validVerticesCountHost, validVerticesCount, sizeof(unsigned int), cudaMemcpyDeviceToHost, stream));

    dim3 block_tri(128);
    dim3 grid_tri(divUp(allTriNums, block_tri.x));
    device::markValidMeshTriangleIndex << <grid_tri, block_tri, 0, stream >> > (TriangleBuffer, MeshTriangleVertex.ArraySize(), allTriNums, allVexNums, markValidTriangleIndex.Ptr());

    unsigned int* validTriangleIndicesCount = NULL;    // 有效的三角索引数组数量
    unsigned int validTriangleIndicesCountHost = 0;
    CHECKCUDA(cudaMallocAsync(reinterpret_cast<void**>(&validTriangleIndicesCount), sizeof(unsigned int), stream));

    void* d_temp_storage_2 = NULL;    // 中间变量，用完即可释放
    size_t temp_storage_bytes_2 = 0;  // 中间变量
    CHECKCUDA(cub::DeviceSelect::Flagged(d_temp_storage_2, temp_storage_bytes_2, TriangleBuffer, markValidTriangleIndex.Ptr(), MeshTriangleIndex.Ptr() + MeshTriangleIndex.ArraySize(), validTriangleIndicesCount, allTriNums, stream, false));	// 确定临时设备存储需求
    CHECKCUDA(cudaMallocAsync(&d_temp_storage_2, temp_storage_bytes_2, stream));
    CHECKCUDA(cub::DeviceSelect::Flagged(d_temp_storage_2, temp_storage_bytes_2, TriangleBuffer, markValidTriangleIndex.Ptr(), MeshTriangleIndex.Ptr() + MeshTriangleIndex.ArraySize(), validTriangleIndicesCount, allTriNums, stream, false));	// 筛选	
    CHECKCUDA(cudaMemcpyAsync(&validTriangleIndicesCountHost, validTriangleIndicesCount, sizeof(unsigned int), cudaMemcpyDeviceToHost, stream));

    CHECKCUDA(cudaStreamSynchronize(stream));
    MeshTriangleVertex.ResizeArrayOrException(validVerticesCountHost + MeshTriangleVertex.ArraySize());
    MeshTriangleIndex.ResizeArrayOrException(validTriangleIndicesCountHost + MeshTriangleIndex.ArraySize());
}

void SparseSurfelFusion::ComputeTriangleIndices::generateSubdivideNodeArrayCountAndAddress(DeviceBufferArray<OctNode>& NodeArray, DeviceArrayView<unsigned int> DepthBuffer, const unsigned int OtherDepthNodeCount, cudaStream_t stream)
{
    SubdivideNode.ResizeArrayOrException(OtherDepthNodeCount);
    CHECKCUDA(cudaMemsetAsync(SubdivideNode.Array().ptr(), 0, sizeof(OctNode) * OtherDepthNodeCount, stream));

    int* SubdivideNodeNum = NULL;
    CHECKCUDA(cudaMallocAsync(reinterpret_cast<void**>(&SubdivideNodeNum), sizeof(int), stream));

    ////// 这里无法使用cub::DeviceSelect::Flagged，调用API会导致共享内存溢出，主要是设置L1 Cache 和 Share Memory的比例
    //int* SubdivideNodeNumPtr = NULL;
    //CHECKCUDA(cudaMallocAsync(reinterpret_cast<void**>(&SubdivideNodeNumPtr), sizeof(int), stream));
    //void* d_temp_storage = NULL;
    //size_t temp_storage_bytes = 0; 
    //CHECKCUDA(cub::DeviceSelect::Flagged(d_temp_storage, temp_storage_bytes, NodeArray.Array().ptr(), markValidSubdividedNode.Array().ptr(), SubdivideNode.Array().ptr(), SubdivideNodeNumPtr, OtherDepthNodeCount, stream, false));	// 确定临时设备存储需求
    //CHECKCUDA(cudaMallocAsync(&d_temp_storage, temp_storage_bytes, stream));
    //CHECKCUDA(cub::DeviceSelect::Flagged(d_temp_storage, temp_storage_bytes, NodeArray.Array().ptr(), markValidSubdividedNode.Array().ptr(), SubdivideNode.Array().ptr(), SubdivideNodeNumPtr, OtherDepthNodeCount, stream, false));	// 筛选
    //CHECKCUDA(cudaMemcpyAsync(&SubdivideNodeNumHost, SubdivideNodeNumPtr, sizeof(int), cudaMemcpyDeviceToHost, stream));
    //CHECKCUDA(cudaStreamSynchronize(stream));

    // thrust::cuda::par.on(stream) -> 诠释Thrust库执行策略是以流的形式
    thrust::device_ptr<OctNode> NodeArray_ptr = thrust::device_pointer_cast<OctNode>(NodeArray.Array().ptr());
    thrust::device_ptr<OctNode> SubdivideNode_ptr = thrust::device_pointer_cast<OctNode>(SubdivideNode.Array().ptr());
    thrust::device_ptr<OctNode> SubdivideNode_end = thrust::copy_if(thrust::cuda::par.on(stream), NodeArray_ptr, NodeArray_ptr + OtherDepthNodeCount, SubdivideNode_ptr, ifSubdivide());
    CHECKCUDA(cudaStreamSynchronize(stream));
    SubdivideNodeNumHost = SubdivideNode_end - SubdivideNode_ptr;

    int* SubdivideDepthNum = NULL;
    CHECKCUDA(cudaMallocAsync(reinterpret_cast<void**>(&SubdivideDepthNum), sizeof(int) * SubdivideNodeNumHost * (Constants::maxDepth_Host + 1), stream));
    CHECKCUDA(cudaMemsetAsync(SubdivideDepthNum, 0, sizeof(int) * SubdivideNodeNumHost * (Constants::maxDepth_Host + 1), stream));

    SubdivideDepthBuffer.ResizeArrayOrException(SubdivideNodeNumHost);
    
    dim3 block(128);
    dim3 grid(divUp(SubdivideNodeNumHost, block.x));
    device::precomputeSubdivideDepth << <grid, block, 0, stream >> > (SubdivideNode.ArrayView(), DepthBuffer, SubdivideNodeNumHost, SubdivideDepthBuffer.DeviceArray().ptr(), SubdivideDepthNum);

    int* subdivideDepthCount = NULL;
    CHECKCUDA(cudaMallocAsync(reinterpret_cast<void**>(&subdivideDepthCount), sizeof(int), stream));


    for (int i = 0; i <= Constants::maxDepth_Host; i++) {
        void* d_temp_storage = NULL;
        size_t temp_storage_bytes = 0;
        cub::DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, SubdivideDepthNum + i * SubdivideNodeNumHost, subdivideDepthCount, SubdivideNodeNumHost, stream);
        CHECKCUDA(cudaMallocAsync(&d_temp_storage, temp_storage_bytes, stream));
        cub::DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, SubdivideDepthNum + i * SubdivideNodeNumHost, subdivideDepthCount, SubdivideNodeNumHost, stream);
        CHECKCUDA(cudaMemcpyAsync(&(SubdivideDepthCount[i]), subdivideDepthCount, sizeof(int), cudaMemcpyDeviceToHost, stream));
        CHECKCUDA(cudaFreeAsync(d_temp_storage, stream));
    }

    CHECKCUDA(cudaFreeAsync(SubdivideDepthNum, stream));    // 用完即刻释放，开销太大
    CHECKCUDA(cudaStreamSynchronize(stream));

    for (int i = 0; i <= Constants::maxDepth_Host; i++) {
        //printf("第 %d 层细分节点数量：%d   ", i, SubdivideDepthCount[i]);
        if (i == 0) SubdivideDepthAddress[i] = 0;
        else SubdivideDepthAddress[i] = SubdivideDepthAddress[i - 1] + SubdivideDepthCount[i - 1];
        //printf("节点偏移：%d\n", SubdivideDepthAddress[i]);
    }

}



void SparseSurfelFusion::ComputeTriangleIndices::generateVertexNumsAndVertexAddress(DeviceArrayView<EdgeNode> EdgeArray, DeviceArrayView<OctNode> NodeArray, DeviceArrayView<float> vvalue, const unsigned int DLevelOffset, cudaStream_t stream)
{
    const unsigned int EdgeArraySize = EdgeArray.Size();
    vexNums.ResizeArrayOrException(EdgeArraySize);
    vexAddress.ResizeArrayOrException(EdgeArraySize);
    markValidVertex.ResizeArrayOrException(EdgeArraySize);
    dim3 block(128);
    dim3 grid(divUp(EdgeArraySize, block.x));
    device::generateVertexNumsKernel << <grid, block, 0, stream >> > (EdgeArray, NodeArray, vvalue, EdgeArraySize, vexNums.Array().ptr(), markValidVertex.Array().ptr());

    void* tempStorage = NULL;	//【算法临时变量，用完即释放】排他前缀和的临时变量
    size_t tempStorageBytes = 0;
    cub::DeviceScan::ExclusiveSum(tempStorage, tempStorageBytes, vexNums.Array().ptr(), vexAddress.Array().ptr(), EdgeArraySize, stream);
    CHECKCUDA(cudaMallocAsync(&tempStorage, tempStorageBytes, stream));
    cub::DeviceScan::ExclusiveSum(tempStorage, tempStorageBytes, vexNums.Array().ptr(), vexAddress.Array().ptr(), EdgeArraySize, stream);

    CHECKCUDA(cudaFreeAsync(tempStorage, stream));
}

void SparseSurfelFusion::ComputeTriangleIndices::generateTriangleNumsAndTriangleAddress(DeviceArrayView<OctNode> NodeArray, DeviceArrayView<float> vvalue, const unsigned int DLevelOffset, const unsigned int DLevelNodeCount, cudaStream_t stream)
{
    triNums.ResizeArrayOrException(DLevelNodeCount);
    cubeCatagory.ResizeArrayOrException(DLevelNodeCount);
    triAddress.ResizeArrayOrException(DLevelNodeCount);
    dim3 block(128);
    dim3 grid(divUp(DLevelNodeCount, block.x));
    device::generateTriangleNumsKernel << <grid, block, 0, stream >> > (NodeArray, vvalue, DLevelOffset, DLevelNodeCount, triNums.Array().ptr(), cubeCatagory.Array().ptr());

    void* tempStorage = NULL;	//【算法临时变量，用完即释放】排他前缀和的临时变量
    size_t tempStorageBytes = 0;
    cub::DeviceScan::ExclusiveSum(tempStorage, tempStorageBytes, triNums.Array().ptr(), triAddress.Array().ptr(), DLevelNodeCount, stream);
    CHECKCUDA(cudaMallocAsync(&tempStorage, tempStorageBytes, stream));
    cub::DeviceScan::ExclusiveSum(tempStorage, tempStorageBytes, triNums.Array().ptr(), triAddress.Array().ptr(), DLevelNodeCount, stream);

    CHECKCUDA(cudaFreeAsync(tempStorage, stream));
}

void SparseSurfelFusion::ComputeTriangleIndices::generateVerticesAndTriangle(DeviceBufferArray<OctNode>& NodeArray, DeviceArrayView<VertexNode> VertexArray, DeviceArrayView<EdgeNode> EdgeArray, DeviceArrayView<FaceNode> FaceArray, const unsigned int DLevelOffset, const unsigned int DLevelNodeCount, cudaStream_t stream)
{
    const unsigned int EdgeArraySize = EdgeArray.Size();
    const unsigned int FaceArraySize = FaceArray.Size();

    int lastVexAddr;
    int lastVexNums;
    CHECKCUDA(cudaMemcpyAsync(&lastVexAddr, vexAddress.Array().ptr() + EdgeArraySize - 1, sizeof(int), cudaMemcpyDeviceToHost, stream));
    CHECKCUDA(cudaMemcpyAsync(&lastVexNums, vexNums.Array().ptr() + EdgeArraySize - 1, sizeof(int), cudaMemcpyDeviceToHost, stream));
    CHECKCUDA(cudaStreamSynchronize(stream));   // 这里需要统计一下所有顶点的数量

    int allVexNums = lastVexAddr + lastVexNums; // vertex的总数

    Point3D<float>* VertexBuffer = NULL;
    CHECKCUDA(cudaMallocAsync(reinterpret_cast<void**>(&VertexBuffer), sizeof(Point3D<float>) * allVexNums, stream));

    EdgeNode* validEdgeArray = NULL;
    CHECKCUDA(cudaMallocAsync(reinterpret_cast<void**>(&validEdgeArray), sizeof(EdgeNode) * allVexNums, stream));

    int* validVertexAddress = NULL;
    CHECKCUDA(cudaMallocAsync(reinterpret_cast<void**>(&validVertexAddress), sizeof(int) * allVexNums, stream));

    int* validEdgeArrayNum = NULL;
    CHECKCUDA(cudaMallocAsync(reinterpret_cast<void**>(&validEdgeArrayNum), sizeof(int), stream));

    int* validVertexAddressNum = NULL;
    CHECKCUDA(cudaMallocAsync(reinterpret_cast<void**>(&validVertexAddressNum), sizeof(int), stream));

    void* d_temp_storage_1 = NULL;
    size_t temp_storage_bytes_1 = 0;
    CHECKCUDA(cub::DeviceSelect::Flagged(d_temp_storage_1, temp_storage_bytes_1, EdgeArray.RawPtr(), markValidVertex.Array().ptr(), validEdgeArray, validEdgeArrayNum, EdgeArraySize, stream, false));	// 确定临时设备存储需求
    CHECKCUDA(cudaMallocAsync(&d_temp_storage_1, temp_storage_bytes_1, stream));
    CHECKCUDA(cub::DeviceSelect::Flagged(d_temp_storage_1, temp_storage_bytes_1, EdgeArray.RawPtr(), markValidVertex.Array().ptr(), validEdgeArray, validEdgeArrayNum, EdgeArraySize, stream, false));	// 筛选	

    void* d_temp_storage_2 = NULL;
    size_t temp_storage_bytes_2 = 0;
    CHECKCUDA(cub::DeviceSelect::Flagged(d_temp_storage_2, temp_storage_bytes_2, vexAddress.Array().ptr(), markValidVertex.Array().ptr(), validVertexAddress, validVertexAddressNum, EdgeArraySize, stream, false));
    CHECKCUDA(cudaMallocAsync(&d_temp_storage_2, temp_storage_bytes_2, stream));
    CHECKCUDA(cub::DeviceSelect::Flagged(d_temp_storage_2, temp_storage_bytes_2, vexAddress.Array().ptr(), markValidVertex.Array().ptr(), validVertexAddress, validVertexAddressNum, EdgeArraySize, stream, false));

    assert(allVexNums == validEdgeArrayNum);        // 合理性检查
    assert(allVexNums == validVertexAddressNum);    // 合理性检查

    dim3 block_1(128);
    dim3 grid_1(divUp(allVexNums, block_1.x));
    device::generateIntersectionPoint << <grid_1, block_1, 0, stream >> > (NodeArray.ArrayView(), VertexArray, vvalue.ArrayView(), validEdgeArray, validVertexAddress, allVexNums, VertexBuffer);

    int lastTriAddr;
    int lastTriNums;
    CHECKCUDA(cudaMemcpyAsync(&lastTriAddr, triAddress.Array().ptr() + DLevelNodeCount - 1, sizeof(int), cudaMemcpyDeviceToHost, stream));
    CHECKCUDA(cudaMemcpyAsync(&lastTriNums, triNums.Array().ptr() + DLevelNodeCount - 1, sizeof(int), cudaMemcpyDeviceToHost, stream));
    CHECKCUDA(cudaStreamSynchronize(stream));   // 这里需要统计一下所有顶点的数量
    int allTriNums = lastTriAddr + lastTriNums;

    TriangleIndex* TriangleBuffer = NULL;
    CHECKCUDA(cudaMallocAsync(reinterpret_cast<void**>(&TriangleBuffer), sizeof(TriangleIndex) * allTriNums, stream));

    int* hasSurfaceIntersection = NULL;
    CHECKCUDA(cudaMallocAsync(reinterpret_cast<void**>(&hasSurfaceIntersection), sizeof(int) * FaceArraySize, stream));
    CHECKCUDA(cudaMemsetAsync(hasSurfaceIntersection, 0, sizeof(int) * FaceArraySize, stream));

    dim3 block_2(128);
    dim3 grid_2(divUp(DLevelNodeCount, block_2.x));
    device::generateTrianglePos << <grid_2, block_2, 0, stream >> > (NodeArray.ArrayView(), FaceArray, triNums.ArrayView(), cubeCatagory.ArrayView(), vexAddress.ArrayView(), triAddress.ArrayView(), DLevelOffset, DLevelNodeCount, TriangleBuffer, hasSurfaceIntersection);
    CHECKCUDA(cudaStreamSynchronize(stream));   // 这里需要统计一下所有顶点的数量

    insertTriangle(VertexBuffer, allVexNums, TriangleBuffer, allTriNums, stream);

    markValidSubdividedNode.ResizeArrayOrException(DLevelOffset);

    dim3 block_3(128);
    dim3 grid_3(divUp(DLevelOffset, block_3.x));
    device::ProcessLeafNodesAtOtherDepth << <grid_3, block_3, 0, stream >> > (VertexArray, vvalue.ArrayView(), DLevelOffset, hasSurfaceIntersection, NodeArray.Array().ptr(), markValidSubdividedNode.Array().ptr());

    // 变量用完即释放
    CHECKCUDA(cudaFreeAsync(VertexBuffer, stream));
    CHECKCUDA(cudaFreeAsync(validEdgeArray, stream));
    CHECKCUDA(cudaFreeAsync(validVertexAddress, stream));
    CHECKCUDA(cudaFreeAsync(validEdgeArrayNum, stream));
    CHECKCUDA(cudaFreeAsync(validVertexAddressNum, stream));
    CHECKCUDA(cudaFreeAsync(d_temp_storage_1, stream));
    CHECKCUDA(cudaFreeAsync(d_temp_storage_2, stream));
    CHECKCUDA(cudaFreeAsync(TriangleBuffer, stream));
    CHECKCUDA(cudaFreeAsync(hasSurfaceIntersection, stream));

}


void SparseSurfelFusion::ComputeTriangleIndices::CoarserSubdivideNodeAndRebuildMesh(DeviceBufferArray<OctNode>& NodeArray, DeviceArrayView<unsigned int> DepthBuffer, DeviceArrayView<Point3D<float>> CenterBuffer, DeviceArrayView<ConfirmedPPolynomial<CONVTIMES + 1, CONVTIMES + 2>> BaseFunction, DeviceArrayView<float> dx, DeviceArrayView<int> encodeNodeIndexInFunction, const float isoValue, cudaStream_t stream)
{
    int minSubdivideRootDepth;
    SubdivideDepthBuffer.SynchronizeToHost(stream);
    std::vector<int>& SubdivideDepthBufferHost = SubdivideDepthBuffer.HostArray();
    std::vector<OctNode> SubdivideNodeHost;
    SubdivideNode.ArrayView().Download(SubdivideNodeHost);
    minSubdivideRootDepth = SubdivideDepthBufferHost[0];

    int maxNodeNums = (powf(8, (Constants::maxDepth_Host - minSubdivideRootDepth + 1)) - 1) / 7;

    EasyOctNode* SubdivideArray = NULL;
    CHECKCUDA(cudaMallocAsync(reinterpret_cast<void**>(&SubdivideArray), sizeof(EasyOctNode) * maxNodeNums, stream));

    int* SubdivideArrayDepthBuffer = NULL;
    CHECKCUDA(cudaMallocAsync(reinterpret_cast<void**>(&SubdivideArrayDepthBuffer), sizeof(int) * maxNodeNums, stream));

    Point3D<float>* SubdivideArrayCenterBuffer = NULL;
    CHECKCUDA(cudaMallocAsync(reinterpret_cast<void**>(&SubdivideArrayCenterBuffer), sizeof(Point3D<float>) * maxNodeNums, stream));

    for (int i = 0; i < SubdivideNodeNumHost; i++) {
        int rootDepth = SubdivideDepthBufferHost[i];
        if (rootDepth >= finerDepth) break;

        int SubdivideArraySize = (powf(8, (Constants::maxDepth_Host - rootDepth + 1)) - 1) / 7;
        int currentNodeNum = 1;
        for (int j = rootDepth; j <= Constants::maxDepth_Host; j++) {
            fixedDepthNodeNum[j] = currentNodeNum;
            currentNodeNum <<= 3;       // 乘8
        }

        //for (int j = 0; j <= Constants::maxDepth_Host; j++) {
        //    printf("fixedDepthNodeNum[%d] = %d\n", j, fixedDepthNodeNum[j]);
        //}

        //printf("SubdivideArraySize - (D-1) = %d   fixedDepthNodeNum = %d\n", SubdivideArraySize - fixedDepthNodeNum[Constants::maxDepth_Host - 1], fixedDepthNodeNum[Constants::maxDepth_Host]);

        for (int j = rootDepth; j <= Constants::maxDepth_Host; j++) {
            fixedDepthNodeAddress[j] = fixedDepthNodeAddress[j - 1] + fixedDepthNodeNum[j - 1];
        }
        OctNode rootNode = SubdivideNodeHost[i];
        int rootIndex = rootNode.neighs[13];
        int rootParent = rootNode.parent;
        int rootKey = rootNode.key;
        int rootSonKey = (rootKey >> (3 * (Constants::maxDepth_Host - rootDepth))) & 7;

        CHECKCUDA(cudaMemsetAsync(SubdivideArray, 0, sizeof(EasyOctNode) * SubdivideArraySize, stream));

        int NodeArraySize = NodeArray.ArraySize();
        OctNode* NodeArrayPtr = NodeArray.Array().ptr();
        CHECKCUDA(cudaMemcpyAsync(&NodeArrayPtr[rootParent].children[rootSonKey], &NodeArraySize, sizeof(int), cudaMemcpyHostToDevice, stream));
        CHECKCUDA(cudaMemcpyAsync(&SubdivideArray[0].parent, &rootParent, sizeof(int), cudaMemcpyHostToDevice, stream));

        dim3 block_1(128);
        dim3 grid_1(divUp(SubdivideArraySize, block_1.x));
        device::singleRebuildArray << <grid_1, block_1, 0, stream >> > (SubdivideNode.ArrayView(), SubdivideDepthBuffer.DeviceArrayReadOnly(), i, NodeArraySize, SubdivideArraySize, SubdivideArray, SubdivideArrayDepthBuffer, SubdivideArrayCenterBuffer);

        for (int depth = rootDepth; depth <= Constants::maxDepth_Host; depth++) {
            dim3 block_2(128);
            dim3 grid_2(divUp(fixedDepthNodeNum[depth], block_2.x));
            device::computeRebuildNeighbor << <grid_2, block_2, 0, stream >> > (NodeArray.ArrayView(), fixedDepthNodeAddress[depth], fixedDepthNodeNum[depth], NodeArraySize, depth, SubdivideArray);
        }

        /**************************************** SubdivideVertexArray ****************************************/

        VertexNode* SubdividePreVertexArray = NULL;     // 即时变量，用完即删
        CHECKCUDA(cudaMallocAsync(reinterpret_cast<void**>(&SubdividePreVertexArray), sizeof(VertexNode) * 8 * fixedDepthNodeNum[Constants::maxDepth_Host], stream));
        CHECKCUDA(cudaMemsetAsync(SubdividePreVertexArray, 0, sizeof(VertexNode) * 8 * fixedDepthNodeNum[Constants::maxDepth_Host], stream));

        markValidSubdivideVertex.ResizeArrayOrException(fixedDepthNodeNum[Constants::maxDepth_Host]);

        dim3 block_3(128);
        dim3 grid_3(divUp(fixedDepthNodeNum[Constants::maxDepth_Host], block_3.x));
        device::initSubdivideVertexOwner << <grid_3, block_3, 0, stream >> > (SubdivideArray, SubdivideArrayCenterBuffer, fixedDepthNodeAddress[Constants::maxDepth_Host], fixedDepthNodeNum[Constants::maxDepth_Host], NodeArraySize, SubdividePreVertexArray, markValidSubdivideVertex.Array().ptr());
    
        VertexNode* SubdivideVertexArray = NULL;
        CHECKCUDA(cudaMallocAsync(reinterpret_cast<void**>(&SubdivideVertexArray), sizeof(VertexNode) * 8 * fixedDepthNodeNum[Constants::maxDepth_Host], stream));
        CHECKCUDA(cudaMemsetAsync(SubdivideVertexArray, 0, sizeof(VertexNode) * 8 * fixedDepthNodeNum[Constants::maxDepth_Host], stream));
    
        int* SubdivideVertexArraySize = NULL;
        int SubdivideVertexArraySizeHost = -1;
        CHECKCUDA(cudaMallocAsync(reinterpret_cast<void**>(&SubdivideVertexArraySize), sizeof(int), stream));

        void* d_temp_storage_1 = NULL;    // 中间变量，用完即可释放
        size_t temp_storage_bytes_1 = 0;  // 中间变量
        CHECKCUDA(cub::DeviceSelect::Flagged(d_temp_storage_1, temp_storage_bytes_1, SubdividePreVertexArray, markValidSubdivideVertex.Array().ptr(), SubdivideVertexArray, SubdivideVertexArraySize, 8 * fixedDepthNodeNum[Constants::maxDepth_Host], stream, false));	// 确定临时设备存储需求
        CHECKCUDA(cudaMallocAsync(&d_temp_storage_1, temp_storage_bytes_1, stream));
        CHECKCUDA(cub::DeviceSelect::Flagged(d_temp_storage_1, temp_storage_bytes_1, SubdividePreVertexArray, markValidSubdivideVertex.Array().ptr(), SubdivideVertexArray, SubdivideVertexArraySize, 8 * fixedDepthNodeNum[Constants::maxDepth_Host], stream, false));	// 筛选	
        CHECKCUDA(cudaMemcpyAsync(&SubdivideVertexArraySizeHost, SubdivideVertexArraySize, sizeof(int), cudaMemcpyDeviceToHost, stream));

        CHECKCUDA(cudaFreeAsync(SubdividePreVertexArray, stream));  // 即时变量，用完即删
        CHECKCUDA(cudaFreeAsync(SubdivideVertexArraySize, stream));
        CHECKCUDA(cudaFreeAsync(d_temp_storage_1, stream));

        CHECKCUDA(cudaStreamSynchronize(stream));       // 同步流，获得SubdivideVertexArraySizeHost

        //printf("SubdivideVertexArraySizeHost = %d\n", SubdivideVertexArraySizeHost);


        dim3 block_4(128);
        dim3 grid_4(divUp(SubdivideVertexArraySizeHost, block_4.x));
        device::maintainSubdivideVertexNodePointer << <grid_4, block_4, 0, stream >> > (CenterBuffer, SubdivideVertexArraySizeHost, NodeArraySize, SubdivideArrayCenterBuffer, SubdivideVertexArray, SubdivideArray);

        /**************************************** SubdivideEdgeArray ****************************************/

        EdgeNode* SubdividePreEdgeArray = NULL;
        CHECKCUDA(cudaMallocAsync(reinterpret_cast<void**>(&SubdividePreEdgeArray), sizeof(EdgeNode) * 12 * fixedDepthNodeNum[Constants::maxDepth_Host], stream));
        CHECKCUDA(cudaMemsetAsync(SubdividePreEdgeArray, 0, sizeof(EdgeNode) * 12 * fixedDepthNodeNum[Constants::maxDepth_Host], stream));

        markValidSubdivideEdge.ResizeArrayOrException(fixedDepthNodeNum[Constants::maxDepth_Host]);

        dim3 block_5(128);
        dim3 grid_5(divUp(fixedDepthNodeNum[Constants::maxDepth_Host], block_5.x));
        device::initSubdivideEdgeArray << <grid_5, block_5, 0, stream >> > (SubdivideArray, SubdivideArrayCenterBuffer, NodeArraySize, fixedDepthNodeAddress[Constants::maxDepth_Host], fixedDepthNodeNum[Constants::maxDepth_Host], SubdividePreEdgeArray, markValidSubdivideEdge.Array().ptr());

        EdgeNode* SubdivideEdgeArray = NULL;
        CHECKCUDA(cudaMallocAsync(reinterpret_cast<void**>(&SubdivideEdgeArray), sizeof(EdgeNode) * 12 * fixedDepthNodeNum[Constants::maxDepth_Host], stream));
        CHECKCUDA(cudaMemsetAsync(SubdivideEdgeArray, 0, sizeof(EdgeNode) * 12 * fixedDepthNodeNum[Constants::maxDepth_Host], stream));

        int* SubdivideEdgeArraySize = NULL;
        int SubdivideEdgeArraySizeHost = -1;
        CHECKCUDA(cudaMallocAsync(reinterpret_cast<void**>(&SubdivideEdgeArraySize), sizeof(int), stream));

        void* d_temp_storage_2 = NULL;    // 中间变量，用完即可释放
        size_t temp_storage_bytes_2 = 0;  // 中间变量
        CHECKCUDA(cub::DeviceSelect::Flagged(d_temp_storage_2, temp_storage_bytes_2, SubdividePreEdgeArray, markValidSubdivideEdge.Array().ptr(), SubdivideEdgeArray, SubdivideEdgeArraySize, 12 * fixedDepthNodeNum[Constants::maxDepth_Host], stream, false));	// 确定临时设备存储需求
        CHECKCUDA(cudaMallocAsync(&d_temp_storage_2, temp_storage_bytes_2, stream));
        CHECKCUDA(cub::DeviceSelect::Flagged(d_temp_storage_2, temp_storage_bytes_2, SubdividePreEdgeArray, markValidSubdivideEdge.Array().ptr(), SubdivideEdgeArray, SubdivideEdgeArraySize, 12 * fixedDepthNodeNum[Constants::maxDepth_Host], stream, false));	// 筛选	
        CHECKCUDA(cudaMemcpyAsync(&SubdivideEdgeArraySizeHost, SubdivideEdgeArraySize, sizeof(int), cudaMemcpyDeviceToHost, stream));

        CHECKCUDA(cudaFreeAsync(SubdividePreEdgeArray, stream));  // 即时变量，用完即删
        CHECKCUDA(cudaFreeAsync(SubdivideEdgeArraySize, stream));
        CHECKCUDA(cudaFreeAsync(d_temp_storage_2, stream));

        CHECKCUDA(cudaStreamSynchronize(stream));       // 同步流，获得SubdivideEdgeArraySizeHost

        //printf("SubdivideEdgeArraySizeHost = %d\n", SubdivideEdgeArraySizeHost);


        dim3 block_6(128);
        dim3 grid_6(divUp(SubdivideEdgeArraySizeHost, block_6.x));
        device::maintainSubdivideEdgeNodePointer << <grid_6, block_6, 0, stream >> > (CenterBuffer, SubdivideArrayCenterBuffer, SubdivideEdgeArraySizeHost, NodeArraySize, SubdivideArray, SubdivideEdgeArray);

        /**************************************** 计算细分节点隐式函数的值, 生成细分顶点的vexNums和vexAddress ****************************************/
        float* SubdivideVvalue = NULL;
        CHECKCUDA(cudaMallocAsync(reinterpret_cast<void**>(&SubdivideVvalue), sizeof(float) * SubdivideVertexArraySizeHost, stream));
        CHECKCUDA(cudaMemsetAsync(SubdivideVvalue, 0, sizeof(float) * SubdivideVertexArraySizeHost, stream));
        dim3 block_7(128);
        dim3 grid_7(divUp(SubdivideVertexArraySizeHost, block_7.x));
        device::computeSubdivideVertexImplicitFunctionValue << <grid_7, block_7, 0, stream >> > (SubdivideVertexArray, SubdivideArray, NodeArray.ArrayView(), dx, encodeNodeIndexInFunction, BaseFunction, NodeArraySize, rootIndex, SubdivideVertexArraySizeHost, isoValue, SubdivideVvalue);

        int* SubdivideVexNums = NULL;
        CHECKCUDA(cudaMallocAsync(reinterpret_cast<void**>(&SubdivideVexNums), sizeof(int) * SubdivideEdgeArraySizeHost, stream));
        CHECKCUDA(cudaMemsetAsync(SubdivideVexNums, 0, sizeof(int) * SubdivideEdgeArraySizeHost, stream));

        markValidSubdivedeVexNum.ResizeArrayOrException(SubdivideEdgeArraySizeHost);
        
        dim3 block_8(128);
        dim3 grid_8(divUp(SubdivideEdgeArraySizeHost, block_8.x));
        device::generateSubdivideVexNums << <grid_8, block_8, 0, stream >> > (SubdivideEdgeArray, SubdivideArray, SubdivideEdgeArraySizeHost, NodeArraySize, SubdivideVvalue, SubdivideVexNums, markValidSubdivedeVexNum.Array().ptr());
        
        
        //printf("SubdivideEdgeArraySizeHost = %d\n", SubdivideEdgeArraySizeHost);


        int* SubdivideVexAddress = NULL;
        CHECKCUDA(cudaMallocAsync(reinterpret_cast<void**>(&SubdivideVexAddress), sizeof(int) * SubdivideEdgeArraySizeHost, stream));
        CHECKCUDA(cudaMemsetAsync(SubdivideVexAddress, 0, sizeof(int) * SubdivideEdgeArraySizeHost, stream));

        void* tempVexAddressStorage = NULL;	//【算法临时变量，用完即释放】排他前缀和的临时变量
        size_t tempVexAddressStorageBytes = 0;
        cub::DeviceScan::ExclusiveSum(tempVexAddressStorage, tempVexAddressStorageBytes, SubdivideVexNums, SubdivideVexAddress, SubdivideEdgeArraySizeHost, stream);
        CHECKCUDA(cudaMallocAsync(&tempVexAddressStorage, tempVexAddressStorageBytes, stream));
        cub::DeviceScan::ExclusiveSum(tempVexAddressStorage, tempVexAddressStorageBytes, SubdivideVexNums, SubdivideVexAddress, SubdivideEdgeArraySizeHost, stream);

        CHECKCUDA(cudaFreeAsync(tempVexAddressStorage, stream));

        int SubdivideLastVexAddr = -1;
        int SubdivideLastVexNums = -1;
        CHECKCUDA(cudaMemcpyAsync(&SubdivideLastVexAddr, SubdivideVexAddress + SubdivideEdgeArraySizeHost - 1, sizeof(int), cudaMemcpyDeviceToHost, stream));
        CHECKCUDA(cudaMemcpyAsync(&SubdivideLastVexNums, SubdivideVexNums + SubdivideEdgeArraySizeHost - 1, sizeof(int), cudaMemcpyDeviceToHost, stream));


        CHECKCUDA(cudaStreamSynchronize(stream));   // 流同步

        int SubdivideAllVexNums = SubdivideLastVexAddr + SubdivideLastVexNums;
        //printf("SubdivideAllVexNums = %d\n", SubdivideAllVexNums);

        if (SubdivideAllVexNums == 0) {
            CHECKCUDA(cudaMemcpyAsync(&(NodeArray[rootParent].children[rootSonKey]), &rootIndex, sizeof(int), cudaMemcpyHostToDevice, stream));
            CHECKCUDA(cudaFreeAsync(SubdivideVertexArray, stream));
            CHECKCUDA(cudaFreeAsync(SubdivideEdgeArray, stream));
            CHECKCUDA(cudaFreeAsync(SubdivideVvalue, stream));
            CHECKCUDA(cudaFreeAsync(SubdivideVexNums, stream));
            CHECKCUDA(cudaFreeAsync(SubdivideVexAddress, stream));
            continue;
        }

        /**************************************** 计算细分顶点的三角形和立方体类型 ****************************************/

        int* SubdivideTriNums = NULL;
        CHECKCUDA(cudaMallocAsync(reinterpret_cast<void**>(&SubdivideTriNums), sizeof(int) * fixedDepthNodeNum[Constants::maxDepth_Host], stream));
        CHECKCUDA(cudaMemsetAsync(SubdivideTriNums, 0, sizeof(int) * fixedDepthNodeNum[Constants::maxDepth_Host], stream));

        int* SubdivideCubeCatagory = NULL;
        CHECKCUDA(cudaMallocAsync(reinterpret_cast<void**>(&SubdivideCubeCatagory), sizeof(int) * fixedDepthNodeNum[Constants::maxDepth_Host], stream));
        CHECKCUDA(cudaMemsetAsync(SubdivideCubeCatagory, 0, sizeof(int) * fixedDepthNodeNum[Constants::maxDepth_Host], stream));

        dim3 block_9(128);
        dim3 grid_9(divUp(fixedDepthNodeNum[Constants::maxDepth_Host], block_9.x));
        device::generateTriNums << <grid_9, block_9, 0, stream >> > (SubdivideArray, fixedDepthNodeAddress[Constants::maxDepth_Host], fixedDepthNodeNum[Constants::maxDepth_Host], SubdivideVvalue, SubdivideTriNums, SubdivideCubeCatagory);

        int* SubdivideTriAddress = NULL;
        CHECKCUDA(cudaMallocAsync(reinterpret_cast<void**>(&SubdivideTriAddress), sizeof(int) * fixedDepthNodeNum[Constants::maxDepth_Host], stream));
        CHECKCUDA(cudaMemsetAsync(SubdivideTriAddress, 0, sizeof(int) * fixedDepthNodeNum[Constants::maxDepth_Host], stream));

        void* tempTriAddressStorage = NULL;	//【算法临时变量，用完即释放】排他前缀和的临时变量
        size_t tempTriAddressStorageBytes = 0;
        cub::DeviceScan::ExclusiveSum(tempTriAddressStorage, tempTriAddressStorageBytes, SubdivideTriNums, SubdivideTriAddress, fixedDepthNodeNum[Constants::maxDepth_Host], stream);
        CHECKCUDA(cudaMallocAsync(&tempTriAddressStorage, tempTriAddressStorageBytes, stream));
        cub::DeviceScan::ExclusiveSum(tempTriAddressStorage, tempTriAddressStorageBytes, SubdivideTriNums, SubdivideTriAddress, fixedDepthNodeNum[Constants::maxDepth_Host], stream);

        CHECKCUDA(cudaFreeAsync(tempTriAddressStorage, stream));

        Point3D<float>* SubdivideVertexBuffer = NULL;
        CHECKCUDA(cudaMallocAsync(reinterpret_cast<void**>(&SubdivideVertexBuffer), sizeof(Point3D<float>) * SubdivideAllVexNums, stream));
        //std::vector<Point3D<float>> SubdivideVertexBufferHost;
        //SubdivideVertexBufferHost.resize(SubdivideAllVexNums);

        EdgeNode* SubdivideValidEdgeArray = NULL;
        CHECKCUDA(cudaMallocAsync(reinterpret_cast<void**>(&SubdivideValidEdgeArray), sizeof(EdgeNode) * SubdivideAllVexNums, stream));

        int* SubdivideValidEdgeArraySize = NULL;    // 有效的细分边device
        int SubdivideValidEdgeArraySizeHost = -1;   // 有效的细分边Host
        CHECKCUDA(cudaMallocAsync(reinterpret_cast<void**>(&SubdivideValidEdgeArraySize), sizeof(int), stream));

        void* d_temp_storage_3 = NULL;    // 中间变量，用完即可释放
        size_t temp_storage_bytes_3 = 0;  // 中间变量
        CHECKCUDA(cub::DeviceSelect::Flagged(d_temp_storage_3, temp_storage_bytes_3, SubdivideEdgeArray, markValidSubdivedeVexNum.Array().ptr(), SubdivideValidEdgeArray, SubdivideValidEdgeArraySize, SubdivideEdgeArraySizeHost, stream, false));	// 确定临时设备存储需求
        CHECKCUDA(cudaMallocAsync(&d_temp_storage_3, temp_storage_bytes_3, stream));
        CHECKCUDA(cub::DeviceSelect::Flagged(d_temp_storage_3, temp_storage_bytes_3, SubdivideEdgeArray, markValidSubdivedeVexNum.Array().ptr(), SubdivideValidEdgeArray, SubdivideValidEdgeArraySize, SubdivideEdgeArraySizeHost, stream, false));	// 筛选	
        CHECKCUDA(cudaMemcpyAsync(&SubdivideValidEdgeArraySizeHost, SubdivideValidEdgeArraySize, sizeof(int), cudaMemcpyDeviceToHost, stream));

        int* SubdivideValidVexAddress = NULL;    // 有效的细分边device
        CHECKCUDA(cudaMallocAsync(reinterpret_cast<void**>(&SubdivideValidVexAddress), sizeof(int) * SubdivideAllVexNums, stream));

        int* SubdivideValidVexAddressSize = NULL;    // 有效的细分边device
        int SubdivideValidVexAddressSizeHost = -1;   // 有效的细分边Host
        CHECKCUDA(cudaMallocAsync(reinterpret_cast<void**>(&SubdivideValidVexAddressSize), sizeof(int), stream));

        void* d_temp_storage_4 = NULL;    // 中间变量，用完即可释放
        size_t temp_storage_bytes_4 = 0;  // 中间变量
        CHECKCUDA(cub::DeviceSelect::Flagged(d_temp_storage_4, temp_storage_bytes_4, SubdivideVexAddress, markValidSubdivedeVexNum.Array().ptr(), SubdivideValidVexAddress, SubdivideValidVexAddressSize, SubdivideEdgeArraySizeHost, stream, false));	// 确定临时设备存储需求
        CHECKCUDA(cudaMallocAsync(&d_temp_storage_4, temp_storage_bytes_4, stream));
        CHECKCUDA(cub::DeviceSelect::Flagged(d_temp_storage_4, temp_storage_bytes_4, SubdivideVexAddress, markValidSubdivedeVexNum.Array().ptr(), SubdivideValidVexAddress, SubdivideValidVexAddressSize, SubdivideEdgeArraySizeHost, stream, false));	// 筛选	
        CHECKCUDA(cudaMemcpyAsync(&SubdivideValidVexAddressSizeHost, SubdivideValidVexAddressSize, sizeof(int), cudaMemcpyDeviceToHost, stream));

        dim3 block_10(128);
        dim3 grid_10(divUp(SubdivideAllVexNums, block_10.x));
        device::generateSubdivideIntersectionPoint << <grid_10, block_10, 0, stream >> > (SubdivideValidEdgeArray, SubdivideVertexArray, SubdivideArray, SubdivideValidVexAddress, SubdivideVvalue, SubdivideValidEdgeArraySizeHost, NodeArraySize, SubdivideVertexBuffer);
        //CHECKCUDA(cudaMemcpyAsync(SubdivideVertexBufferHost.data(), SubdivideVertexBuffer, sizeof(Point3D<float>) * SubdivideAllVexNums, cudaMemcpyDeviceToHost, stream));

        CHECKCUDA(cudaFreeAsync(SubdivideValidEdgeArray, stream));
        CHECKCUDA(cudaFreeAsync(SubdivideValidVexAddress, stream));
        CHECKCUDA(cudaFreeAsync(d_temp_storage_3, stream));
        CHECKCUDA(cudaFreeAsync(d_temp_storage_4, stream));

        //CHECKCUDA(cudaStreamSynchronize(stream));
        //printf("depth = %d   SubdivideValidVexAddressSize = %d\n", i, SubdivideValidVexAddressSizeHost);

        int SubdivideLastTriAddr;
        int SubdivideLastTriNums;
        CHECKCUDA(cudaMemcpyAsync(&SubdivideLastTriAddr, SubdivideTriAddress + fixedDepthNodeNum[Constants::maxDepth_Host] - 1, sizeof(int), cudaMemcpyDeviceToHost));
        CHECKCUDA(cudaMemcpyAsync(&SubdivideLastTriNums, SubdivideTriNums + fixedDepthNodeNum[Constants::maxDepth_Host] - 1, sizeof(int), cudaMemcpyDeviceToHost));
        CHECKCUDA(cudaStreamSynchronize(stream));   // 流同步
        int SubdivideAllTriNums = SubdivideLastTriAddr + SubdivideLastTriNums;
        //printf("depth = %d   SubdivideAllTriNums = %d\n", i, SubdivideAllTriNums);

        TriangleIndex* SubdivideTriangleBuffer = NULL;
        CHECKCUDA(cudaMallocAsync(reinterpret_cast<void**>(&SubdivideTriangleBuffer), sizeof(TriangleIndex) * SubdivideAllTriNums, stream));
        //CHECKCUDA(cudaMemsetAsync(SubdivideTriangleBuffer, 0, sizeof(TriangleIndex) * SubdivideAllTriNums, stream));

        //std::vector<int> SubdivideTriangleBufferHost;
        //SubdivideTriangleBufferHost.resize(3 * SubdivideAllTriNums);

        dim3 block_11(128);
        dim3 grid_11(divUp(fixedDepthNodeNum[Constants::maxDepth_Host], block_11.x));
        device::generateSubdivideTrianglePos << <grid_11, block_11, 0, stream >> > (SubdivideArray, fixedDepthNodeAddress[Constants::maxDepth_Host], fixedDepthNodeNum[Constants::maxDepth_Host], SubdivideTriNums, SubdivideCubeCatagory, SubdivideVexAddress, SubdivideTriAddress, SubdivideTriangleBuffer);



        insertTriangle(SubdivideVertexBuffer, SubdivideAllVexNums, SubdivideTriangleBuffer, SubdivideAllTriNums, stream);
        //CHECKCUDA(cudaMemcpyAsync(SubdivideTriangleBufferHost.data(), SubdivideTriangleBuffer, sizeof(int) * 3 * SubdivideAllTriNums, cudaMemcpyDeviceToHost, stream));

        CHECKCUDA(cudaStreamSynchronize(stream));   // 流同步
        


        //insertTriangle(SubdivideVertexBufferHost.data(), SubdivideAllVexNums, SubdivideTriangleBufferHost.data(), SubdivideAllTriNums, mesh);

        CHECKCUDA(cudaMemcpy(&(NodeArray[rootParent].children[rootSonKey]), &rootIndex, sizeof(int), cudaMemcpyHostToDevice));
        CHECKCUDA(cudaFreeAsync(SubdivideVertexArray, stream));
        CHECKCUDA(cudaFreeAsync(SubdivideEdgeArray, stream));
        CHECKCUDA(cudaFreeAsync(SubdivideVvalue, stream));
        CHECKCUDA(cudaFreeAsync(SubdivideVexNums, stream));
        CHECKCUDA(cudaFreeAsync(SubdivideVexAddress, stream));
        CHECKCUDA(cudaFreeAsync(SubdivideTriNums, stream));
        CHECKCUDA(cudaFreeAsync(SubdivideCubeCatagory, stream));
        CHECKCUDA(cudaFreeAsync(SubdivideTriAddress, stream));
        CHECKCUDA(cudaFreeAsync(SubdivideVertexBuffer, stream));
        CHECKCUDA(cudaFreeAsync(SubdivideTriangleBuffer, stream));
    }
    CHECKCUDA(cudaFreeAsync(SubdivideArray, stream));
    CHECKCUDA(cudaFreeAsync(SubdivideArrayCenterBuffer, stream));
    CHECKCUDA(cudaFreeAsync(SubdivideArrayDepthBuffer, stream));
}

void SparseSurfelFusion::ComputeTriangleIndices::FinerSubdivideNodeAndRebuildMesh(DeviceBufferArray<OctNode>& NodeArray, DeviceArrayView<unsigned int> DepthBuffer, DeviceArrayView<Point3D<float>> CenterBuffer, DeviceArrayView<ConfirmedPPolynomial<CONVTIMES + 1, CONVTIMES + 2>> BaseFunction, DeviceArrayView<float> dx, DeviceArrayView<int> encodeNodeIndexInFunction, const float isoValue, cudaStream_t stream)
{
    const unsigned int NodeArraySize = NodeArray.ArraySize();
    for (int i = finerDepth; i < Constants::maxDepth_Host; i++) {
        int finerDepthStart = SubdivideDepthAddress[i];
        int finerSubdivideNum = SubdivideDepthCount[i];
        int* fixedDepthNums = NULL;
        CHECKCUDA(cudaMallocAsync(reinterpret_cast<void**>(&fixedDepthNums), sizeof(int) * finerSubdivideNum * Constants::maxDepth_Host, stream));
        CHECKCUDA(cudaMemsetAsync(fixedDepthNums, 0, sizeof(int) * finerSubdivideNum * Constants::maxDepth_Host, stream));

        dim3 block_1(128);
        dim3 grid_1(divUp(finerSubdivideNum, block_1.x));
        device::initFixedDepthNums << <grid_1, block_1, 0, stream >> > (SubdivideNode.ArrayView(), SubdivideDepthBuffer.DeviceArrayReadOnly(), finerDepthStart, finerSubdivideNum, fixedDepthNums);

        //CHECKCUDA(cudaStreamSynchronize(stream));   // 流同步
        //std::vector<int> fixedDepthNumsHost;
        //fixedDepthNumsHost.resize(finerSubdivideNum * Constants::maxDepth_Host);
        //CHECKCUDA(cudaMemcpyAsync(fixedDepthNumsHost.data(), fixedDepthNums, sizeof(int) * finerSubdivideNum * Constants::maxDepth_Host, cudaMemcpyDeviceToHost, stream));
        //for (int j = 0; j < fixedDepthNumsHost.size(); j++) {
        //    if (j % 1000 == 0) printf("depth = %d   fixedDepthNumsHost[%d] = %d\n", i, j, fixedDepthNumsHost[j]);
        //}        


        int* rebuildNumsDevice = NULL;
        int rebuildNums = -1;
        CHECKCUDA(cudaMallocAsync(reinterpret_cast<void**>(&rebuildNumsDevice), sizeof(int), stream));
        
        void* d_temp_storage = NULL;
        size_t temp_storage_bytes = 0;
        cub::DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, fixedDepthNums, rebuildNumsDevice, finerSubdivideNum * Constants::maxDepth_Host, stream);
        CHECKCUDA(cudaMallocAsync(&d_temp_storage, temp_storage_bytes, stream));
        cub::DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, fixedDepthNums, rebuildNumsDevice, finerSubdivideNum * Constants::maxDepth_Host, stream);
        CHECKCUDA(cudaMemcpyAsync(&rebuildNums, rebuildNumsDevice, sizeof(int), cudaMemcpyDeviceToHost, stream));
        CHECKCUDA(cudaFreeAsync(d_temp_storage, stream));
        CHECKCUDA(cudaFreeAsync(rebuildNumsDevice, stream));
        //CHECKCUDA(cudaStreamSynchronize(stream));   // 流同步
        //printf("depth = %d   rebuildNums = %d\n", i, rebuildNums);

        for (int depth = 1; depth <= Constants::maxDepth_Host; depth++) {
            void* d_temp_storage_1 = NULL;
            size_t temp_storage_bytes_1 = 0;
            int* LevelNodeCount = NULL;
            CHECKCUDA(cudaMallocAsync(reinterpret_cast<void**>(&LevelNodeCount), sizeof(int), stream));
            cub::DeviceReduce::Sum(d_temp_storage_1, temp_storage_bytes_1, fixedDepthNums + (depth - 1) * finerSubdivideNum, LevelNodeCount, finerSubdivideNum, stream);
            CHECKCUDA(cudaMallocAsync(&d_temp_storage_1, temp_storage_bytes_1, stream));
            cub::DeviceReduce::Sum(d_temp_storage_1, temp_storage_bytes_1, fixedDepthNums + (depth - 1) * finerSubdivideNum, LevelNodeCount, finerSubdivideNum, stream);
            CHECKCUDA(cudaMemcpyAsync(&depthNodeCount[depth], LevelNodeCount, sizeof(int), cudaMemcpyDeviceToHost, stream));
            CHECKCUDA(cudaFreeAsync(d_temp_storage_1, stream));
            CHECKCUDA(cudaFreeAsync(LevelNodeCount, stream));
        }
        //CHECKCUDA(cudaStreamSynchronize(stream));   // 流同步
        //for (int depth = 0; depth <= Constants::maxDepth_Host; depth++) {
        //    printf("depth = %d   depthNodeCount[%d] = %d\n", i, depth, depthNodeCount[depth]);
        //}

        for (int depth = 0; depth <= Constants::maxDepth_Host; depth++) {
            if (depth == 0) depthNodeAddress[depth] = 0;
            else {
                depthNodeAddress[depth] = depthNodeAddress[depth - 1] + depthNodeCount[depth - 1];
            }
        }

        int* depthNodeAddress_Device = NULL;
        CHECKCUDA(cudaMallocAsync(reinterpret_cast<void**>(&depthNodeAddress_Device), sizeof(int) * (Constants::maxDepth_Host + 1), stream));
        CHECKCUDA(cudaMemcpyAsync(depthNodeAddress_Device, depthNodeAddress, sizeof(int) * (Constants::maxDepth_Host + 1), cudaMemcpyHostToDevice, stream));

        int* fixedDepthAddress = NULL;
        CHECKCUDA(cudaMallocAsync(reinterpret_cast<void**>(&fixedDepthAddress), sizeof(int) * finerSubdivideNum * Constants::maxDepth_Host, stream));
        CHECKCUDA(cudaMemsetAsync(fixedDepthAddress, 0, sizeof(int) * finerSubdivideNum * Constants::maxDepth_Host, stream));
        for (int depth = 1; depth <= Constants::maxDepth_Host; depth++) {
            void* d_temp_storage = NULL;
            size_t temp_storage_bytes = 0;
            cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, fixedDepthNums + (depth - 1) * finerSubdivideNum, fixedDepthAddress + (depth - 1) * finerSubdivideNum, finerSubdivideNum, stream);
            CHECKCUDA(cudaMallocAsync(&d_temp_storage, temp_storage_bytes, stream));
            cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, fixedDepthNums + (depth - 1) * finerSubdivideNum, fixedDepthAddress + (depth - 1) * finerSubdivideNum, finerSubdivideNum, stream);
            CHECKCUDA(cudaFreeAsync(d_temp_storage, stream));
        }

        //CHECKCUDA(cudaStreamSynchronize(stream));   // 流同步
        //std::vector<int> fixedDepthAddressTest;
        //fixedDepthAddressTest.resize(finerSubdivideNum * Constants::maxDepth_Host);
        //CHECKCUDA(cudaMemcpy(fixedDepthAddressTest.data(), fixedDepthAddress, sizeof(int) * finerSubdivideNum * Constants::maxDepth_Host, cudaMemcpyDeviceToHost));
        //for (int j = 0; j < fixedDepthAddressTest.size(); j++) {
        //    if (j % 1000 == 0) printf("depth = %d   fixedDepthAddress[%d] = %d\n", i, j, fixedDepthAddressTest[j]);
        //}

        CHECKCUDA(cudaStreamSynchronize(stream));   // 流同步，获得rebuildNums

        const unsigned int rebuildDLevelCount = rebuildNums - depthNodeAddress[Constants::maxDepth_Host];
        //printf("Depth = %d  rebuildNums = %d  depthNodeCount[max] = %d\n", i, rebuildNums, depthNodeCount[Constants::maxDepth_Host]);

        EasyOctNode* RebuildArray = NULL;
        CHECKCUDA(cudaMallocAsync(reinterpret_cast<void**>(&RebuildArray), sizeof(EasyOctNode) * rebuildNums, stream));
        CHECKCUDA(cudaMemsetAsync(RebuildArray, 0, sizeof(EasyOctNode) * rebuildNums, stream));

        int* RebuildDepthBuffer = NULL;
        CHECKCUDA(cudaMallocAsync(reinterpret_cast<void**>(&RebuildDepthBuffer), sizeof(int) * rebuildNums, stream));
        CHECKCUDA(cudaMemsetAsync(RebuildDepthBuffer, 0, sizeof(int) * rebuildNums, stream));

        Point3D<float>* RebuildCenterBuffer = NULL;
        CHECKCUDA(cudaMallocAsync(reinterpret_cast<void**>(&RebuildCenterBuffer), sizeof(Point3D<float>) * rebuildNums, stream));
        CHECKCUDA(cudaMemsetAsync(RebuildCenterBuffer, 0, sizeof(Point3D<float>) * rebuildNums, stream));

        int* ReplaceNodeId = NULL;
        CHECKCUDA(cudaMallocAsync(reinterpret_cast<void**>(&ReplaceNodeId), sizeof(int) * rebuildNums, stream));
        CHECKCUDA(cudaMemsetAsync(ReplaceNodeId, 0, sizeof(int) * rebuildNums, stream));

        int* IsRoot = NULL;
        CHECKCUDA(cudaMallocAsync(reinterpret_cast<void**>(&IsRoot), sizeof(int) * rebuildNums, stream));
        CHECKCUDA(cudaMemsetAsync(IsRoot, 0, sizeof(int) * rebuildNums, stream));

        //printf("depth = %d   finerSubdivideNum = %d\n", i, finerSubdivideNum);
        //printf("depth = %d   finerDepthStart = %d\n", i, finerDepthStart);
        dim3 block_2(128);
        dim3 grid_2(divUp(finerSubdivideNum, block_2.x));
        device::wholeRebuildArray << <grid_2, block_2, 0, stream >> > (SubdivideNode.ArrayView(), finerDepthStart, finerSubdivideNum, NodeArraySize, SubdivideDepthBuffer.DeviceArray().ptr(), depthNodeAddress_Device, fixedDepthAddress, RebuildArray, RebuildDepthBuffer, RebuildCenterBuffer, ReplaceNodeId, IsRoot, NodeArray.Array().ptr());
        
        //CHECKCUDA(cudaStreamSynchronize(stream));
        //std::vector<EasyOctNode> RebuildArrayTest;
        //RebuildArrayTest.resize(rebuildNums);
        //CHECKCUDA(cudaMemcpyAsync(RebuildArrayTest.data(), RebuildArray, sizeof(EasyOctNode) * rebuildNums, cudaMemcpyDeviceToHost, stream));
        //for (int k = 0; k < RebuildArrayTest.size(); k++) {
        //    if (i == finerDepth && k % 1000 == 0) printf("idx = %d    RebuildArray = %d\n", k, RebuildArrayTest[k].children[3]);
        //}

        for (int depth = finerDepth; depth <= Constants::maxDepth_Host; depth++) {
            dim3 block(128);
            dim3 grid(divUp(depthNodeCount[depth], block.x));
            device::computeRebuildNeighbor << <grid, block, 0, stream >> > (NodeArray.ArrayView(), depthNodeAddress[depth], depthNodeCount[depth], NodeArraySize, depth, RebuildArray);
        }

        //CHECKCUDA(cudaStreamSynchronize(stream));
        //std::vector<EasyOctNode> RebuildArrayTest;
        //RebuildArrayTest.resize(rebuildNums);
        //CHECKCUDA(cudaMemcpyAsync(RebuildArrayTest.data(), RebuildArray, sizeof(EasyOctNode) * rebuildNums, cudaMemcpyDeviceToHost, stream));
        //for (int k = 0; k < RebuildArrayTest.size(); k++) {
        //    if (i == finerDepth && k % 1000 == 0) printf("idx = %d    RebuildArray = %d\n", k, RebuildArrayTest[k].neighs[8]);
        //}

        VertexNode* RebuildPreVertexArray = NULL;
        CHECKCUDA(cudaMallocAsync(reinterpret_cast<void**>(&RebuildPreVertexArray), sizeof(VertexNode) * rebuildDLevelCount * 8, stream));
        CHECKCUDA(cudaMemsetAsync(RebuildPreVertexArray, 0, sizeof(VertexNode) * rebuildDLevelCount * 8, stream));

        markValidFinerVexArray.ResizeArrayOrException(rebuildDLevelCount * 8);

        dim3 block_3(128);
        dim3 grid_3(divUp(rebuildDLevelCount, block_3.x));
        device::initSubdivideVertexOwner << <grid_3, block_3, 0, stream >> > (RebuildArray, RebuildCenterBuffer, depthNodeAddress[Constants::maxDepth_Host], rebuildDLevelCount, NodeArraySize, RebuildPreVertexArray, markValidFinerVexArray.Array().ptr());
        
        VertexNode* RebuildVertexArray = NULL;
        CHECKCUDA(cudaMallocAsync(reinterpret_cast<void**>(&RebuildVertexArray), sizeof(VertexNode) * rebuildDLevelCount * 8, stream));
        CHECKCUDA(cudaMemsetAsync(RebuildVertexArray, 0, sizeof(VertexNode) * rebuildDLevelCount * 8, stream));

        int* RebuildVertexArraySize = NULL;
        int RebuildVertexArraySizeHost = -1;
        CHECKCUDA(cudaMallocAsync(reinterpret_cast<void**>(&RebuildVertexArraySize), sizeof(int), stream));

        void* d_temp_storage_1 = NULL;    // 中间变量，用完即可释放
        size_t temp_storage_bytes_1 = 0;  // 中间变量
        CHECKCUDA(cub::DeviceSelect::Flagged(d_temp_storage_1, temp_storage_bytes_1, RebuildPreVertexArray, markValidFinerVexArray.Array().ptr(), RebuildVertexArray, RebuildVertexArraySize, 8 * rebuildDLevelCount, stream, false));	// 确定临时设备存储需求
        CHECKCUDA(cudaMallocAsync(&d_temp_storage_1, temp_storage_bytes_1, stream));
        CHECKCUDA(cub::DeviceSelect::Flagged(d_temp_storage_1, temp_storage_bytes_1, RebuildPreVertexArray, markValidFinerVexArray.Array().ptr(), RebuildVertexArray, RebuildVertexArraySize, 8 * rebuildDLevelCount, stream, false));	// 筛选	
        CHECKCUDA(cudaMemcpyAsync(&RebuildVertexArraySizeHost, RebuildVertexArraySize, sizeof(int), cudaMemcpyDeviceToHost, stream));

        CHECKCUDA(cudaFreeAsync(RebuildPreVertexArray, stream));   // 临时变量，即时删除
        CHECKCUDA(cudaFreeAsync(RebuildVertexArraySize, stream));  // 临时变量，即时删除
        CHECKCUDA(cudaFreeAsync(d_temp_storage_1, stream));        // 临时变量，即时删除

        //CHECKCUDA(cudaStreamSynchronize(stream));
        //printf("depth = %d  RebuildVertexArraySize = %d\n", i, RebuildVertexArraySizeHost);

        dim3 block_4(128);
        dim3 grid_4(divUp(RebuildVertexArraySizeHost, block_4.x));
        device::maintainSubdivideVertexNodePointer << <grid_4, block_4, 0, stream >> > (CenterBuffer, RebuildVertexArraySizeHost, NodeArraySize, RebuildCenterBuffer, RebuildVertexArray, RebuildArray);

        EdgeNode* RebuildPreEdgeArray = NULL;
        CHECKCUDA(cudaMallocAsync(reinterpret_cast<void**>(&RebuildPreEdgeArray), sizeof(EdgeNode) * rebuildDLevelCount * 12, stream));
        CHECKCUDA(cudaMemsetAsync(RebuildPreEdgeArray, 0, sizeof(EdgeNode) * rebuildDLevelCount * 12, stream));

        markValidFinerEdge.ResizeArrayOrException(rebuildDLevelCount * 12);

        dim3 block_5(128);
        dim3 grid_5(divUp(rebuildDLevelCount, block_5.x));
        device::initSubdivideEdgeArray << <grid_5, block_5, 0, stream >> > (RebuildArray, RebuildCenterBuffer, NodeArraySize, depthNodeAddress[Constants::maxDepth_Host], rebuildDLevelCount, RebuildPreEdgeArray, markValidFinerEdge.Array().ptr());

        EdgeNode* RebuildEdgeArray = NULL;
        CHECKCUDA(cudaMallocAsync(reinterpret_cast<void**>(&RebuildEdgeArray), sizeof(EdgeNode)* rebuildDLevelCount * 12, stream));
        CHECKCUDA(cudaMemsetAsync(RebuildEdgeArray, 0, sizeof(EdgeNode)* rebuildDLevelCount * 12, stream));

        int* RebuildEdgeArraySize = NULL;
        int RebuildEdgeArraySizeHost = -1;
        CHECKCUDA(cudaMallocAsync(reinterpret_cast<void**>(&RebuildEdgeArraySize), sizeof(int), stream));

        void* d_temp_storage_2 = NULL;    // 中间变量，用完即可释放
        size_t temp_storage_bytes_2 = 0;  // 中间变量
        CHECKCUDA(cub::DeviceSelect::Flagged(d_temp_storage_2, temp_storage_bytes_2, RebuildPreEdgeArray, markValidFinerEdge.Array().ptr(), RebuildEdgeArray, RebuildEdgeArraySize, 12 * rebuildDLevelCount, stream, false));	// 确定临时设备存储需求
        CHECKCUDA(cudaMallocAsync(&d_temp_storage_2, temp_storage_bytes_2, stream));
        CHECKCUDA(cub::DeviceSelect::Flagged(d_temp_storage_2, temp_storage_bytes_2, RebuildPreEdgeArray, markValidFinerEdge.Array().ptr(), RebuildEdgeArray, RebuildEdgeArraySize, 12 * rebuildDLevelCount, stream, false));	// 筛选	
        CHECKCUDA(cudaMemcpyAsync(&RebuildEdgeArraySizeHost, RebuildEdgeArraySize, sizeof(int), cudaMemcpyDeviceToHost, stream));

        CHECKCUDA(cudaFreeAsync(RebuildPreEdgeArray, stream));      // 临时变量，即时删除
        CHECKCUDA(cudaFreeAsync(RebuildEdgeArraySize, stream));     // 临时变量，即时删除
        CHECKCUDA(cudaFreeAsync(d_temp_storage_2, stream));         // 临时变量，即时删除

        //CHECKCUDA(cudaStreamSynchronize(stream));
        //printf("depth = %d  RebuildVertexArraySize = %d  RebuildEdgeArraySize = %d\n", i, RebuildVertexArraySizeHost, RebuildEdgeArraySizeHost);

        dim3 block_6(128);
        dim3 grid_6(divUp(RebuildEdgeArraySizeHost, block_6.x));
        device::maintainSubdivideEdgeNodePointer << <grid_6, block_6, 0, stream >> > (CenterBuffer, RebuildCenterBuffer, RebuildEdgeArraySizeHost, NodeArraySize, RebuildArray, RebuildEdgeArray);

        float* RebuildVvalue = NULL;
        CHECKCUDA(cudaMallocAsync(reinterpret_cast<void**>(&RebuildVvalue), sizeof(float) * RebuildVertexArraySizeHost, stream));
        CHECKCUDA(cudaMemsetAsync(RebuildVvalue, 0, sizeof(float) * RebuildVertexArraySizeHost, stream));

        dim3 block_7(128);
        dim3 grid_7(divUp(RebuildVertexArraySizeHost, block_7.x));
        device::computeSubdivideVertexImplicitFunctionValue << <grid_7, block_7, 0, stream >> > (RebuildVertexArray, RebuildArray, NodeArray.ArrayView(), dx, encodeNodeIndexInFunction, BaseFunction, NodeArraySize, ReplaceNodeId, IsRoot, RebuildVertexArraySizeHost, isoValue, RebuildVvalue);

        CHECKCUDA(cudaFreeAsync(ReplaceNodeId, stream));
        CHECKCUDA(cudaFreeAsync(IsRoot, stream));

        int* RebuildVexNums = NULL;
        CHECKCUDA(cudaMallocAsync(reinterpret_cast<void**>(&RebuildVexNums), sizeof(int) * RebuildEdgeArraySizeHost, stream));
        CHECKCUDA(cudaMemsetAsync(RebuildVexNums, 0, sizeof(int) * RebuildEdgeArraySizeHost, stream));

        dim3 block_8(128);
        dim3 grid_8(divUp(RebuildEdgeArraySizeHost, block_8.x));
        device::generateSubdivideVexNums << <grid_8, block_8, 0, stream >> > (RebuildEdgeArray, RebuildArray, RebuildEdgeArraySizeHost, NodeArraySize, RebuildVvalue, RebuildVexNums, markValidFinerVexNum.Array().ptr());

        int* RebuildVexAddress = NULL;
        CHECKCUDA(cudaMallocAsync(reinterpret_cast<void**>(&RebuildVexAddress), sizeof(int) * RebuildEdgeArraySizeHost, stream));
        CHECKCUDA(cudaMemsetAsync(RebuildVexAddress, 0, sizeof(int) * RebuildEdgeArraySizeHost, stream));

        void* d_temp_storage_3 = NULL;
        size_t temp_storage_bytes_3 = 0;
        cub::DeviceScan::ExclusiveSum(d_temp_storage_3, temp_storage_bytes_3, RebuildVexNums, RebuildVexAddress, RebuildEdgeArraySizeHost, stream);
        CHECKCUDA(cudaMallocAsync(&d_temp_storage_3, temp_storage_bytes_3, stream));
        cub::DeviceScan::ExclusiveSum(d_temp_storage_3, temp_storage_bytes_3, RebuildVexNums, RebuildVexAddress, RebuildEdgeArraySizeHost, stream);
        CHECKCUDA(cudaFreeAsync(d_temp_storage_3, stream));

        CHECKCUDA(cudaStreamSynchronize(stream));   // 需要同步
        int RebuildLastVexAddr = -1;
        int RebuildLastVexNums = -1;
        CHECKCUDA(cudaMemcpyAsync(&RebuildLastVexAddr, RebuildVexAddress + RebuildEdgeArraySizeHost - 1, sizeof(int), cudaMemcpyDeviceToHost, stream));
        CHECKCUDA(cudaMemcpyAsync(&RebuildLastVexNums, RebuildVexNums + RebuildEdgeArraySizeHost - 1, sizeof(int), cudaMemcpyDeviceToHost, stream));
        int RebuildAllVexNums = RebuildLastVexAddr + RebuildLastVexNums;

        CHECKCUDA(cudaStreamSynchronize(stream));   // 需要同步
        //printf("depth = %d   RebuildAllVexNums = %d\n", i, RebuildAllVexNums);

        int* RebuildTriNums = NULL;
        CHECKCUDA(cudaMallocAsync(reinterpret_cast<void**>(&RebuildTriNums), sizeof(int) * rebuildDLevelCount, stream));
        CHECKCUDA(cudaMemsetAsync(RebuildTriNums, 0, sizeof(int) * rebuildDLevelCount, stream));

        int* RebuildCubeCatagory = NULL;
        CHECKCUDA(cudaMallocAsync(reinterpret_cast<void**>(&RebuildCubeCatagory), sizeof(int) * rebuildDLevelCount, stream));
        CHECKCUDA(cudaMemsetAsync(RebuildCubeCatagory, 0, sizeof(int) * rebuildDLevelCount, stream));

        dim3 block_9(128);
        dim3 grid_9(divUp(rebuildDLevelCount, block_9.x));
        device::generateTriNums << <grid_9, block_9, 0, stream >> > (RebuildArray, depthNodeAddress[Constants::maxDepth_Host], rebuildDLevelCount, RebuildVvalue, RebuildTriNums, RebuildCubeCatagory);

        int* RebuildTriAddress = NULL;
        CHECKCUDA(cudaMallocAsync(reinterpret_cast<void**>(&RebuildTriAddress), sizeof(int)* rebuildDLevelCount, stream));
        CHECKCUDA(cudaMemsetAsync(RebuildTriAddress, 0, sizeof(int)* rebuildDLevelCount, stream));

        void* d_temp_storage_4 = NULL;
        size_t temp_storage_bytes_4 = 0;
        cub::DeviceScan::ExclusiveSum(d_temp_storage_4, temp_storage_bytes_4, RebuildTriNums, RebuildTriAddress, rebuildDLevelCount, stream);
        CHECKCUDA(cudaMallocAsync(&d_temp_storage_4, temp_storage_bytes_4, stream));
        cub::DeviceScan::ExclusiveSum(d_temp_storage_4, temp_storage_bytes_4, RebuildTriNums, RebuildTriAddress, rebuildDLevelCount, stream);
        CHECKCUDA(cudaFreeAsync(d_temp_storage_4, stream));

        Point3D<float>* RebuildVertexBuffer = NULL;
        CHECKCUDA(cudaMallocAsync(reinterpret_cast<void**>(&RebuildVertexBuffer), sizeof(Point3D<float>)* RebuildAllVexNums, stream));
        //std::vector<Point3D<float>> RebuildVertexBufferHost;
        //RebuildVertexBufferHost.resize(RebuildAllVexNums);

        EdgeNode* RebuildValidEdgeArray = NULL;
        CHECKCUDA(cudaMallocAsync(reinterpret_cast<void**>(&RebuildValidEdgeArray), sizeof(EdgeNode)* RebuildAllVexNums, stream));

        int* RebuildValidEdgeArraySize = NULL;
        int RebuildValidEdgeArraySizeHost = -1;
        CHECKCUDA(cudaMallocAsync(reinterpret_cast<void**>(&RebuildValidEdgeArraySize), sizeof(int), stream));

        void* d_temp_storage_5 = NULL;    // 中间变量，用完即可释放
        size_t temp_storage_bytes_5 = 0;  // 中间变量
        CHECKCUDA(cub::DeviceSelect::Flagged(d_temp_storage_5, temp_storage_bytes_5, RebuildEdgeArray, markValidFinerVexNum.Array().ptr(), RebuildValidEdgeArray, RebuildValidEdgeArraySize, RebuildEdgeArraySizeHost, stream, false));	// 确定临时设备存储需求
        CHECKCUDA(cudaMallocAsync(&d_temp_storage_5, temp_storage_bytes_5, stream));
        CHECKCUDA(cub::DeviceSelect::Flagged(d_temp_storage_5, temp_storage_bytes_5, RebuildEdgeArray, markValidFinerVexNum.Array().ptr(), RebuildValidEdgeArray, RebuildValidEdgeArraySize, RebuildEdgeArraySizeHost, stream, false));	// 筛选	
        CHECKCUDA(cudaMemcpyAsync(&RebuildValidEdgeArraySizeHost, RebuildValidEdgeArraySize, sizeof(int), cudaMemcpyDeviceToHost, stream));

        CHECKCUDA(cudaFreeAsync(d_temp_storage_5, stream));
        CHECKCUDA(cudaFreeAsync(RebuildValidEdgeArraySize, stream));

        int* RebuildValidVexAddress = NULL;
        CHECKCUDA(cudaMallocAsync(reinterpret_cast<void**>(&RebuildValidVexAddress), sizeof(int)* RebuildAllVexNums, stream));

        int* RebuildValidVexAddressSize = NULL;
        int RebuildValidVexAddressSizeHost = -1;
        CHECKCUDA(cudaMallocAsync(reinterpret_cast<void**>(&RebuildValidVexAddressSize), sizeof(int), stream));

        void* d_temp_storage_6 = NULL;    // 中间变量，用完即可释放
        size_t temp_storage_bytes_6 = 0;  // 中间变量
        CHECKCUDA(cub::DeviceSelect::Flagged(d_temp_storage_6, temp_storage_bytes_6, RebuildVexAddress, markValidFinerVexNum.Array().ptr(), RebuildValidVexAddress, RebuildValidVexAddressSize, RebuildEdgeArraySizeHost, stream, false));	// 确定临时设备存储需求
        CHECKCUDA(cudaMallocAsync(&d_temp_storage_6, temp_storage_bytes_6, stream));
        CHECKCUDA(cub::DeviceSelect::Flagged(d_temp_storage_6, temp_storage_bytes_6, RebuildVexAddress, markValidFinerVexNum.Array().ptr(), RebuildValidVexAddress, RebuildValidVexAddressSize, RebuildEdgeArraySizeHost, stream, false));	// 筛选	
        CHECKCUDA(cudaMemcpyAsync(&RebuildValidVexAddressSizeHost, RebuildValidVexAddressSize, sizeof(int), cudaMemcpyDeviceToHost, stream));

        CHECKCUDA(cudaFreeAsync(d_temp_storage_6, stream));
        CHECKCUDA(cudaFreeAsync(RebuildValidVexAddressSize, stream));

        dim3 block_10(128);
        dim3 grid_10(divUp(RebuildAllVexNums, block_10.x));
        device::generateSubdivideIntersectionPoint << <grid_10, block_10, 0, stream >> > (RebuildValidEdgeArray, RebuildVertexArray, RebuildArray, RebuildValidVexAddress, RebuildVvalue, RebuildAllVexNums, NodeArraySize, RebuildVertexBuffer);

        CHECKCUDA(cudaFreeAsync(RebuildValidEdgeArray, stream));
        CHECKCUDA(cudaFreeAsync(RebuildValidVexAddress, stream));
        CHECKCUDA(cudaFreeAsync(RebuildVvalue, stream));

        int RebuildLastTriAddr = -1;
        int RebuildLastTriNums = -1;

        CHECKCUDA(cudaMemcpyAsync(&RebuildLastTriAddr, RebuildTriAddress + rebuildDLevelCount - 1, sizeof(int), cudaMemcpyDeviceToHost, stream));
        CHECKCUDA(cudaMemcpyAsync(&RebuildLastTriNums, RebuildTriNums + rebuildDLevelCount - 1, sizeof(int), cudaMemcpyDeviceToHost, stream));
    
        CHECKCUDA(cudaStreamSynchronize(stream));
        int RebuildAllTriNums = RebuildLastTriAddr + RebuildLastTriNums;

        CHECKCUDA(cudaFreeAsync(RebuildVexNums, stream));

        //printf("RebuildAllTriNums = %d\n", RebuildAllTriNums);

        TriangleIndex* RebuildTriangleBuffer = NULL;
        CHECKCUDA(cudaMallocAsync(reinterpret_cast<void**>(&RebuildTriangleBuffer), sizeof(TriangleIndex) * RebuildAllTriNums, stream));
        //std::vector<int> RebuildTriangleBufferHost;
        //RebuildTriangleBufferHost.resize(RebuildAllTriNums * 3);

        CHECKCUDA(cudaStreamSynchronize(stream));   // 流同步
        //printf("###############################   Depth = %d   #################################\n", i);
        dim3 block_11(128);
        dim3 grid_11(divUp(rebuildDLevelCount, block_11.x));
        device::generateSubdivideTrianglePos << <grid_11, block_11, 0, stream >> > (RebuildArray, depthNodeAddress[Constants::maxDepth_Host], rebuildDLevelCount, RebuildTriNums, RebuildCubeCatagory, RebuildVexAddress, RebuildTriAddress, RebuildTriangleBuffer);
        insertTriangle(RebuildVertexBuffer, RebuildAllVexNums, RebuildTriangleBuffer, RebuildAllTriNums, stream);

        //CHECKCUDA(cudaMemcpyAsync(RebuildVertexBufferHost.data(), RebuildVertexBuffer, sizeof(Point3D<float>) * RebuildAllVexNums, cudaMemcpyDeviceToHost, stream));
        //CHECKCUDA(cudaMemcpyAsync(RebuildTriangleBufferHost.data(), RebuildTriangleBuffer, sizeof(int) * 3 * RebuildAllTriNums, cudaMemcpyDeviceToHost, stream));

        //CHECKCUDA(cudaStreamSynchronize(stream));   // 流同步

        //insertTriangle(RebuildVertexBufferHost.data(), RebuildAllVexNums, RebuildTriangleBufferHost.data(), RebuildAllTriNums, mesh);

        CHECKCUDA(cudaFreeAsync(fixedDepthNums, stream));
        CHECKCUDA(cudaFreeAsync(depthNodeAddress_Device, stream));
        CHECKCUDA(cudaFreeAsync(fixedDepthAddress, stream));
        CHECKCUDA(cudaFreeAsync(RebuildArray, stream));
        CHECKCUDA(cudaFreeAsync(RebuildDepthBuffer, stream));
        CHECKCUDA(cudaFreeAsync(RebuildCenterBuffer, stream));
        CHECKCUDA(cudaFreeAsync(RebuildVertexArray, stream));
        CHECKCUDA(cudaFreeAsync(RebuildEdgeArray, stream));
        CHECKCUDA(cudaFreeAsync(RebuildVexAddress, stream));
        CHECKCUDA(cudaFreeAsync(RebuildTriNums, stream));
        CHECKCUDA(cudaFreeAsync(RebuildCubeCatagory, stream));
        CHECKCUDA(cudaFreeAsync(RebuildTriAddress, stream));
        CHECKCUDA(cudaFreeAsync(RebuildVertexBuffer, stream));
        CHECKCUDA(cudaFreeAsync(RebuildTriangleBuffer, stream));
    }
}

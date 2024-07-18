#pragma once

//#define CHECK_MESH_BUILD_TIME_COST	// 查看网格重建过程中每个步骤时间消耗

#define CUB_IGNORE_DEPRECATED_API

#define MAX_SURFEL_COUNT 200000			// 最大面元个数
#define MAX_MESH_TRIANGLE_COUNT 600000	// 最大网格三角形数量

#define FORCE_UNIT_NORMALS 1

#define STACKCAPACITY 2000

#define CONVTIMES 2

#define NORMALIZE 0

#define DIMENSION 3

#define TxTDataPath "E:/PoissonReconstructionGPU/PoissonSurfaceReconstructionGPU/Poisson-Surface-Reconstruction-GPU/PointCloudData/data_txt/bunny.txt"

#define PlyDataPath "E:/PoissonReconstructionGPU/PoissonSurfaceReconstructionGPU/Poisson-Surface-Reconstruction-GPU/OutputResult/bunny.ply"

#define PcdDataPath "E:/PoissonReconstructionGPU/PoissonSurfaceReconstructionGPU/Poisson-Surface-Reconstruction-GPU/PointCloudData/data_pcd_without_normal/bunny.pcd"

#define PlySavePath "E:/PoissonReconstructionGPU/PoissonSurfaceReconstructionGPU/Poisson-Surface-Reconstruction-GPU/PointCloudData/data_ply_normal/bunny/bunny.ply"

#define ESC_KEY 27

#define MAX_THREADS 10

#define MAX_DEPTH_OCTREE 7	// octree最大深度

#define MAX_MESH_STREAM 5	// 最大执行mesh任务的cuda流数量

#define F_DATA_RES ((1 << (MAX_DEPTH_OCTREE + 1)) - 1)					// 2^(maxDepth + 1) - 1
#define F_DATA_RES_SQUARE F_DATA_RES * F_DATA_RES						// 2047^2

#define D_LEVEL_MAX_NODE 8 * MAX_SURFEL_COUNT							// maxDepth层节点的最坏情况应该是8 * MAX_SURFEL_COUNT
#define TOTAL_NODEARRAY_MAX_COUNT MAX_SURFEL_COUNT * 10					// NodeArray最大的数量
#define TOTAL_VERTEXARRAY_MAX_COUNT 8* TOTAL_NODEARRAY_MAX_COUNT		// NodeArray最大的数量 * 8(8个顶点)
#define TOTAL_EDGEARRAY_MAX_COUNT 12 * D_LEVEL_MAX_NODE					// NodeArray中maxDepth层中节点数量 * 12
#define TOTAL_FACEARRAY_MAX_COUNT 6 * TOTAL_NODEARRAY_MAX_COUNT			// NodeArray最大的数量 * 6(6个面)

#define RESOLUTION (1 << (MAX_DEPTH_OCTREE + 1)) - 1	// 分辨率

#define COARSER_DIVERGENCE_LEVEL_NUM 4												// 计算粗略节点所需节点层数[0, LevelNum]
#define TOTAL_FINER_NODE_NUM 6 * MAX_SURFEL_COUNT									// [maxDepth - LevelNum, maxDepth]层的节点总数
#define TOTAL_COARSER_NODE_NUM TOTAL_NODEARRAY_MAX_COUNT - TOTAL_FINER_NODE_NUM		// [1, maxDepth - LevelNum - 1]层的节点总数

#define EPSILON float(1e-6)

#define WINDOW_WIDTH 1728
#define WINDOW_HEIGHT 972

#define SHADER_PATH_PREFIX "E:/PoissonReconstructionGPU/PoissonSurfaceReconstructionGPU/Poisson-Surface-Reconstruction-GPU/render/shaders/"

//#define ROUND_EPS float(1e-5)
//#define maxDepth 8
//#define markOffset 31
//#define resolution (1 << (maxDepth + 1)) - 1
//#define resolution 511
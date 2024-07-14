/*****************************************************************//**
 * \file   ComputeTriangleIndices.h
 * \brief  插入修复三角网格，构建网格
 * 
 * \author LUOJIAXUAN
 * \date   June 3rd 2024
 *********************************************************************/
#pragma once
#include <base/Constants.h>
#include <base/DeviceReadWrite/SynchronizeArray.h>
#include <base/DeviceReadWrite/DeviceBufferArray.h>
#include <base/GlobalConfigs.h>
#include "ConfirmedPPolynomial.h"
#include "OctNode.cuh"

namespace SparseSurfelFusion {
	namespace device {
		/**
		 * \brief 计算顶点vertex隐式函数核函数.
		 * 
		 * \param VertexArray 顶点数组
		 * \param NodeArray 节点数组
		 * \param BaseFunctions 基函数
		 * \param dx 散度
		 * \param encodeNodeIndexInFunction 基函数索引
		 * \param isoValue 等值
		 * \param vvalue 顶点隐函数值
		 */
		__global__ void ComputeVertexImplicitFunctionValueKernel(DeviceArrayView<VertexNode> VertexArray, DeviceArrayView<OctNode> NodeArray, DeviceArrayView<ConfirmedPPolynomial<CONVTIMES + 1, CONVTIMES + 2>> BaseFunctions, DeviceArrayView<float> dx, DeviceArrayView<int> encodeNodeIndexInFunction, const unsigned int VertexArraySize, const float isoValue, float* vvalue);
	
		/**
		 * \brief 生成顶点的vertexNums和顶点的vertexAddress的核函数.
		 *
		 * \param EdgeArray 边数组
		 * \param NodeArray 节点数组
		 * \param vvalue 顶点隐式函数值
		 * \param vexNums 辅助计算节点位置VertexAddress
		 */
		__global__ void generateVertexNumsKernel(DeviceArrayView<EdgeNode> EdgeArray, DeviceArrayView<OctNode> NodeArray, DeviceArrayView<float> vvalue, const unsigned int EdgeArraySize, int* vexNums, bool* markValidVertex);

		/**
		 * \brief 生成TriangleNums以及cubeCatagory核函数.
		 * 
		 * \param NodeArray 节点数组
		 * \param vvalue 顶点隐式函数值
		 * \param DLevelOffset 第maxDepth层的首节点在NodeArray中偏移
		 * \param DLevelNodeCount 第maxDepth层节点数量
		 * \param triNums 三角形数量
		 * \param cubeCatagory 立方体类型
		 */
		__global__ void generateTriangleNumsKernel(DeviceArrayView<OctNode> NodeArray, DeviceArrayView<float> vvalue, const unsigned int DLevelOffset, const unsigned int DLevelNodeCount, int* triNums, int* cubeCatagory);

		/**
		 * \brief 生成交叉点.
		 * 
		 * \param NodeArray 节点数组
		 * \param validEdgeArray 筛选后有效的边数组
		 * \param VertexArray 顶点数组
		 * \param vvalue 顶点隐式函数值
		 * \param validVexAddress 有效顶点的位置
		 * \param validEdgeArraySize 有效边数组的大小
		 * \param VertexBuffer 有效的顶点
		 */
		__global__ void generateIntersectionPoint(DeviceArrayView<OctNode> NodeArray, DeviceArrayView<VertexNode> VertexArray, DeviceArrayView<float> vvalue, const EdgeNode* validEdgeArray, const int* validVexAddress, const unsigned int validEdgeArraySize, Point3D<float>* VertexBuffer);
		
		/**
		 * \brief 计算两点之间的插入点.
		 * 
		 * \param p1 点1
		 * \param p2 点2
		 * \param dim 需要进行插入的维度
		 * \param v1 隐式函数值1
		 * \param v2 隐式函数值2
		 * \param out 【输入】插入点的坐标
		 */
		__device__ void interpolatePoint(const Point3D<float>& p1, const Point3D<float>& p2, const int& dim, const float& v1, const float& v2, Point3D<float>& out);

		/**
		 * \brief 生成三角形位置以及构造面是否相交的数组.
		 * 
		 * \param NodeArray 节点数组
		 * \param FaceArray 面数组
		 * \param DLevelOffset 第maxDepth层的首节点在NodeArray中偏移
		 * \param DLevelNodeCount 第maxDepth层节点数量
		 * \param triNums 三角形数量
		 * \param cubeCatagory 立方体类型
		 * \param vexAddress 顶点偏移地址
		 * \param triAddress 三角形偏移地址
		 * \param TriangleBuffer 记录所构成的三角形
		 */
		__global__ void generateTrianglePos(DeviceArrayView<OctNode> NodeArray, DeviceArrayView<FaceNode> FaceArray, DeviceArrayView<int> triNums, DeviceArrayView<int> cubeCatagory, DeviceArrayView<int> vexAddress, DeviceArrayView<int> triAddress, const unsigned int DLevelOffset, const unsigned int DLevelNodeCount, int* TriangleBuffer, int* hasSurfaceIntersection);

		/**
		 * \brief 获得细分三角形位置.
		 */
		__global__ void generateSubdivideTrianglePos(const EasyOctNode* SubdivideArray, const unsigned int DLevelOffset, const unsigned int DLevelNodeCount, const int* SubdivideTriNums, const int* SubdivideCubeCatagory, const int* SubdivideVexAddress, const int* SubdivideTriAddress, int* SubdivideTriangleBuffer);

		/**
		 * \brief 处理其他层级的叶子节点的三角形和相交情况, 记录在NodeArray中.
		 * 
		 * \param VertexArray 顶点数组
		 * \param vvalue 顶点隐式函数值
		 * \param OtherDepthNodeCount 第[0, maxDepth - 1]层的首节点在NodeArray中偏移
		 * \param hasSurfaceIntersection 是否面相交
		 * \param NodeArray 节点数组
		 * \param markValidSubdividedNode 标记可以细分的节点idx
		 */
		__global__ void ProcessLeafNodesAtOtherDepth(DeviceArrayView<VertexNode> VertexArray, DeviceArrayView<float> vvalue, const unsigned int OtherDepthNodeCount, const int* hasSurfaceIntersection, OctNode* NodeArray, bool* markValidSubdividedNode);
	
		/**
		 * \brief 计算细分节点所在的深度，以及每一层一个多少细分节点.
		 * 
		 * \param SubdivideNode 细分节点数组
		 * \param DepthBuffer 记录NodeArray深度节点的数组
		 * \param SubdivideNum 细分节点的数量
		 * \param SubdivideDepthBuffer 记录细分节点所在深度数组
		 * \param SubdivideDepthNum 记录当前深度细分节点的总数
		 */
		__global__ void precomputeSubdivideDepth(DeviceArrayView<OctNode> SubdivideNode, DeviceArrayView<unsigned int> DepthBuffer, const int SubdivideNum, int* SubdivideDepthBuffer, int* SubdivideDepthNum);
	
		/**
		 * \brief 重构NodeArray，记录在SubdivideArray中.
		 * 
		 * \param SubdivideNode 细分节点
		 * \param SubdivideDepthBuffer 细分节点深度
		 * \param iterRound 迭代次数
		 * \param NodeArraySize NodeArray数组的长度
		 * \param SubdivideArraySize 细分节点数组的长度
		 * \param SubdivideArray 记录重构的数据
		 * \param SubdivideArrayDepthBuffer 重构数据节点深度
		 * \param SubdivideArrayCenterBuffer 重构数据节点中心点位置
		 */
		__global__ void singleRebuildArray(DeviceArrayView<OctNode> SubdivideNode, DeviceArrayView<int> SubdivideDepthBuffer, const unsigned int iterRound, const unsigned int NodeArraySize, const unsigned int SubdivideArraySize, EasyOctNode* SubdivideArray, int* SubdivideArrayDepthBuffer, Point3D<float>* SubdivideArrayCenterBuffer);

		/**
		 * \brief 获得细分节点的深度.
		 * 
		 * \param rootDepth 细分节点根的深度
		 * \param idx 细分节点的index
		 */
		__device__ int getSubdivideDepth(const int& rootDepth, const int& idx);

		/**
		 * \brief 获得当前节点的中心点.
		 * 
		 * \param key 当前节点的key键值
		 * \param currentDepth 当前深度
		 * \param center 【输出】当前节点的中心点
		 */
		__device__ void getNodeCenterAllDepth(const int& key, const int& currentDepth, Point3D<float>& center);

		/**
		 * \brief 计算重构的节点邻居.
		 * 
		 * \param NodeArray 原始节点数组
		 * \param currentLevelOffset 当前层节点的偏移
		 * \param currentLevelNodesCount 当前层节点的数量
		 * \param NodeArraySize 原始节点数量
		 * \param depth 当前深度
		 * \param SubdivideArray 细分节点数组
		 */
		__global__ void computeRebuildNeighbor(DeviceArrayView<OctNode> NodeArray, const unsigned int currentLevelOffset, const unsigned int currentLevelNodesCount, const unsigned int NodeArraySize, const unsigned int depth, EasyOctNode* SubdivideArray);
	
		/**
		 * \brief 初始化细分顶点的所拥有的细分节点.
		 * 
		 * \param SubdivideArray 细分节点数组
		 * \param SubdivideArrayCenterBuffer 细分节点中心点数组
		 * \param currentLevelOffset 当前层节点的偏移
		 * \param currentLevelNodesCount 当前层节点的数量
		 * \param NodeArraySize 原始节点数量
		 * \param SubdividePreVertexArray 预处理细分节点，后续获得有效的VertexArray
		 */
		__global__ void initSubdivideVertexOwner(const EasyOctNode* SubdivideArray, const Point3D<float>* SubdivideArrayCenterBuffer, const unsigned int currentLevelOffset, const unsigned int currentLevelNodesCount, const unsigned int NodeArraySize, VertexNode* SubdividePreVertexArray, bool* markValidSubdivideVertex);

		/**
		 * \brief 计算两点之间距离平方.
		 *
		 * \param p1 点1
		 * \param p2 点2
		 * \return 返回两点之间距离平方
		 */
		__forceinline__ __device__ double SquareDistance(const Point3D<float>& p1, const Point3D<float>& p2);

		/**
		 * \brief 更新SubdivideArray中的顶点.
		 * 
		 * \param CenterBuffer 节点的中心位置
		 * \param VertexArraySize 顶点数量
		 * \param NodeArraySize 节点数量
		 * \param SubdivideArrayCenterBuffer 细分节点的中心位置
		 * \param VertexArray 顶点数组
		 * \param SubdivideArray 细分数组
		 */
		__global__ void maintainSubdivideVertexNodePointer(DeviceArrayView<Point3D<float>> CenterBuffer, const unsigned int VertexArraySize, const unsigned int NodeArraySize, const Point3D<float>* SubdivideArrayCenterBuffer, VertexNode* VertexArray, EasyOctNode* SubdivideArray);

		/**
		 * \brief 初始化细分顶点的所有边.
		 * 
		 * \param SubdivideArray 细分数组
		 * \param SubdivideArrayCenterBuffer 细分节点的中心位置
		 * \param NodeArraySize 节点数量
		 * \param DLevelOffset 最大层偏移
		 * \param DLevelNodeCount 最大层节点数量
		 * \param SubdividePreEdgeArray 预处理边数组，后续会筛掉无效边
		 * \param markValidSubdivideEdge 对有效边做标记
		 */ 
		__global__ void initSubdivideEdgeArray(const EasyOctNode* SubdivideArray, const Point3D<float>* SubdivideArrayCenterBuffer, const unsigned int NodeArraySize, const unsigned int DLevelOffset, const unsigned int DLevelNodeCount, EdgeNode* SubdividePreEdgeArray, bool* markValidSubdivideEdge);

		/**
		 * \brief 维护更新细分边的参数.
		 * 
		 * \param CenterBuffer 节点的中心位置
		 * \param SubdivideArrayCenterBuffer 细分节点的中心位置
		 * \param EdgeArraySize 边数组的大小
		 * \param NodeArraySize 节点数量
		 * \param SubdivideArray 细分数组
		 * \param EdgeArray 细分边数组
		 */
		__global__ void maintainSubdivideEdgeNodePointer(DeviceArrayView<Point3D<float>> CenterBuffer, const Point3D<float>* SubdivideArrayCenterBuffer, const unsigned int EdgeArraySize, const unsigned int NodeArraySize, EasyOctNode* SubdivideArray, EdgeNode* EdgeArray);
		
		/**
		 * \brief 计算细分顶点的隐式函数值.
		 * 
		 * \param SubdivideVertexArray 
		 * \param SubdivideArray
		 * \param NodeArray
		 * \param dx
		 * \param EncodedNodeIdxInFunction
		 * \param baseFunctions
		 * \param NodeArraySize
		 * \param rootId
		 * \param SubdivideVertexArraySize
		 * \param isoValue
		 * \param SubdivideVvalue 
		 */
		__global__ void computeSubdivideVertexImplicitFunctionValue(const VertexNode* SubdivideVertexArray, const EasyOctNode* SubdivideArray, DeviceArrayView<OctNode> NodeArray, DeviceArrayView<float> dx, DeviceArrayView<int> EncodedNodeIdxInFunction, DeviceArrayView<ConfirmedPPolynomial<CONVTIMES + 1, CONVTIMES + 2>> baseFunctions, const unsigned int NodeArraySize, const unsigned int rootId, const unsigned int SubdivideVertexArraySize, const float isoValue, float* SubdivideVvalue);

		/**
		 * \brief 计算细分顶点的隐式函数值【Finer】.
		 */
		__global__ void computeSubdivideVertexImplicitFunctionValue(const VertexNode* SubdivideVertexArray, const EasyOctNode* SubdivideArray, DeviceArrayView<OctNode> NodeArray, DeviceArrayView<float> dx, DeviceArrayView<int> EncodedNodeIdxInFunction, DeviceArrayView<ConfirmedPPolynomial<CONVTIMES + 1, CONVTIMES + 2>> baseFunctions, const unsigned int NodeArraySize, const int* ReplacedNodeId, const int* IsRoot, const unsigned int SubdivideVertexArraySize, const float isoValue, float* SubdivideVvalue);
			 
			 
			 
			
			

		/**
		 * \brief 计算细分顶点数量，从而获得Address.
		 * 
		 * \param SubdivideEdgeArray 细分边
		 * \param SubdivideArray 细分数组
		 * \param SubdivideEdgeArraySize 细分边数组的大小
		 * \param NodeArraySize 节点数量
		 * \param SubdivideVvalue 节点的隐式函数值
		 * \param markValidSubdivedeVexNum 标记有效的细分vexNum的index
		 */
		__global__ void generateSubdivideVexNums(const EdgeNode* SubdivideEdgeArray, const EasyOctNode* SubdivideArray, const unsigned int SubdivideEdgeArraySize, const unsigned int NodeArraySize, const float* SubdivideVvalue, int* SubdivideVexNums, bool* markValidSubdivedeVexNum);

		/**
		 * \brief 生成细分节点的三角形.
		 * 
		 * \param SubdivideNodeArray 细分节点数组，与前面的NodeArray不一样
		 * \param DLevelOffset maxDepth层的首节点在SubdivideNodeArray中的位置
		 * \param DLevelNodeCount maxDepth层的节点数量
		 * \param vvalue 隐函数值
		 * \param triNums 三角形数量
		 * \param cubeCatagory 立方体类型
		 */
		__global__ void generateTriNums(const EasyOctNode* SubdivideNodeArray, const unsigned int DLevelOffset, const unsigned int DLevelNodeCount, const float* vvalue, int* triNums, int* cubeCatagory);

		
		/**
		 * \brief 生成细分的相交点.
		 */
		__global__ void generateSubdivideIntersectionPoint(const EdgeNode* SubdivideValidEdgeArray, const VertexNode* SubdivideVertexArray, const EasyOctNode* SubdivideArray, const int* SubdivideValidVexAddress, const float* SubdivideVvalue, const unsigned int SubdivideValidEdgeArraySize, const unsigned int NodeArraySize, Point3D<float>* SubdivideVertexBuffer);
	
		/**
		 * \brief 初始化固定每层的节点数量.
		 */
		__global__ void initFixedDepthNums(DeviceArrayView<OctNode> SubdivideNode, DeviceArrayView<int> SubdivideDepthBuffer, const unsigned int DepthOffset, const unsigned int DepthNodeCount, int* fixedDepthNums);

		/**
		 * \brief 重建整个数组.
		 */
		__global__ void wholeRebuildArray(DeviceArrayView<OctNode> SubdivideNode, const unsigned int finerDepthStart, const unsigned int finerSubdivideNum, const unsigned int NodeArraySize, const int* SubdivideDepthBuffer, const int* depthNodeAddress_Device, const int* fixedDepthAddress, EasyOctNode* RebuildArray, int* RebuildDepthBuffer, Point3D<float>* RebuildCenterBuffer, int* ReplaceNodeId, int* IsRoot, OctNode* NodeArray);

	}
	class ComputeTriangleIndices
	{
	public:
		ComputeTriangleIndices();

		~ComputeTriangleIndices();

		using Ptr = std::shared_ptr<ComputeTriangleIndices>;

		/**
		 * \brief 计算顶点构造三角剖分的索引数组.
		 * 
		 * \param VertexArray 顶点数组
		 * \param EdgeArray 边数组
		 * \param FaceArray 面数组
		 * \param NodeArray 节点数组
		 * \param BaseFunction 基函数
		 * \param dx 散度
		 * \param encodeNodeIndexInFunction 基函数索引
		 * \param BaseFunctions 基函数
		 * \param isoValue 等值
		 * \param DLevelOffset maxDepth层NodeArray偏移
		 * \param stream cuda流
		 */
		void calculateTriangleIndices(DeviceArrayView<VertexNode> VertexArray, DeviceArrayView<EdgeNode> EdgeArray, DeviceArrayView<FaceNode> FaceArray, DeviceBufferArray<OctNode>& NodeArray, DeviceArrayView<ConfirmedPPolynomial<CONVTIMES + 1, CONVTIMES + 2>> BaseFunction, DeviceArrayView<float> dx, DeviceArrayView<int> encodeNodeIndexInFunction, DeviceArrayView<unsigned int> DepthBuffer, DeviceArrayView<Point3D<float>> CenterBuffer, const float isoValue, const unsigned int DLevelOffset, const unsigned int DLevelNodeCount, CoredVectorMeshData& mesh, cudaStream_t stream);

		///**
		// * \brief 获得计算好的网格.
		// *
		// * \return 计算好的网格
		// */
		//CoredVectorMeshData GetMesh() { return mesh; }

	private:
		DeviceBufferArray<float> vvalue;								// 【论文参数】顶点隐式函数值
		DeviceBufferArray<int> vexNums;									// 【论文参数】顶点的数量
		DeviceBufferArray<int> vexAddress;								// 【论文参数】顶点的位置
		DeviceBufferArray<int> triNums;									// 【论文参数】三角形数量
		DeviceBufferArray<int> triAddress;								// 【论文参数】三角形位置
		DeviceBufferArray<int> cubeCatagory;							// 记录这是属于哪一种类型的立方体，方便使用Marching Cubes算法构建网格

		DeviceBufferArray<bool> markValidVertex;						// 标记有效的顶点的位置

		DeviceBufferArray<bool> markValidSubdividedNode;				// 标记节点是否可以被细分优化

		//CoredVectorMeshData mesh;										// 三角形网格

		DeviceBufferArray<OctNode> SubdivideNode;						// 细分节点，将生成的三角剖分细分

		int SubdivideDepthCount[Constants::maxDepth_Host] = { 0 };		// 细分节点每一层节点的数量
		int SubdivideDepthAddress[Constants::maxDepth_Host] = { 0 };	// 细分节点每层在SubdivideNode的偏移

		int SubdivideNodeNumHost = 0;									// 细分节点的个数
		SynchronizeArray<int> SubdivideDepthBuffer;						// 记录SubdivideNode数组细分节点的深度

		// 细化节点的层：[0, finerDepth)是Coarser层, [finerDepth, maxDepth]是Finer层
		const unsigned int finerDepth = MAX_DEPTH_OCTREE - COARSER_DIVERGENCE_LEVEL_NUM;	
		int fixedDepthNodeNum[Constants::maxDepth_Host + 1] = { 0 };	// 每层节点数量设置固定大小【用于Coarser层细分】
		int fixedDepthNodeAddress[Constants::maxDepth_Host + 1] = { 0 };// 每层节点偏移【用于Coarser层细分】
		int depthNodeCount[Constants::maxDepth_Host + 1] = { 0 };;		// 每层节点的数量【用于Finer层细分】
		int depthNodeAddress[Constants::maxDepth_Host + 1] = { 0 };;	// 每层节点的偏移【用于Finer层细分】

		DeviceBufferArray<bool> markValidSubdivideVertex;				// 细分节点中标记有效的顶点【用于Coarser层细分】
		DeviceBufferArray<bool> markValidSubdivideEdge;					// 细分节点中标记有效的边【用于Coarser层细分】
		DeviceBufferArray<bool> markValidSubdivedeVexNum;				// 细分节点中标记有效的vexNums【用于Coarser层细分】

		DeviceBufferArray<bool> markValidFinerVexArray;
		DeviceBufferArray<bool> markValidFinerEdge;
		DeviceBufferArray<bool> markValidFinerVexNum;
		/**
		 * \brief 计算顶点vertex的隐式函数值.
		 *
		 * \param VertexArray 顶点数组
		 * \param NodeArray 节点数组
		 * \param BaseFunction 基函数
		 * \param dx 散度
		 * \param encodeNodeIndexInFunction 基函数索引
		 * \param isoValue 等值
		 * \param stream cuda流
		 */
		void ComputeVertexImplicitFunctionValue(DeviceArrayView<VertexNode> VertexArray, DeviceArrayView<OctNode> NodeArray, DeviceArrayView<ConfirmedPPolynomial<CONVTIMES + 1, CONVTIMES + 2>> BaseFunction, DeviceArrayView<float> dx, DeviceArrayView<int> encodeNodeIndexInFunction, const float isoValue, cudaStream_t stream);

		/**
		 * \brief 生成顶点的vertexNums和顶点的vertexAddress.
		 *
		 * \param EdgeArray 边数组
		 * \param NodeArray 节点数组
		 * \param vvalue 顶点隐式函数值
		 * \param stream cuda流
		 */
		void generateVertexNumsAndVertexAddress(DeviceArrayView<EdgeNode> EdgeArray, DeviceArrayView<OctNode> NodeArray, DeviceArrayView<float> vvalue, const unsigned int DLevelOffset, cudaStream_t stream);

		/**
		 * \brief 生成TriNums和TriAddress，以及立方体类型cubeCatagory.
		 *
		 * \param NodeArray 节点数组
		 * \param vvalue 顶点隐式函数值
		 * \param DLevelOffset 第maxDepth层的首节点在NodeArray中偏移
		 * \param DLevelNodeCount 第maxDepth层节点数量
		 * \param stream cuda流
		 */
		void generateTriangleNumsAndTriangleAddress(DeviceArrayView<OctNode> NodeArray, DeviceArrayView<float> vvalue, const unsigned int DLevelOffset, const unsigned int DLevelNodeCount, CoredVectorMeshData& mesh, cudaStream_t stream);

		/**
		 * \brief 生成顶点以及三角形.
		 * 
		 * \param VertexArray 顶点数组
		 * \param EdgeArray 边数组
		 * \param FaceArray 面数组
		 * \param DLevelOffset 第maxDepth层的首节点在NodeArray中偏移
		 * \param DLevelNodeCount 第maxDepth层节点数量
		 * \param stream cuda流
		 */
		void generateVerticesAndTriangle(DeviceBufferArray<OctNode>& NodeArray, DeviceArrayView<VertexNode> VertexArray, DeviceArrayView<EdgeNode> EdgeArray, DeviceArrayView<FaceNode> FaceArray, const unsigned int DLevelOffset, const unsigned int DLevelNodeCount, CoredVectorMeshData& mesh, cudaStream_t stream);

		/**
		 * \brief 插入三角形.
		 *
		 * \param VertexBuffer 顶点数组
		 * \param allVexNums 所有有效顶点数量
		 * \param TriangleBuffer 三角形数组
		 * \param allTriNums 所有有效三角形数量
		 * \param mesh 网格
		 */
		void insertTriangle(const Point3D<float>* VertexBufferHost, const int& allVexNums, const int* TriangleBufferHost, const int& allTriNums, CoredVectorMeshData& mesh);
	
	
		/**
		 * \brief 生成细分节点的数组以及不同层细分节点的数量和偏移【GPU硬件限制，无法使用流异步操作，需要Share Memory > 64kb的GPU】.
		 * 
		 * \param NodeArray 节点数组
		 * \param DepthBuffer 节点数组NodeArray的深度查找数组
		 * \param OtherDepthNodeCount 第[0, maxDepth - 1]层的首节点在NodeArray中偏移
		 * \param stream cuda流
		 */
		void generateSubdivideNodeArrayCountAndAddress(DeviceBufferArray<OctNode>& NodeArray, DeviceArrayView<unsigned int> DepthBuffer, const unsigned int OtherDepthNodeCount, cudaStream_t stream);

		/**
		 * \brief 粗节点细分并重构网格, 异步速度快了69.47%，原先代码147.503ms，当前代码45.0349ms【似乎可以与Finer并行，开两个线程】.
		 * 
		 * \param NodeArray 节点数组
		 * \param DepthBuffer 节点深度数组
		 * \param CenterBuffer 节点中心位置数组
		 * \param BaseFunction 基函数
		 * \param dx 节点
		 * \param encodeNodeIndexInFunction 基函数节点编码
		 * \param isoValue 等值
		 * \param stream cuda流
		 */
		void CoarserSubdivideNodeAndRebuildMesh(DeviceBufferArray<OctNode>& NodeArray, DeviceArrayView<unsigned int> DepthBuffer, DeviceArrayView<Point3D<float>> CenterBuffer, DeviceArrayView<ConfirmedPPolynomial<CONVTIMES + 1, CONVTIMES + 2>> BaseFunction, DeviceArrayView<float> dx, DeviceArrayView<int> encodeNodeIndexInFunction, const float isoValue, CoredVectorMeshData& mesh, cudaStream_t stream);

		/**
		 * \brief 精节点细分并重构网格【似乎可以与Coarser并行，开两个线程】.
		 *
		 * \param NodeArray 节点数组
		 * \param DepthBuffer 节点深度数组
		 * \param CenterBuffer 节点中心位置数组
		 * \param BaseFunction 基函数
		 * \param dx 节点
		 * \param encodeNodeIndexInFunction 基函数节点编码
		 * \param isoValue 等值
		 * \param stream cuda流
		 */
		void FinerSubdivideNodeAndRebuildMesh(DeviceBufferArray<OctNode>& NodeArray, DeviceArrayView<unsigned int> DepthBuffer, DeviceArrayView<Point3D<float>> CenterBuffer, DeviceArrayView<ConfirmedPPolynomial<CONVTIMES + 1, CONVTIMES + 2>> BaseFunction, DeviceArrayView<float> dx, DeviceArrayView<int> encodeNodeIndexInFunction, const float isoValue, CoredVectorMeshData& mesh, cudaStream_t stream);


	};
}



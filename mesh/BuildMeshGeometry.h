/*****************************************************************//**
 * \file   BuildMeshGeometry.h
 * \brief  构建网格算法
 * 
 * \author LUOJIAXUAN
 * \date   June 1st 2024
 *********************************************************************/
#pragma once
#include <vector>
#include <base/DeviceAPI/safe_call.hpp>
#include <base/DeviceReadWrite/DeviceBufferArray.h>
#include <mesh/OctNode.cuh>
namespace SparseSurfelFusion {
	namespace device {
		/**
		 * \brief 初始化顶点的Owner.
		 *
		 * \param NodeArray 八叉树节点数组
		 * \param NodeArraySize 八叉树节点数组大小
		 * \param DepthBuffer 节点数组NodeArray深度查询表
		 * \param CenterBuffer 节点数组NodeArray节点中心查询表
		 * \param preVertexArray 预处理顶点数组计算
		 * \param markPreVertexArray 标记有效的vertex的索引
		 */
		__global__ void initVertexOwner(DeviceArrayView<OctNode> NodeArray, const unsigned int NodeArraySize, DeviceArrayView<unsigned int> depthBuffer, DeviceArrayView<Point3D<float>> centerBuffer, VertexNode* preVertexArray, bool* markPreVertexArray);

		/**
		 * \brief 计算两点之间距离平方.
		 *
		 * \param p1 点1
		 * \param p2 点2
		 * \return 返回两点之间距离平方
		 */
		__forceinline__ __device__ double SquareDistance(const Point3D<float>& p1, const Point3D<float>& p2);

		/**
		 * \brief 生成VertexArray并且将NodeArray中的vertice[8]参数赋值.
		 * 
		 * \param DepthBuffer 节点数组NodeArray深度查询表
		 * \param CenterBuffer 节点数组NodeArray节点中心查询表
		 * \param VertexArraySize 顶点数组的数量
		 * \param VertexArray 顶点数组【可写入】
		 * \param NodeArray 八叉树节点数组
		 */
		__global__ void maintainVertexNodePointer(DeviceArrayView<unsigned int> DepthBuffer, DeviceArrayView<Point3D<float>> CenterBuffer, const unsigned int VertexArraySize, VertexNode* VertexArray, OctNode* NodeArray);

		/**
		 * \brief 初始化边数组.
		 * 
		 * \param NodeArray 八叉树节点数组
		 * \param DLevelOffset maxDepth层节点的偏移
		 * \param DLevelNodeCount maxDepth层节点数量
		 * \param DepthBuffer 节点数组NodeArray深度查询表
		 * \param CenterBuffer 节点数组NodeArray节点中心查询表
		 * \param preEdgeArray 预处理的边数组
		 * \param markPreEdgeArray 标记有效的Edge索引
		 */
		__global__ void initEdgeArray(DeviceArrayView<OctNode> NodeArray, const unsigned int DLevelOffset, const unsigned int DLevelNodeCount, DeviceArrayView<unsigned int> DepthBuffer, DeviceArrayView<Point3D<float>> CenterBuffer, EdgeNode* preEdgeArray, bool* markPreEdgeArray);

		/**
		 * \brief 生成EdgeArray并且将NodeArray中的edges[12]参数赋值.
		 * 
		 * \param DepthBuffer 节点数组NodeArray深度查询表
		 * \param CenterBuffer 节点数组NodeArray节点中心查询表
		 * \param EdgeArraySize 边数组数量
		 * \param EdgeArray 有效边数组
		 * \param NodeArray 节点数组【可写入】
		 */
		__global__ void maintainEdgeNodePointer(DeviceArrayView<unsigned int> DepthBuffer, DeviceArrayView<Point3D<float>> CenterBuffer, const unsigned int EdgeArraySize, EdgeNode* EdgeArray, OctNode* NodeArray);

		/**
		 * \brief 初始化面数组.
		 * 
		 * \param NodeArray 八叉树节点数组【只读】
		 * \param DepthBuffer 节点数组NodeArray深度查询表
		 * \param CenterBuffer 节点数组NodeArray节点中心查询表
		 * \param NodeArraySize 八叉树节点数组大小
		 * \param preFaceArray 预处理的面数组
		 */
		__global__ void initFaceArray(DeviceArrayView<OctNode> NodeArray, DeviceArrayView<unsigned int> DepthBuffer, DeviceArrayView<Point3D<float>> CenterBuffer, const unsigned int NodeArraySize, FaceNode* preFaceArray, bool* markPreFaceArray);

		/**
		 * \brief 生成FaceArray并且将NodeArray中的faces[6]参数赋值.
		 * 
		 * \param DepthBuffer 节点数组NodeArray深度查询表
		 * \param CenterBuffer 节点数组NodeArray节点中心查询表
		 * \param FaceArraySize 面数组数量
		 * \param FaceArray 有效面数组
		 * \param NodeArray 节点数组【可写入】
		 */
		__global__ void maintainFaceNodePointer(DeviceArrayView<unsigned int> DepthBuffer, DeviceArrayView<Point3D<float>> CenterBuffer, const unsigned int FaceArraySize, OctNode* NodeArray, FaceNode* FaceArray);
	}

	class BuildMeshGeometry
	{
	public:
		BuildMeshGeometry();

		~BuildMeshGeometry();

		using Ptr = std::shared_ptr<BuildMeshGeometry>;

		/**
		 * \brief 获得顶点数组.
		 * 
		 * \return 返回顶点数组
		 */
		DeviceArrayView<VertexNode> GetVertexArray() { return VertexArray.ArrayView(); }

		/**
		 * \brief 获得边数组.
		 * 
		 * \return 返回边数组
		 */
		DeviceArrayView<EdgeNode> GetEdgeArray() { return EdgeArray.ArrayView(); }

		/**
		 * \brief 获得面数组.
		 *
		 * \return 返回面数组
		 */
		DeviceArrayView<FaceNode> GetFaceArray() { return FaceArray.ArrayView(); }
		/**
		 * \brief 计算顶点vertex数组，并将NodeArray中的vertices数组中记录对应在VertexArray中的index，以供后续调用.
		 * 
		 * \param NodeArray 节点数组
		 * \param NodeArrayDepthIndex 节点深度查询数组
		 * \param NodeArrayNodeCenter 节点中心查询数组
		 * \param stream cuda流
		 */
		void GenerateVertexArray(DeviceBufferArray<OctNode>& NodeArray, DeviceArrayView<unsigned int> NodeArrayDepthIndex, DeviceArrayView<Point3D<float>> NodeArrayNodeCenter, cudaStream_t stream);

		/**
		 * \brief 计算边edge数组，并将NodeArray中的edges数组中记录对应在EdgeArray中的index，以供后续调用.
		 * 
		 * \param NodeArray 节点数组
		 * \param DLevelOffset maxDepth层偏移
		 * \param DLevelNodeCount maxDepth层节点数量
		 * \param NodeArrayDepthIndex 节点深度查询数组
		 * \param NodeArrayNodeCenter 节点中心查询数组
		 * \param stream cuda流
		 */
		void GenerateEdgeArray(DeviceBufferArray<OctNode>& NodeArray, const unsigned int DLevelOffset, const unsigned int DLevelNodeCount, DeviceArrayView<unsigned int> NodeArrayDepthIndex, DeviceArrayView<Point3D<float>> NodeArrayNodeCenter, cudaStream_t stream);

		/**
		 * \brief 计算顶点face数组，并将NodeArray中的faces数组中记录对应在FaceArray中的index，以供后续调用.
		 *
		 * \param NodeArray 节点数组
		 * \param NodeArrayDepthIndex 节点深度查询数组
		 * \param NodeArrayNodeCenter 节点中心查询数组
		 * \param stream cuda流
		 */
		void GenerateFaceArray(DeviceBufferArray<OctNode>& NodeArray, DeviceArrayView<unsigned int> NodeArrayDepthIndex, DeviceArrayView<Point3D<float>> NodeArrayNodeCenter, cudaStream_t stream);

	private:
		DeviceBufferArray<VertexNode> VertexArray;			// 顶点数组
		DeviceBufferArray<EdgeNode> EdgeArray;				// 边数组
		DeviceBufferArray<FaceNode> FaceArray;				// 面数组

		DeviceBufferArray<bool> markValidVertexArray;		// 记录有效的顶点位置，即ownerNodeIdx > 0找到了拥有者节点的vertex 
		DeviceBufferArray<bool> markValidEdgeArray;			// 记录有效边
		DeviceBufferArray<bool> markValidFaceArray;			// 记录有效面
	};
}



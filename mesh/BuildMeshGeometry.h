/*****************************************************************//**
 * \file   BuildMeshGeometry.h
 * \brief  ���������㷨
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
		 * \brief ��ʼ�������Owner.
		 *
		 * \param NodeArray �˲����ڵ�����
		 * \param NodeArraySize �˲����ڵ������С
		 * \param DepthBuffer �ڵ�����NodeArray��Ȳ�ѯ��
		 * \param CenterBuffer �ڵ�����NodeArray�ڵ����Ĳ�ѯ��
		 * \param preVertexArray Ԥ�������������
		 * \param markPreVertexArray �����Ч��vertex������
		 */
		__global__ void initVertexOwner(DeviceArrayView<OctNode> NodeArray, const unsigned int NodeArraySize, DeviceArrayView<unsigned int> depthBuffer, DeviceArrayView<Point3D<float>> centerBuffer, VertexNode* preVertexArray, bool* markPreVertexArray);

		/**
		 * \brief ��������֮�����ƽ��.
		 *
		 * \param p1 ��1
		 * \param p2 ��2
		 * \return ��������֮�����ƽ��
		 */
		__forceinline__ __device__ double SquareDistance(const Point3D<float>& p1, const Point3D<float>& p2);

		/**
		 * \brief ����VertexArray���ҽ�NodeArray�е�vertice[8]������ֵ.
		 * 
		 * \param DepthBuffer �ڵ�����NodeArray��Ȳ�ѯ��
		 * \param CenterBuffer �ڵ�����NodeArray�ڵ����Ĳ�ѯ��
		 * \param VertexArraySize �������������
		 * \param VertexArray �������顾��д�롿
		 * \param NodeArray �˲����ڵ�����
		 */
		__global__ void maintainVertexNodePointer(DeviceArrayView<unsigned int> DepthBuffer, DeviceArrayView<Point3D<float>> CenterBuffer, const unsigned int VertexArraySize, VertexNode* VertexArray, OctNode* NodeArray);

		/**
		 * \brief ��ʼ��������.
		 * 
		 * \param NodeArray �˲����ڵ�����
		 * \param DLevelOffset maxDepth��ڵ��ƫ��
		 * \param DLevelNodeCount maxDepth��ڵ�����
		 * \param DepthBuffer �ڵ�����NodeArray��Ȳ�ѯ��
		 * \param CenterBuffer �ڵ�����NodeArray�ڵ����Ĳ�ѯ��
		 * \param preEdgeArray Ԥ����ı�����
		 * \param markPreEdgeArray �����Ч��Edge����
		 */
		__global__ void initEdgeArray(DeviceArrayView<OctNode> NodeArray, const unsigned int DLevelOffset, const unsigned int DLevelNodeCount, DeviceArrayView<unsigned int> DepthBuffer, DeviceArrayView<Point3D<float>> CenterBuffer, EdgeNode* preEdgeArray, bool* markPreEdgeArray);

		/**
		 * \brief ����EdgeArray���ҽ�NodeArray�е�edges[12]������ֵ.
		 * 
		 * \param DepthBuffer �ڵ�����NodeArray��Ȳ�ѯ��
		 * \param CenterBuffer �ڵ�����NodeArray�ڵ����Ĳ�ѯ��
		 * \param EdgeArraySize ����������
		 * \param EdgeArray ��Ч������
		 * \param NodeArray �ڵ����顾��д�롿
		 */
		__global__ void maintainEdgeNodePointer(DeviceArrayView<unsigned int> DepthBuffer, DeviceArrayView<Point3D<float>> CenterBuffer, const unsigned int EdgeArraySize, EdgeNode* EdgeArray, OctNode* NodeArray);

		/**
		 * \brief ��ʼ��������.
		 * 
		 * \param NodeArray �˲����ڵ����顾ֻ����
		 * \param DepthBuffer �ڵ�����NodeArray��Ȳ�ѯ��
		 * \param CenterBuffer �ڵ�����NodeArray�ڵ����Ĳ�ѯ��
		 * \param NodeArraySize �˲����ڵ������С
		 * \param preFaceArray Ԥ�����������
		 */
		__global__ void initFaceArray(DeviceArrayView<OctNode> NodeArray, DeviceArrayView<unsigned int> DepthBuffer, DeviceArrayView<Point3D<float>> CenterBuffer, const unsigned int NodeArraySize, FaceNode* preFaceArray, bool* markPreFaceArray);

		/**
		 * \brief ����FaceArray���ҽ�NodeArray�е�faces[6]������ֵ.
		 * 
		 * \param DepthBuffer �ڵ�����NodeArray��Ȳ�ѯ��
		 * \param CenterBuffer �ڵ�����NodeArray�ڵ����Ĳ�ѯ��
		 * \param FaceArraySize ����������
		 * \param FaceArray ��Ч������
		 * \param NodeArray �ڵ����顾��д�롿
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
		 * \brief ��ö�������.
		 * 
		 * \return ���ض�������
		 */
		DeviceArrayView<VertexNode> GetVertexArray() { return VertexArray.ArrayView(); }

		/**
		 * \brief ��ñ�����.
		 * 
		 * \return ���ر�����
		 */
		DeviceArrayView<EdgeNode> GetEdgeArray() { return EdgeArray.ArrayView(); }

		/**
		 * \brief ���������.
		 *
		 * \return ����������
		 */
		DeviceArrayView<FaceNode> GetFaceArray() { return FaceArray.ArrayView(); }
		/**
		 * \brief ���㶥��vertex���飬����NodeArray�е�vertices�����м�¼��Ӧ��VertexArray�е�index���Թ���������.
		 * 
		 * \param NodeArray �ڵ�����
		 * \param NodeArrayDepthIndex �ڵ���Ȳ�ѯ����
		 * \param NodeArrayNodeCenter �ڵ����Ĳ�ѯ����
		 * \param stream cuda��
		 */
		void GenerateVertexArray(DeviceBufferArray<OctNode>& NodeArray, DeviceArrayView<unsigned int> NodeArrayDepthIndex, DeviceArrayView<Point3D<float>> NodeArrayNodeCenter, cudaStream_t stream);

		/**
		 * \brief �����edge���飬����NodeArray�е�edges�����м�¼��Ӧ��EdgeArray�е�index���Թ���������.
		 * 
		 * \param NodeArray �ڵ�����
		 * \param DLevelOffset maxDepth��ƫ��
		 * \param DLevelNodeCount maxDepth��ڵ�����
		 * \param NodeArrayDepthIndex �ڵ���Ȳ�ѯ����
		 * \param NodeArrayNodeCenter �ڵ����Ĳ�ѯ����
		 * \param stream cuda��
		 */
		void GenerateEdgeArray(DeviceBufferArray<OctNode>& NodeArray, const unsigned int DLevelOffset, const unsigned int DLevelNodeCount, DeviceArrayView<unsigned int> NodeArrayDepthIndex, DeviceArrayView<Point3D<float>> NodeArrayNodeCenter, cudaStream_t stream);

		/**
		 * \brief ���㶥��face���飬����NodeArray�е�faces�����м�¼��Ӧ��FaceArray�е�index���Թ���������.
		 *
		 * \param NodeArray �ڵ�����
		 * \param NodeArrayDepthIndex �ڵ���Ȳ�ѯ����
		 * \param NodeArrayNodeCenter �ڵ����Ĳ�ѯ����
		 * \param stream cuda��
		 */
		void GenerateFaceArray(DeviceBufferArray<OctNode>& NodeArray, DeviceArrayView<unsigned int> NodeArrayDepthIndex, DeviceArrayView<Point3D<float>> NodeArrayNodeCenter, cudaStream_t stream);

	private:
		DeviceBufferArray<VertexNode> VertexArray;			// ��������
		DeviceBufferArray<EdgeNode> EdgeArray;				// ������
		DeviceBufferArray<FaceNode> FaceArray;				// ������

		DeviceBufferArray<bool> markValidVertexArray;		// ��¼��Ч�Ķ���λ�ã���ownerNodeIdx > 0�ҵ���ӵ���߽ڵ��vertex 
		DeviceBufferArray<bool> markValidEdgeArray;			// ��¼��Ч��
		DeviceBufferArray<bool> markValidFaceArray;			// ��¼��Ч��
	};
}



/*****************************************************************//**
 * \file   ComputeTriangleIndices.h
 * \brief  �����޸��������񣬹�������
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
		 * \brief ���㶥��vertex��ʽ�����˺���.
		 * 
		 * \param VertexArray ��������
		 * \param NodeArray �ڵ�����
		 * \param BaseFunctions ������
		 * \param dx ɢ��
		 * \param encodeNodeIndexInFunction ����������
		 * \param isoValue ��ֵ
		 * \param vvalue ����������ֵ
		 */
		__global__ void ComputeVertexImplicitFunctionValueKernel(DeviceArrayView<VertexNode> VertexArray, DeviceArrayView<OctNode> NodeArray, DeviceArrayView<ConfirmedPPolynomial<CONVTIMES + 1, CONVTIMES + 2>> BaseFunctions, DeviceArrayView<float> dx, DeviceArrayView<int> encodeNodeIndexInFunction, const unsigned int VertexArraySize, const float isoValue, float* vvalue);
	
		/**
		 * \brief ���ɶ����vertexNums�Ͷ����vertexAddress�ĺ˺���.
		 *
		 * \param EdgeArray ������
		 * \param NodeArray �ڵ�����
		 * \param vvalue ������ʽ����ֵ
		 * \param vexNums ��������ڵ�λ��VertexAddress
		 */
		__global__ void generateVertexNumsKernel(DeviceArrayView<EdgeNode> EdgeArray, DeviceArrayView<OctNode> NodeArray, DeviceArrayView<float> vvalue, const unsigned int EdgeArraySize, int* vexNums, bool* markValidVertex);

		/**
		 * \brief ����TriangleNums�Լ�cubeCatagory�˺���.
		 * 
		 * \param NodeArray �ڵ�����
		 * \param vvalue ������ʽ����ֵ
		 * \param DLevelOffset ��maxDepth����׽ڵ���NodeArray��ƫ��
		 * \param DLevelNodeCount ��maxDepth��ڵ�����
		 * \param triNums ����������
		 * \param cubeCatagory ����������
		 */
		__global__ void generateTriangleNumsKernel(DeviceArrayView<OctNode> NodeArray, DeviceArrayView<float> vvalue, const unsigned int DLevelOffset, const unsigned int DLevelNodeCount, int* triNums, int* cubeCatagory);

		/**
		 * \brief ���ɽ����.
		 * 
		 * \param NodeArray �ڵ�����
		 * \param validEdgeArray ɸѡ����Ч�ı�����
		 * \param VertexArray ��������
		 * \param vvalue ������ʽ����ֵ
		 * \param validVexAddress ��Ч�����λ��
		 * \param validEdgeArraySize ��Ч������Ĵ�С
		 * \param VertexBuffer ��Ч�Ķ���
		 */
		__global__ void generateIntersectionPoint(DeviceArrayView<OctNode> NodeArray, DeviceArrayView<VertexNode> VertexArray, DeviceArrayView<float> vvalue, const EdgeNode* validEdgeArray, const int* validVexAddress, const unsigned int validEdgeArraySize, Point3D<float>* VertexBuffer);
		
		/**
		 * \brief ��������֮��Ĳ����.
		 * 
		 * \param p1 ��1
		 * \param p2 ��2
		 * \param dim ��Ҫ���в����ά��
		 * \param v1 ��ʽ����ֵ1
		 * \param v2 ��ʽ����ֵ2
		 * \param out �����롿����������
		 */
		__device__ void interpolatePoint(const Point3D<float>& p1, const Point3D<float>& p2, const int& dim, const float& v1, const float& v2, Point3D<float>& out);

		/**
		 * \brief ����������λ���Լ��������Ƿ��ཻ������.
		 * 
		 * \param NodeArray �ڵ�����
		 * \param FaceArray ������
		 * \param DLevelOffset ��maxDepth����׽ڵ���NodeArray��ƫ��
		 * \param DLevelNodeCount ��maxDepth��ڵ�����
		 * \param triNums ����������
		 * \param cubeCatagory ����������
		 * \param vexAddress ����ƫ�Ƶ�ַ
		 * \param triAddress ������ƫ�Ƶ�ַ
		 * \param TriangleBuffer ��¼�����ɵ�������
		 */
		__global__ void generateTrianglePos(DeviceArrayView<OctNode> NodeArray, DeviceArrayView<FaceNode> FaceArray, DeviceArrayView<int> triNums, DeviceArrayView<int> cubeCatagory, DeviceArrayView<int> vexAddress, DeviceArrayView<int> triAddress, const unsigned int DLevelOffset, const unsigned int DLevelNodeCount, int* TriangleBuffer, int* hasSurfaceIntersection);

		/**
		 * \brief ���ϸ��������λ��.
		 */
		__global__ void generateSubdivideTrianglePos(const EasyOctNode* SubdivideArray, const unsigned int DLevelOffset, const unsigned int DLevelNodeCount, const int* SubdivideTriNums, const int* SubdivideCubeCatagory, const int* SubdivideVexAddress, const int* SubdivideTriAddress, int* SubdivideTriangleBuffer);

		/**
		 * \brief ���������㼶��Ҷ�ӽڵ�������κ��ཻ���, ��¼��NodeArray��.
		 * 
		 * \param VertexArray ��������
		 * \param vvalue ������ʽ����ֵ
		 * \param OtherDepthNodeCount ��[0, maxDepth - 1]����׽ڵ���NodeArray��ƫ��
		 * \param hasSurfaceIntersection �Ƿ����ཻ
		 * \param NodeArray �ڵ�����
		 * \param markValidSubdividedNode ��ǿ���ϸ�ֵĽڵ�idx
		 */
		__global__ void ProcessLeafNodesAtOtherDepth(DeviceArrayView<VertexNode> VertexArray, DeviceArrayView<float> vvalue, const unsigned int OtherDepthNodeCount, const int* hasSurfaceIntersection, OctNode* NodeArray, bool* markValidSubdividedNode);
	
		/**
		 * \brief ����ϸ�ֽڵ����ڵ���ȣ��Լ�ÿһ��һ������ϸ�ֽڵ�.
		 * 
		 * \param SubdivideNode ϸ�ֽڵ�����
		 * \param DepthBuffer ��¼NodeArray��Ƚڵ������
		 * \param SubdivideNum ϸ�ֽڵ������
		 * \param SubdivideDepthBuffer ��¼ϸ�ֽڵ������������
		 * \param SubdivideDepthNum ��¼��ǰ���ϸ�ֽڵ������
		 */
		__global__ void precomputeSubdivideDepth(DeviceArrayView<OctNode> SubdivideNode, DeviceArrayView<unsigned int> DepthBuffer, const int SubdivideNum, int* SubdivideDepthBuffer, int* SubdivideDepthNum);
	
		/**
		 * \brief �ع�NodeArray����¼��SubdivideArray��.
		 * 
		 * \param SubdivideNode ϸ�ֽڵ�
		 * \param SubdivideDepthBuffer ϸ�ֽڵ����
		 * \param iterRound ��������
		 * \param NodeArraySize NodeArray����ĳ���
		 * \param SubdivideArraySize ϸ�ֽڵ�����ĳ���
		 * \param SubdivideArray ��¼�ع�������
		 * \param SubdivideArrayDepthBuffer �ع����ݽڵ����
		 * \param SubdivideArrayCenterBuffer �ع����ݽڵ����ĵ�λ��
		 */
		__global__ void singleRebuildArray(DeviceArrayView<OctNode> SubdivideNode, DeviceArrayView<int> SubdivideDepthBuffer, const unsigned int iterRound, const unsigned int NodeArraySize, const unsigned int SubdivideArraySize, EasyOctNode* SubdivideArray, int* SubdivideArrayDepthBuffer, Point3D<float>* SubdivideArrayCenterBuffer);

		/**
		 * \brief ���ϸ�ֽڵ�����.
		 * 
		 * \param rootDepth ϸ�ֽڵ�������
		 * \param idx ϸ�ֽڵ��index
		 */
		__device__ int getSubdivideDepth(const int& rootDepth, const int& idx);

		/**
		 * \brief ��õ�ǰ�ڵ�����ĵ�.
		 * 
		 * \param key ��ǰ�ڵ��key��ֵ
		 * \param currentDepth ��ǰ���
		 * \param center ���������ǰ�ڵ�����ĵ�
		 */
		__device__ void getNodeCenterAllDepth(const int& key, const int& currentDepth, Point3D<float>& center);

		/**
		 * \brief �����ع��Ľڵ��ھ�.
		 * 
		 * \param NodeArray ԭʼ�ڵ�����
		 * \param currentLevelOffset ��ǰ��ڵ��ƫ��
		 * \param currentLevelNodesCount ��ǰ��ڵ������
		 * \param NodeArraySize ԭʼ�ڵ�����
		 * \param depth ��ǰ���
		 * \param SubdivideArray ϸ�ֽڵ�����
		 */
		__global__ void computeRebuildNeighbor(DeviceArrayView<OctNode> NodeArray, const unsigned int currentLevelOffset, const unsigned int currentLevelNodesCount, const unsigned int NodeArraySize, const unsigned int depth, EasyOctNode* SubdivideArray);
	
		/**
		 * \brief ��ʼ��ϸ�ֶ������ӵ�е�ϸ�ֽڵ�.
		 * 
		 * \param SubdivideArray ϸ�ֽڵ�����
		 * \param SubdivideArrayCenterBuffer ϸ�ֽڵ����ĵ�����
		 * \param currentLevelOffset ��ǰ��ڵ��ƫ��
		 * \param currentLevelNodesCount ��ǰ��ڵ������
		 * \param NodeArraySize ԭʼ�ڵ�����
		 * \param SubdividePreVertexArray Ԥ����ϸ�ֽڵ㣬���������Ч��VertexArray
		 */
		__global__ void initSubdivideVertexOwner(const EasyOctNode* SubdivideArray, const Point3D<float>* SubdivideArrayCenterBuffer, const unsigned int currentLevelOffset, const unsigned int currentLevelNodesCount, const unsigned int NodeArraySize, VertexNode* SubdividePreVertexArray, bool* markValidSubdivideVertex);

		/**
		 * \brief ��������֮�����ƽ��.
		 *
		 * \param p1 ��1
		 * \param p2 ��2
		 * \return ��������֮�����ƽ��
		 */
		__forceinline__ __device__ double SquareDistance(const Point3D<float>& p1, const Point3D<float>& p2);

		/**
		 * \brief ����SubdivideArray�еĶ���.
		 * 
		 * \param CenterBuffer �ڵ������λ��
		 * \param VertexArraySize ��������
		 * \param NodeArraySize �ڵ�����
		 * \param SubdivideArrayCenterBuffer ϸ�ֽڵ������λ��
		 * \param VertexArray ��������
		 * \param SubdivideArray ϸ������
		 */
		__global__ void maintainSubdivideVertexNodePointer(DeviceArrayView<Point3D<float>> CenterBuffer, const unsigned int VertexArraySize, const unsigned int NodeArraySize, const Point3D<float>* SubdivideArrayCenterBuffer, VertexNode* VertexArray, EasyOctNode* SubdivideArray);

		/**
		 * \brief ��ʼ��ϸ�ֶ�������б�.
		 * 
		 * \param SubdivideArray ϸ������
		 * \param SubdivideArrayCenterBuffer ϸ�ֽڵ������λ��
		 * \param NodeArraySize �ڵ�����
		 * \param DLevelOffset ����ƫ��
		 * \param DLevelNodeCount ����ڵ�����
		 * \param SubdividePreEdgeArray Ԥ��������飬������ɸ����Ч��
		 * \param markValidSubdivideEdge ����Ч�������
		 */ 
		__global__ void initSubdivideEdgeArray(const EasyOctNode* SubdivideArray, const Point3D<float>* SubdivideArrayCenterBuffer, const unsigned int NodeArraySize, const unsigned int DLevelOffset, const unsigned int DLevelNodeCount, EdgeNode* SubdividePreEdgeArray, bool* markValidSubdivideEdge);

		/**
		 * \brief ά������ϸ�ֱߵĲ���.
		 * 
		 * \param CenterBuffer �ڵ������λ��
		 * \param SubdivideArrayCenterBuffer ϸ�ֽڵ������λ��
		 * \param EdgeArraySize ������Ĵ�С
		 * \param NodeArraySize �ڵ�����
		 * \param SubdivideArray ϸ������
		 * \param EdgeArray ϸ�ֱ�����
		 */
		__global__ void maintainSubdivideEdgeNodePointer(DeviceArrayView<Point3D<float>> CenterBuffer, const Point3D<float>* SubdivideArrayCenterBuffer, const unsigned int EdgeArraySize, const unsigned int NodeArraySize, EasyOctNode* SubdivideArray, EdgeNode* EdgeArray);
		
		/**
		 * \brief ����ϸ�ֶ������ʽ����ֵ.
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
		 * \brief ����ϸ�ֶ������ʽ����ֵ��Finer��.
		 */
		__global__ void computeSubdivideVertexImplicitFunctionValue(const VertexNode* SubdivideVertexArray, const EasyOctNode* SubdivideArray, DeviceArrayView<OctNode> NodeArray, DeviceArrayView<float> dx, DeviceArrayView<int> EncodedNodeIdxInFunction, DeviceArrayView<ConfirmedPPolynomial<CONVTIMES + 1, CONVTIMES + 2>> baseFunctions, const unsigned int NodeArraySize, const int* ReplacedNodeId, const int* IsRoot, const unsigned int SubdivideVertexArraySize, const float isoValue, float* SubdivideVvalue);
			 
			 
			 
			
			

		/**
		 * \brief ����ϸ�ֶ����������Ӷ����Address.
		 * 
		 * \param SubdivideEdgeArray ϸ�ֱ�
		 * \param SubdivideArray ϸ������
		 * \param SubdivideEdgeArraySize ϸ�ֱ�����Ĵ�С
		 * \param NodeArraySize �ڵ�����
		 * \param SubdivideVvalue �ڵ����ʽ����ֵ
		 * \param markValidSubdivedeVexNum �����Ч��ϸ��vexNum��index
		 */
		__global__ void generateSubdivideVexNums(const EdgeNode* SubdivideEdgeArray, const EasyOctNode* SubdivideArray, const unsigned int SubdivideEdgeArraySize, const unsigned int NodeArraySize, const float* SubdivideVvalue, int* SubdivideVexNums, bool* markValidSubdivedeVexNum);

		/**
		 * \brief ����ϸ�ֽڵ��������.
		 * 
		 * \param SubdivideNodeArray ϸ�ֽڵ����飬��ǰ���NodeArray��һ��
		 * \param DLevelOffset maxDepth����׽ڵ���SubdivideNodeArray�е�λ��
		 * \param DLevelNodeCount maxDepth��Ľڵ�����
		 * \param vvalue ������ֵ
		 * \param triNums ����������
		 * \param cubeCatagory ����������
		 */
		__global__ void generateTriNums(const EasyOctNode* SubdivideNodeArray, const unsigned int DLevelOffset, const unsigned int DLevelNodeCount, const float* vvalue, int* triNums, int* cubeCatagory);

		
		/**
		 * \brief ����ϸ�ֵ��ཻ��.
		 */
		__global__ void generateSubdivideIntersectionPoint(const EdgeNode* SubdivideValidEdgeArray, const VertexNode* SubdivideVertexArray, const EasyOctNode* SubdivideArray, const int* SubdivideValidVexAddress, const float* SubdivideVvalue, const unsigned int SubdivideValidEdgeArraySize, const unsigned int NodeArraySize, Point3D<float>* SubdivideVertexBuffer);
	
		/**
		 * \brief ��ʼ���̶�ÿ��Ľڵ�����.
		 */
		__global__ void initFixedDepthNums(DeviceArrayView<OctNode> SubdivideNode, DeviceArrayView<int> SubdivideDepthBuffer, const unsigned int DepthOffset, const unsigned int DepthNodeCount, int* fixedDepthNums);

		/**
		 * \brief �ؽ���������.
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
		 * \brief ���㶥�㹹�������ʷֵ���������.
		 * 
		 * \param VertexArray ��������
		 * \param EdgeArray ������
		 * \param FaceArray ������
		 * \param NodeArray �ڵ�����
		 * \param BaseFunction ������
		 * \param dx ɢ��
		 * \param encodeNodeIndexInFunction ����������
		 * \param BaseFunctions ������
		 * \param isoValue ��ֵ
		 * \param DLevelOffset maxDepth��NodeArrayƫ��
		 * \param stream cuda��
		 */
		void calculateTriangleIndices(DeviceArrayView<VertexNode> VertexArray, DeviceArrayView<EdgeNode> EdgeArray, DeviceArrayView<FaceNode> FaceArray, DeviceBufferArray<OctNode>& NodeArray, DeviceArrayView<ConfirmedPPolynomial<CONVTIMES + 1, CONVTIMES + 2>> BaseFunction, DeviceArrayView<float> dx, DeviceArrayView<int> encodeNodeIndexInFunction, DeviceArrayView<unsigned int> DepthBuffer, DeviceArrayView<Point3D<float>> CenterBuffer, const float isoValue, const unsigned int DLevelOffset, const unsigned int DLevelNodeCount, CoredVectorMeshData& mesh, cudaStream_t stream);

		///**
		// * \brief ��ü���õ�����.
		// *
		// * \return ����õ�����
		// */
		//CoredVectorMeshData GetMesh() { return mesh; }

	private:
		DeviceBufferArray<float> vvalue;								// �����Ĳ�����������ʽ����ֵ
		DeviceBufferArray<int> vexNums;									// �����Ĳ��������������
		DeviceBufferArray<int> vexAddress;								// �����Ĳ����������λ��
		DeviceBufferArray<int> triNums;									// �����Ĳ���������������
		DeviceBufferArray<int> triAddress;								// �����Ĳ�����������λ��
		DeviceBufferArray<int> cubeCatagory;							// ��¼����������һ�����͵������壬����ʹ��Marching Cubes�㷨��������

		DeviceBufferArray<bool> markValidVertex;						// �����Ч�Ķ����λ��

		DeviceBufferArray<bool> markValidSubdividedNode;				// ��ǽڵ��Ƿ���Ա�ϸ���Ż�

		//CoredVectorMeshData mesh;										// ����������

		DeviceBufferArray<OctNode> SubdivideNode;						// ϸ�ֽڵ㣬�����ɵ������ʷ�ϸ��

		int SubdivideDepthCount[Constants::maxDepth_Host] = { 0 };		// ϸ�ֽڵ�ÿһ��ڵ������
		int SubdivideDepthAddress[Constants::maxDepth_Host] = { 0 };	// ϸ�ֽڵ�ÿ����SubdivideNode��ƫ��

		int SubdivideNodeNumHost = 0;									// ϸ�ֽڵ�ĸ���
		SynchronizeArray<int> SubdivideDepthBuffer;						// ��¼SubdivideNode����ϸ�ֽڵ�����

		// ϸ���ڵ�Ĳ㣺[0, finerDepth)��Coarser��, [finerDepth, maxDepth]��Finer��
		const unsigned int finerDepth = MAX_DEPTH_OCTREE - COARSER_DIVERGENCE_LEVEL_NUM;	
		int fixedDepthNodeNum[Constants::maxDepth_Host + 1] = { 0 };	// ÿ��ڵ��������ù̶���С������Coarser��ϸ�֡�
		int fixedDepthNodeAddress[Constants::maxDepth_Host + 1] = { 0 };// ÿ��ڵ�ƫ�ơ�����Coarser��ϸ�֡�
		int depthNodeCount[Constants::maxDepth_Host + 1] = { 0 };;		// ÿ��ڵ������������Finer��ϸ�֡�
		int depthNodeAddress[Constants::maxDepth_Host + 1] = { 0 };;	// ÿ��ڵ��ƫ�ơ�����Finer��ϸ�֡�

		DeviceBufferArray<bool> markValidSubdivideVertex;				// ϸ�ֽڵ��б����Ч�Ķ��㡾����Coarser��ϸ�֡�
		DeviceBufferArray<bool> markValidSubdivideEdge;					// ϸ�ֽڵ��б����Ч�ıߡ�����Coarser��ϸ�֡�
		DeviceBufferArray<bool> markValidSubdivedeVexNum;				// ϸ�ֽڵ��б����Ч��vexNums������Coarser��ϸ�֡�

		DeviceBufferArray<bool> markValidFinerVexArray;
		DeviceBufferArray<bool> markValidFinerEdge;
		DeviceBufferArray<bool> markValidFinerVexNum;
		/**
		 * \brief ���㶥��vertex����ʽ����ֵ.
		 *
		 * \param VertexArray ��������
		 * \param NodeArray �ڵ�����
		 * \param BaseFunction ������
		 * \param dx ɢ��
		 * \param encodeNodeIndexInFunction ����������
		 * \param isoValue ��ֵ
		 * \param stream cuda��
		 */
		void ComputeVertexImplicitFunctionValue(DeviceArrayView<VertexNode> VertexArray, DeviceArrayView<OctNode> NodeArray, DeviceArrayView<ConfirmedPPolynomial<CONVTIMES + 1, CONVTIMES + 2>> BaseFunction, DeviceArrayView<float> dx, DeviceArrayView<int> encodeNodeIndexInFunction, const float isoValue, cudaStream_t stream);

		/**
		 * \brief ���ɶ����vertexNums�Ͷ����vertexAddress.
		 *
		 * \param EdgeArray ������
		 * \param NodeArray �ڵ�����
		 * \param vvalue ������ʽ����ֵ
		 * \param stream cuda��
		 */
		void generateVertexNumsAndVertexAddress(DeviceArrayView<EdgeNode> EdgeArray, DeviceArrayView<OctNode> NodeArray, DeviceArrayView<float> vvalue, const unsigned int DLevelOffset, cudaStream_t stream);

		/**
		 * \brief ����TriNums��TriAddress���Լ�����������cubeCatagory.
		 *
		 * \param NodeArray �ڵ�����
		 * \param vvalue ������ʽ����ֵ
		 * \param DLevelOffset ��maxDepth����׽ڵ���NodeArray��ƫ��
		 * \param DLevelNodeCount ��maxDepth��ڵ�����
		 * \param stream cuda��
		 */
		void generateTriangleNumsAndTriangleAddress(DeviceArrayView<OctNode> NodeArray, DeviceArrayView<float> vvalue, const unsigned int DLevelOffset, const unsigned int DLevelNodeCount, CoredVectorMeshData& mesh, cudaStream_t stream);

		/**
		 * \brief ���ɶ����Լ�������.
		 * 
		 * \param VertexArray ��������
		 * \param EdgeArray ������
		 * \param FaceArray ������
		 * \param DLevelOffset ��maxDepth����׽ڵ���NodeArray��ƫ��
		 * \param DLevelNodeCount ��maxDepth��ڵ�����
		 * \param stream cuda��
		 */
		void generateVerticesAndTriangle(DeviceBufferArray<OctNode>& NodeArray, DeviceArrayView<VertexNode> VertexArray, DeviceArrayView<EdgeNode> EdgeArray, DeviceArrayView<FaceNode> FaceArray, const unsigned int DLevelOffset, const unsigned int DLevelNodeCount, CoredVectorMeshData& mesh, cudaStream_t stream);

		/**
		 * \brief ����������.
		 *
		 * \param VertexBuffer ��������
		 * \param allVexNums ������Ч��������
		 * \param TriangleBuffer ����������
		 * \param allTriNums ������Ч����������
		 * \param mesh ����
		 */
		void insertTriangle(const Point3D<float>* VertexBufferHost, const int& allVexNums, const int* TriangleBufferHost, const int& allTriNums, CoredVectorMeshData& mesh);
	
	
		/**
		 * \brief ����ϸ�ֽڵ�������Լ���ͬ��ϸ�ֽڵ��������ƫ�ơ�GPUӲ�����ƣ��޷�ʹ�����첽��������ҪShare Memory > 64kb��GPU��.
		 * 
		 * \param NodeArray �ڵ�����
		 * \param DepthBuffer �ڵ�����NodeArray����Ȳ�������
		 * \param OtherDepthNodeCount ��[0, maxDepth - 1]����׽ڵ���NodeArray��ƫ��
		 * \param stream cuda��
		 */
		void generateSubdivideNodeArrayCountAndAddress(DeviceBufferArray<OctNode>& NodeArray, DeviceArrayView<unsigned int> DepthBuffer, const unsigned int OtherDepthNodeCount, cudaStream_t stream);

		/**
		 * \brief �ֽڵ�ϸ�ֲ��ع�����, �첽�ٶȿ���69.47%��ԭ�ȴ���147.503ms����ǰ����45.0349ms���ƺ�������Finer���У��������̡߳�.
		 * 
		 * \param NodeArray �ڵ�����
		 * \param DepthBuffer �ڵ��������
		 * \param CenterBuffer �ڵ�����λ������
		 * \param BaseFunction ������
		 * \param dx �ڵ�
		 * \param encodeNodeIndexInFunction �������ڵ����
		 * \param isoValue ��ֵ
		 * \param stream cuda��
		 */
		void CoarserSubdivideNodeAndRebuildMesh(DeviceBufferArray<OctNode>& NodeArray, DeviceArrayView<unsigned int> DepthBuffer, DeviceArrayView<Point3D<float>> CenterBuffer, DeviceArrayView<ConfirmedPPolynomial<CONVTIMES + 1, CONVTIMES + 2>> BaseFunction, DeviceArrayView<float> dx, DeviceArrayView<int> encodeNodeIndexInFunction, const float isoValue, CoredVectorMeshData& mesh, cudaStream_t stream);

		/**
		 * \brief ���ڵ�ϸ�ֲ��ع������ƺ�������Coarser���У��������̡߳�.
		 *
		 * \param NodeArray �ڵ�����
		 * \param DepthBuffer �ڵ��������
		 * \param CenterBuffer �ڵ�����λ������
		 * \param BaseFunction ������
		 * \param dx �ڵ�
		 * \param encodeNodeIndexInFunction �������ڵ����
		 * \param isoValue ��ֵ
		 * \param stream cuda��
		 */
		void FinerSubdivideNodeAndRebuildMesh(DeviceBufferArray<OctNode>& NodeArray, DeviceArrayView<unsigned int> DepthBuffer, DeviceArrayView<Point3D<float>> CenterBuffer, DeviceArrayView<ConfirmedPPolynomial<CONVTIMES + 1, CONVTIMES + 2>> BaseFunction, DeviceArrayView<float> dx, DeviceArrayView<int> encodeNodeIndexInFunction, const float isoValue, CoredVectorMeshData& mesh, cudaStream_t stream);


	};
}



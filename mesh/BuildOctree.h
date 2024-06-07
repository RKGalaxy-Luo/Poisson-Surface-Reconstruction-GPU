/*****************************************************************//**
 * \file   BuildOctree.h
 * \brief  ����˲���
 * 
 * \author LUOJIAXUAN
 * \date   May 5th 2024
 *********************************************************************/
#pragma once
#include <base/CommonTypes.h>
#include <base/Constants.h>
#include <base/DeviceReadWrite/SynchronizeArray.h>
#include <base/DeviceReadWrite/DeviceBufferArray.h>
#include <base/DeviceReadWrite/DeviceArrayView.h>

#include <core/AlgorithmTypes.h>

#include <pcl/io/ply_io.h>  // ply �ļ���ȡͷ�ļ�
#include <pcl/io/pcd_io.h>
#include <pcl/point_cloud.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/common/common_headers.h>
#include <pcl/features/normal_3d_omp.h>
#include <pcl/visualization/cloud_viewer.h>
#include <pcl/visualization/pcl_visualizer.h>

#include <boost/thread/thread.hpp>

#include "DebugTest.h"
#include "Geometry.h"
#include "OctNode.cuh"

namespace SparseSurfelFusion {
	namespace device {

		enum {
			MaxCudaThreadsPerBlock = 128
		};

		/**
		 * \brief ��device��OP�������� = sqrt(x^2 + y^2 + z^2).
		 *
		 * \param P����ά����
		 * \return ����double����OP��������ƽ��
		 */
		__device__ float Length(const float3& vec);


		/**
		 * \brief ��Լ���ÿ��block������С��Point3D.
		 * 
		 * \param maxBlockData ��ǰblock�е����ֵ
		 * \param minBlockData ��ǰblock�е���Сֵ
		 * \param points ���ܵ���
		 * \param pointsCount ��������
		 */
		__global__ void reduceMaxMinKernel(Point3D<float>* maxBlockData, Point3D<float>* minBlockData, DeviceArrayView<OrientedPoint3D<float>> points, const unsigned int pointsCount);
		/**
		 * \brief �ҹ�Լ���������С��.
		 * 
		 * \param MaxPoint �������
		 * \param MinPoint �����С��
		 * \param maxArray ÿ��block���ĵ�Array
		 * \param minArray ÿ��block��С���Array
		 * \param GridNum ��Լʱ�������������
		 */
		__host__ void findMaxMinPoint(Point3D<float>& MaxPoint, Point3D<float>& MinPoint, Point3D<float>* maxArray, Point3D<float>* minArray, const unsigned int GridNum);

		/**
		 * \brief ��������Ԫ������ͷ�����ȡ����.
		 * 
		 * \param coordinate ��Ԫ����
		 * \param normals ��Ԫ����
		 * \param PointCloud ����ĳ�����Ԫ
		 * \param pointsCount ������Ԫ�ĸ���
		 */
		__global__ void getCoordinateAndNormalKernel(OrientedPoint3D<float>* point, DeviceArrayView<DepthSurfel> PointCloud, const unsigned int pointsCount);
		
		/**
		 * \brief ����������ĺ˺���.
		 * 
		 * \param coordinate ��Ҫ����ĵ������
		 * \param normal ��ķ���
		 * \param center ���ĵ������
		 * \param scale ���������ߴ�
		 */
		__global__ void adjustPointsCoordinateAndNormalKernel(OrientedPoint3D<float>* points, const Point3D<float> center, const float scale, const unsigned int pointsCount);
		
		/**
		 * \brief Ϊÿ�����ܵ�����һ�����룬���а�������Octree�е�Nodeλ�ã��ٽ��㣬���ڵ㣬�ӽڵ��.
		 * 
		 * \param pos ���ܵ�λ��
		 * \param keys ÿ����ı���key
		 * \param pointsNum ������
		 */
		__global__ void generateCodeKernel(OrientedPoint3D<float>* pos, long long* keys, const unsigned int pointsNum);

		/**
		 * \brief ����64λ����ĺ�32λ��������Ҫ���õ�32λ��index����ΪDensePoints�����Ѿ���������ˣ�֮ǰ��index�������DensePoints����.
		 */
		__global__ void updataLower32ForSortedDensePoints(const unsigned int sortedKeysCount, long long* sortedVerticesKey);

		/**
		 * \brief ��ֻ�Ƚϸ�32bit�����������кõ����ؼ�ֵ(�����ǰֵ������ǰһ��ֵ��label = 1�� �����ǰֵ����ǰһ��ֵ��label = 0).
		 *
		 * \param sortedVerticesKey ����õ�key
		 * \param keyLabel ��VoxelLabel��m_array��GPU���飩��ֵ��
		 */
		__global__ void labelSortedVerticesKeysKernel(const unsigned int sortedKeysCount, DeviceArrayView<long long> sortedVerticesKey, unsigned int* keyLabel);

		/**
		 * \brief ��compactedKey��compactedOffset��ֵ.
		 *
		 * \param sortedVoxelKey ��Ч�����кõļ�ֵ
		 * \param voxelKeyLabel key��label����
		 * \param prefixsumedLabel GPU��ǰ׺�͵ĵ�ַ
		 * \param compactedKey ��á���ǰһ�������ֵ��һ�����ı��루��ǰһ��һ������ȥ��
		 * \param compactedOffset ����������ǰһ�������ֵ��һ�����ı�����pointKeySort.valid_sorted_key�е�λ�ã�idx��
		 */
		__global__ void compactedVoxelKeyKernel(const PtrSize<long long> sortedVoxelKey, const unsigned int* voxelKeyLabel, const unsigned int* prefixsumedLabel, long long* compactedKey, DeviceArrayHandle<int> compactedOffset);
	
		/**
		 * \brief �ռ�������Octree��Node.
		 * 
		 * \param uniqueNode ��Ҫ������Octree��Node.
		 * \param compactedKey ѹ���ļ�
		 * \param compactedNum ѹ���Ľڵ�����
		 * \param compactedOffset ѹ����ƫ��������ͬkey��������
		 */
		__global__ void initUniqueNodeKernel(OctNode* uniqueNode, const DeviceArrayView<long long> compactedKey, const unsigned int compactedNum, const int* compactedOffset);
	
		/**
		 * \brief ����nodeNums�ĺ˺�����nodeNums  -> i - 1 �� i ����ͬһ�����ڵ㣬�� nodeNums = 0�������ͬһ�����ڵ㣬�� nodeNums = 8.
		 * 
		 * \param uniqueCode ѹ����ı���
		 * \param nodesCount ѹ����Ψһ�ڵ������
		 * \param nodeNums ����nodeNums
		 */
		__global__ void generateNodeNumsKernel(const DeviceArrayView<long long> uniqueCode, const unsigned int nodesCount, unsigned int* nodeNums);

		/**
		 * \brief ����NodeArrayD�Լ�ԭʼ���ܵ�����index��NodeArrayD֮���ӳ��.
		 * 
		 * \param uniqueNode ѹ����Ľڵ�
		 * \param nodeAddress �ڵ�ĵ�ַ��eg��0 0 0 0 0 0 0 0 8 8 8 8 8 8 8 8 ...Ϊ�����Ӧ�ڵ�������λ��
		 * \param compactedKey ѹ���Ľڵ�
		 * \param nodesCount ѹ����ڵ������
		 * \param Point2NodeArrayD ��������index��NodeArrayD֮���ӳ��
		 * \param NodeArrayD �ڵ�����
		 */
		__global__ void buildNodeArrayDKernel(DeviceArrayView<OctNode> uniqueNode, DeviceArrayView<unsigned int> nodeAddress, DeviceArrayView<long long> compactedKey, const unsigned int nodesCount, int* Point2NodeArrayD, OctNode* NodeArrayD, unsigned int* NodeArrayDAddressFull);
	
		/**
		 * \brief ��ʼ���ڵ���didx��dnum��������.
		 * 
		 * \param NodeArrayD ��Ҫ��ʼ�������Ľڵ�
		 * \param nodesCount �ڵ�����
		 */
		__global__ void initNodeArrayDidxDnumKernel(OctNode* NodeArrayD, const unsigned int nodesCount);

		/**
		 * \brief ������ܵ㵽NodeArrayD��ӳ�䣬Num���ܵ� >> NodeArrayD����ÿ��δ����Node�ĳ��ܵ㣬��ǰѰ�ҷ�����node�ĳ��ܵ�.
		 * 
		 * \param Point2NodeArrayD ����ӳ������
		 * \param verticesCount ���ܽڵ�����
		 */
		__global__ void processPoint2NodeArrayDKernel(int* Point2NodeArrayD, const unsigned int verticesCount);

		/**
		 * \brief ����Ҫ������Сֵ�Ƚϵ���������pidx��didx��ֵΪ0x7fffffff(���ֵ).
		 * 
		 * \param uniqueNodeArrayPreviousLevel ��һ���uniqueNodeArray
		 * \param TotalNodeNums ���뵱ǰ�㡾���š��ڵ�����
		 */
		__global__ void setPidxDidxInvalidValue(OctNode* uniqueNodeArrayPreviousLevel, const unsigned int TotalNodeNums);

		/**
		 * \brief ������һ���uniqueNodeArray.
		 * 
		 * \param NodeArrayD ���뵱ǰ���Octree�ڵ㡾���š�
		 * \param nodeAddress ���뵱ǰ���NodeAddress�����š�
		 * \param TotalNodeNums ���뵱ǰ�㡾���š��ڵ�����
		 * \param depth ��ǰ������
		 * \param uniqueNodeArrayPreviousLevel ��һ���uniqueNodeArray
		 */
		__global__ void generateUniqueNodeArrayPreviousLevelKernel(DeviceArrayView<OctNode> NodeArrayD, DeviceArrayView<unsigned int> nodeAddress, const unsigned int TotalNodeNums, const unsigned int depth, OctNode* uniqueNodeArrayPreviousLevel);
		
		/**
		 * \brief ������һ�㸸�ڵ��NodeNums�ĺ˺���.
		 * 
		 * \param uniqueNodePreviousLevel ��һ���uniqueNode�� 
		 * \param uniqueCount ��һ��unique�������
		 * \param depth ��ǰ��ȡ�ʵ�ʴ��δ�����depth - 1��
		 * \param NodeNumsPreviousLevel ��һ���NodeNums
		 */
		__global__ void generateNodeNumsPreviousLevelKernel(DeviceArrayView<OctNode> uniqueNodePreviousLevel, const unsigned int uniqueCount, const unsigned int depth, unsigned int* NodeNumsPreviousLevel);

		/**
		 * \brief ������һ�㸸�ڵ��nodeArray��������ǰ���nodeArray�е�parent��ֵ.
		 * 
		 * \param uniqueNodePreviousLevel ���븸�ڵ�
		 * \param nodeAddressPreviousLevel ���븸�ڵ��nodeAddress
		 * \param uniqueCount ���븸�ڵ������
		 * \param depth ��ǰ������
		 * \param nodeArrayPreviousLevel ����������ڵ㹹���nodeArray(����)
		 * \param nodeArrayD ��������ڹ�����һ�㸸�ڵ��NodeArrayʱ����Ҫ����ǰ��ڵ��е�parent���Ը�ֵ
		 * \param NodeAddressFull �������������ҵ�ǰ�������һ���Unique���ұ�
		 */
		__global__ void generateNodeArrayPreviousLevelKernel(DeviceArrayView<OctNode> uniqueNodePreviousLevel, DeviceArrayView<unsigned int> nodeAddressPreviousLevel, const unsigned int uniqueCount, const unsigned int depth, OctNode* nodeArrayPreviousLevel, OctNode* nodeArrayD, unsigned int* NodeAddressFull);
	
		/**
		 * \brief ���½ڵ�NodeArray�ĸ��ڵ�ͺ��ӽڵ�.
		 * 
		 * \param BaseAddressArray_Device ÿһ��ڵ���NodeArray�е�ƫ��
		 * \param totalNodeArrayLength ����NodeArrayһ�����ٸ��ڵ�
		 * \param NodeArray ���º��NodeArray
		 */
		__global__ void updateNodeArrayParentAndChildrenKernel(const unsigned int totalNodeArrayLength, OctNode* NodeArray);

		/**
		 * \brief ����Ч�ڵ�Ҳ������Ӧ��ֵ.
		 * 
		 * \param totalNodeArrayLength ����NodeArrayһ�����ٸ��ڵ�
		 * \param NodeArray ���º��NodeArray
		 */
		__global__ void updateEmptyNodeInfo(const unsigned int totalNodeArrayLength, OctNode* NodeArray);

		/**
		 * \brief ����ڵ���ھӣ���ʱ��Ҫע�����˳�����ÿһ��ڵ㣬�������в㲢�С�ԭ�ģ��ڼ���ڵ���ھ�ʱ����Ҫ�����丸�ڵ���ھӡ��������ԭ������ʹ��˳������˲���ÿ���㣬��ÿһ��ִ���嵥2��
		 *        (ԭ�ģ�When computing a node��s neighbors, its parent��s neighbors are required. For this reason, we perform Listing 2 for all depths using a (forward) level-order traversal of the octree).
		 * 
		 * \param left ��ǰ����׽ڵ���NodeArray�е�index
		 * \param thisLevelNodeCount ��ǰ��Ľڵ�����
		 * \param depth ��ǰ�����
		 * \param NodeArray �˲����ڵ�����
		 */
		__global__ void computeNodeNeighborKernel(const unsigned int left, const unsigned int thisLevelNodeCount, const unsigned int depth, OctNode* NodeArray);
	
		/**
		 * \brief .
		 * 
		 * \param NodeArray �˲����ڵ�����
		 * \param totalNodeCount �˲����ڵ�����NodeArray�Ľڵ�����
		 * \param NodeIndexInFunction 
		 */
		__global__ void computeEncodedFunctionNodeIndexKernel(DeviceArrayView<unsigned int> depthBuffer, DeviceArrayView<OctNode> NodeArray, const unsigned int totalNodeCount, int* NodeIndexInFunction);

		/**
		 * \brief ��key���뵽int32����������[0, 11], [12, 21], [22, 31].
		 * 
		 * \param key �����x,y,z��ֵ
		 * \param CurrentDepth ��ǰ�ڵ���ڵڼ���
		 * \param ����õ���index
		 */
		__device__ void getEncodedFunctionNodeIndex(const int& key, const int& CurrentDepth, int& index);

		/**
		 * \brief ����NodeArray��ÿ���ڵ��ڰ˲����е�����Լ�ʵ�����ĵ�.
		 * 
		 * \param NodeArray �˲����ڵ�����
		 * \param NodeArraySize �˲����ڵ������С
		 * \param DepthBuffer �ڵ�����NodeArrray�нڵ��Ӧ�����(����ֱ�Ӳ��)
		 * \param CenterBuffer �ڵ�����NodeArrray�нڵ��Ӧ���ĵ�����(����ֱ�Ӳ��)
		 */
		__global__ void ComputeDepthAndCenterKernel(DeviceArrayView<OctNode> NodeArray, const unsigned int NodeArraySize, unsigned int* DepthBuffer, Point3D<float>* CenterBuffer);
	

		/**
		 * \brief ��ýڵ������λ��.
		 *
		 * \param key ����Ľڵ��key
		 * \param currentDepth ��ǰ�ڵ�����
		 * \param Center �����õ�ǰ�ڵ��Key��ʵ�����ĵ�
		 */
		__device__ void getNodeCenterAllDepth(const int& key, int currentDepth, Point3D<float>& Center);


	}


	class BuildOctree {
	public:

		using Ptr = std::shared_ptr<BuildOctree>;

		/**
		 * \brief ������Ҫ�ؽ��ĵ���.
		 * 
		 * \param PointCloud
		 */
		BuildOctree();

		/**
		 * \brief ��������.
		 * 
		 */
		~BuildOctree();
		/**
		 * \brief �����˲����ڵ�Array����������������.
		 * 
		 * \param depthSurfel ������Ҫ�ؽ��ĳ�����Ԫ
		 * \param cloud ����xyz
		 * \param normals ���Ʒ���
		 * \param stream cuda��
		 */
		void BuildNodesArray(DeviceArrayView<DepthSurfel> depthSurfel, pcl::PointCloud<pcl::PointXYZ>::Ptr cloud, pcl::PointCloud<pcl::Normal>::Ptr normals, cudaStream_t stream = 0);

		/**
		 * \brief Ԥ����ڵ�Ļ�������������������������.
		 * 
		 * \param stream
		 */
		void ComputeEncodedFunctionNodeIndex(cudaStream_t stream);

		/**
		 * \brief ��ù����õ�Octree��������.
		 * 
		 * \return �����õ�Octree����һά��������
		 */
		DeviceArrayView<OctNode> GetOctreeNodeArray() { return NodeArray.ArrayView(); }		

		/**
		 * \brief �����޸ġ���ù����õ�Octree��������.
		 *
		 * \return �����õ�Octree����һά��������
		 */
		DeviceBufferArray<OctNode>& GetOctreeNodeArrayHandle() { return NodeArray; }

		/**
		 * \brief ���NodeArray��ÿһ��ڵ������.
		 * 
		 * \return ÿһ��ڵ�����������
		 */
		int* GetNodeArrayCount() { return NodeArrayCount_Host; }
			
		/**
		 * \brief ����ÿһ��ڵ���һά�����е�ƫ�ƣ���ÿһ���һ������һά�����е�λ��.
		 * 
		 * \return ÿһ���һ������һά�����е�λ��
		 */
		int* GetBaseAddressArray() { return BaseAddressArray_Host; }
			
		/**
		 * \brief ���ԭʼ����sampleOrientedPoints�����е��ӦNodeArrayD��node��λ�ã�û�ж�Ӧ��һ��дΪ-1.
		 * 
		 * \return ԭʼ����sampleOrientedPoints�����е��ӦNodeArrayD��node��λ��
		 */
		DeviceArrayView<int> GetPoint2NodeArray() { return Point2NodeArray.ArrayView(); }		

		/**
		 * \brief ��ó��ܵ�ת�ɵ����������.
		 * 
		 * \return ���ܵ�ת�ɵ����������.
		 */
		DeviceArrayView<OrientedPoint3D<float>> GetOrientedPoints() { return sampleOrientedPoints.ArrayView(); }		

		/**
		 * \brief ���Ԥ����ڵ�ĺ�������.
		 * 
		 * \return Ԥ����ڵ�ĺ�������
		 */
		DeviceArrayView<int> GetEncodedFunctionNodeIndex() { return EncodedFunctionNodeIndex.ArrayView(); }
		
		/**
		 * \brief ���BaseAddressArray_Device.
		 * 
		 * \return BaseAddressArray_Device
		 */
		DeviceArrayView<int> GetBaseAddressArrayDevice() { return BaseAddressArray_Device.ArrayView(); }

		/**
		 * \brief ���NodeArray��ÿ���ڵ��ڵڼ����ѯ����.
		 * 
		 * \return NodeArrayDepthIndex
		 */
		DeviceArrayView<unsigned int> GetNodeArrayDepthIndex() { return NodeArrayDepthIndex.ArrayView(); }

		/**
		 * \brief ���NodeArray�нڵ��ʵ�����������ѯ����.
		 * 
		 * \return NodeArrayNodeCenter
		 */
		DeviceArrayView<Point3D<float>> GetNodeArrayNodeCenter() { return NodeArrayNodeCenter.ArrayView(); }

	private:

		DeviceBufferArray<OrientedPoint3D<float>> sampleOrientedPoints;			// ��¼��˳����ܵ㣬��������
		SynchronizeArray<Point3D<float>> perBlockMaxPoint;						// ��¼ÿ���߳̿������
		SynchronizeArray<Point3D<float>> perBlockMinPoint;						// ��¼ÿ���߳̿����С��
		DeviceBufferArray<long long> sortCode;									// <���Ĳ���>��¼���ܵ��Ӧ��Octree����Key

		KeyValueSort<long long, OrientedPoint3D<float>> pointKeySort;			// ������ά����ӳ�䵽���أ��������ر��룬�ٽ��������򡿶����ؼ�ִ�������ѹ��
		DeviceBufferArray<unsigned int> keyLabel;								// ���������ı������ҵ�����ǰһ�����벻ͬ���������е�index����¼�����������ر��룬���m_voxel_label[idx] != m_voxel_label[idx-1]����label = 1�� ����label = 0
		PrefixSum nodeNumsPrefixsum;											// ��nodeNums��ǰ׺�͡�����label��ǰ׺�ͣ���Ҫ��������ʾǰ���м�������ǰһ�����벻һ�����ı���
		DeviceBufferArray<long long> uniqueCode;								// <���Ĳ���>����һ�޶������ر���������顿����ǰһ�������ֵ��һ�����ı��루��ǰһ��һ������ȥ��
		DeviceBufferArray<int> compactedVerticesOffset;							// �������һ�޶������ر�����pointKeySort���ĸ�λ�á��������ǰһ�������ֵ��һ�����ı�����m_point_key_sort.valid_sorted_key�е�λ�ã�idx��

		DeviceBufferArray<unsigned int> nodeNums;								// <���Ĳ���>��¼��ͬ���ڵ�Ľڵ�����

		DeviceBufferArray<int> Point2NodeArray;						// ��ԭʼ����sampleOrientedPoints�����е��ӦNodeArrayD��node��λ�ã�û�ж�Ӧ��һ��дΪ-1
		
		int BaseAddressArray_Host[Constants::maxDepth_Host + 1] = { 0 };		// <���Ĳ���>���������ÿ��Ԫ�ؼ�¼NodeArray��ÿ����ȴ��ĵ�һ���ڵ������
		int NodeArrayCount_Host[Constants::maxDepth_Host + 1] = { 0 };			// ��¼ÿһ��ڵ���������

		DeviceBufferArray<int> BaseAddressArray_Device;

		DeviceBufferArray<OctNode> NodeArrays[Constants::maxDepth_Host + 1];	// ÿ���NodeArray���׵�ַ����NodeArrays��Ӧ��������
		DeviceBufferArray<OctNode> NodeArray;									// <���Ĳ���>��ÿһ��NodeArray����(�׵�ַ)��������
		DeviceBufferArray<unsigned int> NodeAddressFull;						// ��ʱ��¼��ǰ�ڵ��Address��ֵ,��ÿһ���NodeArray����һ�����������ӳ��

		DeviceBufferArray<OctNode> uniqueNodeD;									// ���м��������¼�� D ���UniqueNode
		DeviceBufferArray<OctNode> uniqueNodePrevious;							// ���м��������¼��һ���UniqueNode

		DeviceBufferArray<unsigned int> nodeAddressD;							// ���м��������¼�� D ���NodeAddress
		DeviceBufferArray<unsigned int> nodeAddressPrevious;					// ���м��������¼��һ���NodeAddress

		DeviceBufferArray<int> EncodedFunctionNodeIndex;						// Ԥ����ڵ�ĺ�������

		DeviceBufferArray<unsigned int> NodeArrayDepthIndex;					// ��¼NodeArray��ÿ��������������һ��
		DeviceBufferArray<Point3D<float>> NodeArrayNodeCenter;					// ��¼NodeArray��ÿ����������ĵ�

		/**
		 * \brief ��depthsurfel�л����Ԫ����ͷ���.
		 *
		 * \param PointCloud ������Ԫ
		 * \param stream cuda��
		 */
		void getCoordinateAndNormal(DeviceArrayView<DepthSurfel> denseSurfel, cudaStream_t stream = 0);

		/**
		 * \brief �ҵ����Ʒ�Χ.
		 * 
		 * \param points �������
		 * \param MaxPoint ����õ��İ�Χ�����point
		 * \param MinPoint ����õ��İ�Χ����Сpoint
		 * \param stream cuda��
		 */
		void getBoundingBox(DeviceArrayView<OrientedPoint3D<float>> points, Point3D<float>& MaxPoint, Point3D<float>& MinPoint, cudaStream_t stream = 0);

		/**
		 * \brief ��Χ�п��ӻ�.
		 * 
		 * \param cloud �������
		 * \param MaxPoint �����������
		 * \param MinPoint ������С����
		 */
		void BoundBoxVisualization(const pcl::PointCloud<pcl::PointXYZ>::Ptr cloud, Point3D<float> MaxPoint, Point3D<float> MinPoint);

		/**
		 * \brief ����������귶Χ.
		 *
		 * \param points ��Ҫ�޸ĵ������
		 * \param normals ��Ҫ�޸ĵķ���
		 * \param MxPoint �������
		 * \param MnPoint ��С����
		 * \param Scale BoundingBox���ı�
		 * \param Center BoundingBox�����ĵ�
		 * \param stream cuda��
		 */
		void adjustPointsCoordinateAndNormal(DeviceBufferArray<OrientedPoint3D<float>>& points, const Point3D<float> MxPoint, const Point3D<float> MnPoint, float& MaxEdge, float ScaleFactor, Point3D<float>& Center, cudaStream_t stream = 0);
		/**
		 * \brief Ϊÿ�������ɱ���.
		 * 
		 * \param points ������ܵ�
		 * \param key ����Array
		 * \param count ���ܵ�ĸ���
		 * \param stream cuda��
		 */
		void generateCode(DeviceBufferArray<OrientedPoint3D<float>>& points, DeviceBufferArray<long long>& keys, size_t count, cudaStream_t stream = 0);
	
		/**
		 * \brief ���е�ļ�.
		 * 
		 * \param points ���в�ѹ����
		 * \param stream cuda��
		 */
		void sortAndCompactVerticesKeys(DeviceArray<OrientedPoint3D<float>>& points, cudaStream_t stream = 0);
		/**
		 * \brief ����Ψһ��Node.
		 * 
		 * \param uniqueNode Vertex����Node
		 * \param stream cuda��Stream
		 */
		void initUniqueNode(DeviceBufferArray<OctNode>& uniqueNode, DeviceBufferArray<long long>& uniqueCode, cudaStream_t stream = 0);

		/**
		 * \brief ���ɽڵ��NodeNum��i �� i - 1 ��ͬһ������Ϊ8��ͬһ������Ϊ0.
		 * 
		 * \param uniqueKey ѹ�����verticesKeys
		 * \param NodeNums ��¼�ڵ��NodeNum
		 * \param stream cuda��Stream
		 */
		void generateNodeNumsAndNodeAddress(DeviceBufferArray<long long>& uniqueCode, DeviceBufferArray<unsigned int>& NodeNums, DeviceBufferArray<unsigned int>& NodeAddress, cudaStream_t stream = 0);
	
		/**
		 * \brief ������D��Ľڵ�Array��NodeArrayD.
		 * 
		 * \param denseVertices �������򶥵�
		 * \param uniqueNode ѹ���Ľڵ�
		 * \param compactedKey ѹ���ļ�
		 * \param NodeAddress ��¼8���ӽڵ��λ��
		 * \param Point2NodeArray ���ܵ���NodeArray�е�λ��
		 * \param NodeArrayD ������D��Ľڵ�����
		 * \param stream cuda��
		 */
		void buildNodeArrayD(DeviceArrayView<OrientedPoint3D<float>> denseVertices, DeviceArrayView<OctNode> uniqueNode, DeviceArrayView<long long> compactedKey, DeviceBufferArray<unsigned int>& NodeAddress, DeviceBufferArray<unsigned int>& NodeAddressFull, DeviceBufferArray<int>& Point2NodeArray, DeviceBufferArray<OctNode>& NodeArrayD, cudaStream_t stream = 0);
		
		/**
		 * \brief �������в�ڵ��ƴ��NodeArray�Լ�ÿһ����׽ڵ�BaseAddressArray.
		 * 
		 * \param BaseAddressArray ÿһ����׽ڵ�BaseAddressArray
		 * \param stream cuda��
		 */
		void buildOtherDepthNodeArray(int* BaseAddressArray_Host, cudaStream_t stream = 0);

		/**
		 * \brief �ǰ�湹���ڵ�Ĳ���������Ҽ���ڵ�Depth���ұ��ڵ����ĵ�Center���ұ�.
		 * 
		 * \param BaseAddressArray_Host ÿ��ƫ��
		 * \param NodeArray �ڵ�����
		 * \param stream cuda��
		 */
		void updateNodeInfo(int* BaseAddressArray_Host, DeviceBufferArray<OctNode>& NodeArray, cudaStream_t stream = 0);
		
		/**
		 * \brief �����ڵ���ھ�.
		 * 
		 * \param NodeArray �ڵ�����
		 * \param stream cuda��
		 */
		void computeNodeNeighbor(DeviceBufferArray<OctNode>& NodeArray, cudaStream_t stream = 0);
	};
}


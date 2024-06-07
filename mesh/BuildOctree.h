/*****************************************************************//**
 * \file   BuildOctree.h
 * \brief  构造八叉树
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

#include <pcl/io/ply_io.h>  // ply 文件读取头文件
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
		 * \brief 【device】OP向量长度 = sqrt(x^2 + y^2 + z^2).
		 *
		 * \param P点三维坐标
		 * \return 返回double类型OP向量长度平方
		 */
		__device__ float Length(const float3& vec);


		/**
		 * \brief 规约获得每个block最大和最小的Point3D.
		 * 
		 * \param maxBlockData 当前block中的最大值
		 * \param minBlockData 当前block中的最小值
		 * \param points 稠密点云
		 * \param pointsCount 点云数量
		 */
		__global__ void reduceMaxMinKernel(Point3D<float>* maxBlockData, Point3D<float>* minBlockData, DeviceArrayView<OrientedPoint3D<float>> points, const unsigned int pointsCount);
		/**
		 * \brief 找规约后的最大和最小点.
		 * 
		 * \param MaxPoint 输出最大点
		 * \param MinPoint 输出最小点
		 * \param maxArray 每个block最大的点Array
		 * \param minArray 每个block最小点点Array
		 * \param GridNum 规约时分配的网格数量
		 */
		__host__ void findMaxMinPoint(Point3D<float>& MaxPoint, Point3D<float>& MinPoint, Point3D<float>* maxArray, Point3D<float>* minArray, const unsigned int GridNum);

		/**
		 * \brief 将稠密面元中坐标和发现提取出来.
		 * 
		 * \param coordinate 面元坐标
		 * \param normals 面元发现
		 * \param PointCloud 传入的稠密面元
		 * \param pointsCount 传入面元的个数
		 */
		__global__ void getCoordinateAndNormalKernel(OrientedPoint3D<float>* point, DeviceArrayView<DepthSurfel> PointCloud, const unsigned int pointsCount);
		
		/**
		 * \brief 调整点坐标的核函数.
		 * 
		 * \param coordinate 需要变更的点的坐标
		 * \param normal 点的法线
		 * \param center 中心点的坐标
		 * \param scale 坐标点放缩尺寸
		 */
		__global__ void adjustPointsCoordinateAndNormalKernel(OrientedPoint3D<float>* points, const Point3D<float> center, const float scale, const unsigned int pointsCount);
		
		/**
		 * \brief 为每个稠密点生成一个编码，其中包括点在Octree中的Node位置，临近点，父节点，子节点等.
		 * 
		 * \param pos 稠密点位置
		 * \param keys 每个点的编码key
		 * \param pointsNum 点数量
		 */
		__global__ void generateCodeKernel(OrientedPoint3D<float>* pos, long long* keys, const unsigned int pointsNum);

		/**
		 * \brief 更新64位编码的后32位，这里需要重置低32位的index，因为DensePoints数组已经是有序的了，之前的index是无需的DensePoints数组.
		 */
		__global__ void updataLower32ForSortedDensePoints(const unsigned int sortedKeysCount, long long* sortedVerticesKey);

		/**
		 * \brief 【只比较高32bit情况】标记排列好的体素键值(如果当前值不等于前一个值则label = 1， 如果当前值等于前一个值则label = 0).
		 *
		 * \param sortedVerticesKey 排序好的key
		 * \param keyLabel 给VoxelLabel的m_array（GPU数组）赋值了
		 */
		__global__ void labelSortedVerticesKeysKernel(const unsigned int sortedKeysCount, DeviceArrayView<long long> sortedVerticesKey, unsigned int* keyLabel);

		/**
		 * \brief 给compactedKey和compactedOffset赋值.
		 *
		 * \param sortedVoxelKey 有效且排列好的键值
		 * \param voxelKeyLabel key的label数组
		 * \param prefixsumedLabel GPU中前缀和的地址
		 * \param compactedKey 获得“与前一个编码键值不一样”的编码（与前一个一样的舍去）
		 * \param compactedOffset 获得这个“与前一个编码键值不一样”的编码在pointKeySort.valid_sorted_key中的位置（idx）
		 */
		__global__ void compactedVoxelKeyKernel(const PtrSize<long long> sortedVoxelKey, const unsigned int* voxelKeyLabel, const unsigned int* prefixsumedLabel, long long* compactedKey, DeviceArrayHandle<int> compactedOffset);
	
		/**
		 * \brief 收集并构建Octree的Node.
		 * 
		 * \param uniqueNode 需要构建的Octree的Node.
		 * \param compactedKey 压缩的键
		 * \param compactedNum 压缩的节点数量
		 * \param compactedOffset 压缩的偏移量（相同key的数量）
		 */
		__global__ void initUniqueNodeKernel(OctNode* uniqueNode, const DeviceArrayView<long long> compactedKey, const unsigned int compactedNum, const int* compactedOffset);
	
		/**
		 * \brief 生成nodeNums的核函数：nodeNums  -> i - 1 和 i 共享同一个父节点，则 nodeNums = 0，如果非同一个父节点，则 nodeNums = 8.
		 * 
		 * \param uniqueCode 压缩后的编码
		 * \param nodesCount 压缩后唯一节点的总数
		 * \param nodeNums 生成nodeNums
		 */
		__global__ void generateNodeNumsKernel(const DeviceArrayView<long long> uniqueCode, const unsigned int nodesCount, unsigned int* nodeNums);

		/**
		 * \brief 构建NodeArrayD以及原始稠密点数组index与NodeArrayD之间的映射.
		 * 
		 * \param uniqueNode 压缩后的节点
		 * \param nodeAddress 节点的地址，eg：0 0 0 0 0 0 0 0 8 8 8 8 8 8 8 8 ...为了求对应节点在树的位置
		 * \param compactedKey 压缩的节点
		 * \param nodesCount 压缩后节点的总数
		 * \param Point2NodeArrayD 稠密数组index与NodeArrayD之间的映射
		 * \param NodeArrayD 节点数组
		 */
		__global__ void buildNodeArrayDKernel(DeviceArrayView<OctNode> uniqueNode, DeviceArrayView<unsigned int> nodeAddress, DeviceArrayView<long long> compactedKey, const unsigned int nodesCount, int* Point2NodeArrayD, OctNode* NodeArrayD, unsigned int* NodeArrayDAddressFull);
	
		/**
		 * \brief 初始化节点中didx和dnum两个参数.
		 * 
		 * \param NodeArrayD 需要初始化参数的节点
		 * \param nodesCount 节点总数
		 */
		__global__ void initNodeArrayDidxDnumKernel(OctNode* NodeArrayD, const unsigned int nodesCount);

		/**
		 * \brief 处理稠密点到NodeArrayD的映射，Num稠密点 >> NodeArrayD，将每个未分配Node的稠密点，往前寻找分配了node的稠密点.
		 * 
		 * \param Point2NodeArrayD 传入映射数组
		 * \param verticesCount 稠密节点数量
		 */
		__global__ void processPoint2NodeArrayDKernel(int* Point2NodeArrayD, const unsigned int verticesCount);

		/**
		 * \brief 将需要进行最小值比较的两个参数pidx和didx赋值为0x7fffffff(最大值).
		 * 
		 * \param uniqueNodeArrayPreviousLevel 上一层的uniqueNodeArray
		 * \param TotalNodeNums 传入当前层【满排】节点数量
		 */
		__global__ void setPidxDidxInvalidValue(OctNode* uniqueNodeArrayPreviousLevel, const unsigned int TotalNodeNums);

		/**
		 * \brief 生成上一层的uniqueNodeArray.
		 * 
		 * \param NodeArrayD 传入当前层的Octree节点【满排】
		 * \param nodeAddress 传入当前层的NodeAddress【满排】
		 * \param TotalNodeNums 传入当前层【满排】节点数量
		 * \param depth 当前层的深度
		 * \param uniqueNodeArrayPreviousLevel 上一层的uniqueNodeArray
		 */
		__global__ void generateUniqueNodeArrayPreviousLevelKernel(DeviceArrayView<OctNode> NodeArrayD, DeviceArrayView<unsigned int> nodeAddress, const unsigned int TotalNodeNums, const unsigned int depth, OctNode* uniqueNodeArrayPreviousLevel);
		
		/**
		 * \brief 构建上一层父节点的NodeNums的核函数.
		 * 
		 * \param uniqueNodePreviousLevel 上一层的uniqueNode， 
		 * \param uniqueCount 上一层unique点的数量
		 * \param depth 当前深度【实际传参传的是depth - 1】
		 * \param NodeNumsPreviousLevel 上一层的NodeNums
		 */
		__global__ void generateNodeNumsPreviousLevelKernel(DeviceArrayView<OctNode> uniqueNodePreviousLevel, const unsigned int uniqueCount, const unsigned int depth, unsigned int* NodeNumsPreviousLevel);

		/**
		 * \brief 构建上一层父节点的nodeArray，并给当前层的nodeArray中的parent赋值.
		 * 
		 * \param uniqueNodePreviousLevel 传入父节点
		 * \param nodeAddressPreviousLevel 传入父节点的nodeAddress
		 * \param uniqueCount 传入父节点的数量
		 * \param depth 当前层的深度
		 * \param nodeArrayPreviousLevel 【输出】父节点构造的nodeArray(满排)
		 * \param nodeArrayD 【输出】在构建上一层父节点的NodeArray时，需要给当前层节点中的parent属性赋值
		 * \param NodeAddressFull 【输出】构造查找当前层查找上一层的Unique查找表
		 */
		__global__ void generateNodeArrayPreviousLevelKernel(DeviceArrayView<OctNode> uniqueNodePreviousLevel, DeviceArrayView<unsigned int> nodeAddressPreviousLevel, const unsigned int uniqueCount, const unsigned int depth, OctNode* nodeArrayPreviousLevel, OctNode* nodeArrayD, unsigned int* NodeAddressFull);
	
		/**
		 * \brief 更新节点NodeArray的父节点和孩子节点.
		 * 
		 * \param BaseAddressArray_Device 每一层节点在NodeArray中的偏移
		 * \param totalNodeArrayLength 整个NodeArray一共多少个节点
		 * \param NodeArray 更新后的NodeArray
		 */
		__global__ void updateNodeArrayParentAndChildrenKernel(const unsigned int totalNodeArrayLength, OctNode* NodeArray);

		/**
		 * \brief 将无效节点也赋上相应的值.
		 * 
		 * \param totalNodeArrayLength 整个NodeArray一共多少个节点
		 * \param NodeArray 更新后的NodeArray
		 */
		__global__ void updateEmptyNodeInfo(const unsigned int totalNodeArrayLength, OctNode* NodeArray);

		/**
		 * \brief 计算节点的邻居，此时需要注意必须顺序计算每一层节点，不能所有层并行【原文：在计算节点的邻居时，需要计算其父节点的邻居。出于这个原因，我们使用顺序遍历八叉树每个层，对每一层执行清单2】
		 *        (原文：When computing a node’s neighbors, its parent’s neighbors are required. For this reason, we perform Listing 2 for all depths using a (forward) level-order traversal of the octree).
		 * 
		 * \param left 当前层的首节点在NodeArray中的index
		 * \param thisLevelNodeCount 当前层的节点数量
		 * \param depth 当前层深度
		 * \param NodeArray 八叉树节点数组
		 */
		__global__ void computeNodeNeighborKernel(const unsigned int left, const unsigned int thisLevelNodeCount, const unsigned int depth, OctNode* NodeArray);
	
		/**
		 * \brief .
		 * 
		 * \param NodeArray 八叉树节点数组
		 * \param totalNodeCount 八叉树节点数组NodeArray的节点数量
		 * \param NodeIndexInFunction 
		 */
		__global__ void computeEncodedFunctionNodeIndexKernel(DeviceArrayView<unsigned int> depthBuffer, DeviceArrayView<OctNode> NodeArray, const unsigned int totalNodeCount, int* NodeIndexInFunction);

		/**
		 * \brief 将key编码到int32的三个区间[0, 11], [12, 21], [22, 31].
		 * 
		 * \param key 传入的x,y,z键值
		 * \param CurrentDepth 当前节点的在第几层
		 * \param 计算得到的index
		 */
		__device__ void getEncodedFunctionNodeIndex(const int& key, const int& CurrentDepth, int& index);

		/**
		 * \brief 计算NodeArray中每个节点在八叉树中的深度以及实际中心点.
		 * 
		 * \param NodeArray 八叉树节点数组
		 * \param NodeArraySize 八叉树节点数组大小
		 * \param DepthBuffer 节点数组NodeArrray中节点对应的深度(后续直接查表)
		 * \param CenterBuffer 节点数组NodeArrray中节点对应中心点坐标(后续直接查表)
		 */
		__global__ void ComputeDepthAndCenterKernel(DeviceArrayView<OctNode> NodeArray, const unsigned int NodeArraySize, unsigned int* DepthBuffer, Point3D<float>* CenterBuffer);
	

		/**
		 * \brief 获得节点的中心位置.
		 *
		 * \param key 传入的节点的key
		 * \param currentDepth 当前节点的深度
		 * \param Center 计算获得当前节点的Key的实际中心点
		 */
		__device__ void getNodeCenterAllDepth(const int& key, int currentDepth, Point3D<float>& Center);


	}


	class BuildOctree {
	public:

		using Ptr = std::shared_ptr<BuildOctree>;

		/**
		 * \brief 传入需要重建的点云.
		 * 
		 * \param PointCloud
		 */
		BuildOctree();

		/**
		 * \brief 析构函数.
		 * 
		 */
		~BuildOctree();
		/**
		 * \brief 构建八叉树节点Array【函数内有阻塞】.
		 * 
		 * \param depthSurfel 传入需要重建的稠密面元
		 * \param cloud 点云xyz
		 * \param normals 点云法线
		 * \param stream cuda流
		 */
		void BuildNodesArray(DeviceArrayView<DepthSurfel> depthSurfel, pcl::PointCloud<pcl::PointXYZ>::Ptr cloud, pcl::PointCloud<pcl::Normal>::Ptr normals, cudaStream_t stream = 0);

		/**
		 * \brief 预计算节点的基函数索引【函数内无阻塞】.
		 * 
		 * \param stream
		 */
		void ComputeEncodedFunctionNodeIndex(cudaStream_t stream);

		/**
		 * \brief 获得构建好的Octree树的数组.
		 * 
		 * \return 构建好的Octree树的一维数组数组
		 */
		DeviceArrayView<OctNode> GetOctreeNodeArray() { return NodeArray.ArrayView(); }		

		/**
		 * \brief 【可修改】获得构建好的Octree树的数组.
		 *
		 * \return 构建好的Octree树的一维数组数组
		 */
		DeviceBufferArray<OctNode>& GetOctreeNodeArrayHandle() { return NodeArray; }

		/**
		 * \brief 获得NodeArray，每一层节点的数量.
		 * 
		 * \return 每一层节点数量的数组
		 */
		int* GetNodeArrayCount() { return NodeArrayCount_Host; }
			
		/**
		 * \brief 返回每一层节点在一维数组中的偏移，即每一层第一个点在一维数组中的位置.
		 * 
		 * \return 每一层第一个点在一维数组中的位置
		 */
		int* GetBaseAddressArray() { return BaseAddressArray_Host; }
			
		/**
		 * \brief 获得原始稠密sampleOrientedPoints数组中点对应NodeArrayD中node的位置，没有对应的一律写为-1.
		 * 
		 * \return 原始稠密sampleOrientedPoints数组中点对应NodeArrayD中node的位置
		 */
		DeviceArrayView<int> GetPoint2NodeArray() { return Point2NodeArray.ArrayView(); }		

		/**
		 * \brief 获得稠密点转成的有向采样点.
		 * 
		 * \return 稠密点转成的有向采样点.
		 */
		DeviceArrayView<OrientedPoint3D<float>> GetOrientedPoints() { return sampleOrientedPoints.ArrayView(); }		

		/**
		 * \brief 获得预计算节点的函数索引.
		 * 
		 * \return 预计算节点的函数索引
		 */
		DeviceArrayView<int> GetEncodedFunctionNodeIndex() { return EncodedFunctionNodeIndex.ArrayView(); }
		
		/**
		 * \brief 获得BaseAddressArray_Device.
		 * 
		 * \return BaseAddressArray_Device
		 */
		DeviceArrayView<int> GetBaseAddressArrayDevice() { return BaseAddressArray_Device.ArrayView(); }

		/**
		 * \brief 获得NodeArray中每个节点在第几层查询数组.
		 * 
		 * \return NodeArrayDepthIndex
		 */
		DeviceArrayView<unsigned int> GetNodeArrayDepthIndex() { return NodeArrayDepthIndex.ArrayView(); }

		/**
		 * \brief 获得NodeArray中节点的实际中心坐标查询数组.
		 * 
		 * \return NodeArrayNodeCenter
		 */
		DeviceArrayView<Point3D<float>> GetNodeArrayNodeCenter() { return NodeArrayNodeCenter.ArrayView(); }

	private:

		DeviceBufferArray<OrientedPoint3D<float>> sampleOrientedPoints;			// 记录无顺序稠密点，后续排序
		SynchronizeArray<Point3D<float>> perBlockMaxPoint;						// 记录每个线程块的最大点
		SynchronizeArray<Point3D<float>> perBlockMinPoint;						// 记录每个线程块的最小点
		DeviceBufferArray<long long> sortCode;									// <论文参数>记录稠密点对应的Octree编码Key

		KeyValueSort<long long, OrientedPoint3D<float>> pointKeySort;			// 【将三维坐标映射到体素，并将体素编码，再将编码排序】对体素键执行排序和压缩
		DeviceBufferArray<unsigned int> keyLabel;								// 【在排序后的编码中找到，与前一个编码不同的在数组中的index】记录着排序后的体素编码，如果m_voxel_label[idx] != m_voxel_label[idx-1]，则label = 1， 否则label = 0
		PrefixSum nodeNumsPrefixsum;											// 【nodeNums的前缀和】体素label的前缀和，主要作用是显示前面有几个“与前一个编码不一样”的编码
		DeviceBufferArray<long long> uniqueCode;								// <论文参数>【独一无二的体素编码放入数组】“与前一个编码键值不一样”的编码（与前一个一样的舍去）
		DeviceBufferArray<int> compactedVerticesOffset;							// 【这个独一无二的体素编码在pointKeySort的哪个位置】这个“与前一个编码键值不一样”的编码在m_point_key_sort.valid_sorted_key中的位置（idx）

		DeviceBufferArray<unsigned int> nodeNums;								// <论文参数>记录相同父节点的节点数量

		DeviceBufferArray<int> Point2NodeArray;						// 从原始稠密sampleOrientedPoints数组中点对应NodeArrayD中node的位置，没有对应的一律写为-1
		
		int BaseAddressArray_Host[Constants::maxDepth_Host + 1] = { 0 };		// <论文参数>其中数组的每个元素记录NodeArray中每个深度处的第一个节点的索引
		int NodeArrayCount_Host[Constants::maxDepth_Host + 1] = { 0 };			// 记录每一层节点满排数量

		DeviceBufferArray<int> BaseAddressArray_Device;

		DeviceBufferArray<OctNode> NodeArrays[Constants::maxDepth_Host + 1];	// 每层的NodeArray的首地址存在NodeArrays对应的数组中
		DeviceBufferArray<OctNode> NodeArray;									// <论文参数>将每一层NodeArray数组(首地址)连接起来
		DeviceBufferArray<unsigned int> NodeAddressFull;						// 临时记录当前节点的Address的值,与每一层的NodeArray必须一样大，做到查表映射

		DeviceBufferArray<OctNode> uniqueNodeD;									// 【中间变量】记录第 D 层的UniqueNode
		DeviceBufferArray<OctNode> uniqueNodePrevious;							// 【中间变量】记录上一层的UniqueNode

		DeviceBufferArray<unsigned int> nodeAddressD;							// 【中间变量】记录第 D 层的NodeAddress
		DeviceBufferArray<unsigned int> nodeAddressPrevious;					// 【中间变量】记录上一层的NodeAddress

		DeviceBufferArray<int> EncodedFunctionNodeIndex;						// 预计算节点的函数索引

		DeviceBufferArray<unsigned int> NodeArrayDepthIndex;					// 记录NodeArray中每个顶点来自于哪一层
		DeviceBufferArray<Point3D<float>> NodeArrayNodeCenter;					// 记录NodeArray中每个顶点的中心点

		/**
		 * \brief 从depthsurfel中获得面元坐标和法线.
		 *
		 * \param PointCloud 稠密面元
		 * \param stream cuda流
		 */
		void getCoordinateAndNormal(DeviceArrayView<DepthSurfel> denseSurfel, cudaStream_t stream = 0);

		/**
		 * \brief 找到点云范围.
		 * 
		 * \param points 传入点云
		 * \param MaxPoint 计算得到的包围盒最大point
		 * \param MinPoint 计算得到的包围盒最小point
		 * \param stream cuda流
		 */
		void getBoundingBox(DeviceArrayView<OrientedPoint3D<float>> points, Point3D<float>& MaxPoint, Point3D<float>& MinPoint, cudaStream_t stream = 0);

		/**
		 * \brief 包围盒可视化.
		 * 
		 * \param cloud 传入点云
		 * \param MaxPoint 传入最大坐标
		 * \param MinPoint 传入最小坐标
		 */
		void BoundBoxVisualization(const pcl::PointCloud<pcl::PointXYZ>::Ptr cloud, Point3D<float> MaxPoint, Point3D<float> MinPoint);

		/**
		 * \brief 调整点的坐标范围.
		 *
		 * \param points 需要修改的坐标点
		 * \param normals 需要修改的法线
		 * \param MxPoint 最大坐标
		 * \param MnPoint 最小坐标
		 * \param Scale BoundingBox最大的边
		 * \param Center BoundingBox的中心点
		 * \param stream cuda流
		 */
		void adjustPointsCoordinateAndNormal(DeviceBufferArray<OrientedPoint3D<float>>& points, const Point3D<float> MxPoint, const Point3D<float> MnPoint, float& MaxEdge, float ScaleFactor, Point3D<float>& Center, cudaStream_t stream = 0);
		/**
		 * \brief 为每个点生成编码.
		 * 
		 * \param points 传入稠密点
		 * \param key 编码Array
		 * \param count 稠密点的个数
		 * \param stream cuda流
		 */
		void generateCode(DeviceBufferArray<OrientedPoint3D<float>>& points, DeviceBufferArray<long long>& keys, size_t count, cudaStream_t stream = 0);
	
		/**
		 * \brief 排列点的键.
		 * 
		 * \param points 排列并压缩点
		 * \param stream cuda流
		 */
		void sortAndCompactVerticesKeys(DeviceArray<OrientedPoint3D<float>>& points, cudaStream_t stream = 0);
		/**
		 * \brief 构造唯一的Node.
		 * 
		 * \param uniqueNode Vertex构造Node
		 * \param stream cuda流Stream
		 */
		void initUniqueNode(DeviceBufferArray<OctNode>& uniqueNode, DeviceBufferArray<long long>& uniqueCode, cudaStream_t stream = 0);

		/**
		 * \brief 生成节点的NodeNum，i 与 i - 1 非同一父亲则为8，同一父亲则为0.
		 * 
		 * \param uniqueKey 压缩后的verticesKeys
		 * \param NodeNums 记录节点的NodeNum
		 * \param stream cuda流Stream
		 */
		void generateNodeNumsAndNodeAddress(DeviceBufferArray<long long>& uniqueCode, DeviceBufferArray<unsigned int>& NodeNums, DeviceBufferArray<unsigned int>& NodeAddress, cudaStream_t stream = 0);
	
		/**
		 * \brief 构建第D层的节点Array：NodeArrayD.
		 * 
		 * \param denseVertices 稠密有向顶点
		 * \param uniqueNode 压缩的节点
		 * \param compactedKey 压缩的键
		 * \param NodeAddress 记录8个子节点的位置
		 * \param Point2NodeArray 稠密点在NodeArray中的位置
		 * \param NodeArrayD 构建第D层的节点数组
		 * \param stream cuda流
		 */
		void buildNodeArrayD(DeviceArrayView<OrientedPoint3D<float>> denseVertices, DeviceArrayView<OctNode> uniqueNode, DeviceArrayView<long long> compactedKey, DeviceBufferArray<unsigned int>& NodeAddress, DeviceBufferArray<unsigned int>& NodeAddressFull, DeviceBufferArray<int>& Point2NodeArray, DeviceBufferArray<OctNode>& NodeArrayD, cudaStream_t stream = 0);
		
		/**
		 * \brief 构建所有层节点的拼接NodeArray以及每一层的首节点BaseAddressArray.
		 * 
		 * \param BaseAddressArray 每一层的首节点BaseAddressArray
		 * \param stream cuda流
		 */
		void buildOtherDepthNodeArray(int* BaseAddressArray_Host, cudaStream_t stream = 0);

		/**
		 * \brief 填补前面构建节点的不完整项，并且计算节点Depth查找表，节点中心点Center查找表.
		 * 
		 * \param BaseAddressArray_Host 每层偏移
		 * \param NodeArray 节点数组
		 * \param stream cuda流
		 */
		void updateNodeInfo(int* BaseAddressArray_Host, DeviceBufferArray<OctNode>& NodeArray, cudaStream_t stream = 0);
		
		/**
		 * \brief 构建节点的邻居.
		 * 
		 * \param NodeArray 节点数组
		 * \param stream cuda流
		 */
		void computeNodeNeighbor(DeviceBufferArray<OctNode>& NodeArray, cudaStream_t stream = 0);
	};
}


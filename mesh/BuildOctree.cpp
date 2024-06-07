/*****************************************************************//**
 * \file   BuildOctree.cpp
 * \brief  构造八叉树方法实现
 * 
 * \author LUOJIAXUAN
 * \date   May 5th 2024
 *********************************************************************/
#include "BuildOctree.h"

SparseSurfelFusion::BuildOctree::BuildOctree()
{
	
	sampleOrientedPoints.AllocateBuffer(MAX_SURFEL_COUNT);
	perBlockMaxPoint.AllocateBuffer(divUp(MAX_SURFEL_COUNT, device::MaxCudaThreadsPerBlock));
	perBlockMinPoint.AllocateBuffer(divUp(MAX_SURFEL_COUNT, device::MaxCudaThreadsPerBlock));
	sortCode.AllocateBuffer(MAX_SURFEL_COUNT);

	pointKeySort.AllocateBuffer(MAX_SURFEL_COUNT);
	keyLabel.AllocateBuffer(MAX_SURFEL_COUNT);
	uniqueCode.AllocateBuffer(MAX_SURFEL_COUNT);
	compactedVerticesOffset.AllocateBuffer(MAX_SURFEL_COUNT);

	nodeNums.AllocateBuffer(MAX_SURFEL_COUNT);
	nodeNumsPrefixsum.AllocateBuffer(MAX_SURFEL_COUNT);

	Point2NodeArray.AllocateBuffer(MAX_SURFEL_COUNT);

	uniqueNodeD.AllocateBuffer(MAX_SURFEL_COUNT);
	uniqueNodePrevious.AllocateBuffer(MAX_SURFEL_COUNT);

	nodeAddressD.AllocateBuffer(MAX_SURFEL_COUNT);
	nodeAddressPrevious.AllocateBuffer(MAX_SURFEL_COUNT);

	NodeArray.AllocateBuffer(TOTAL_NODEARRAY_MAX_COUNT);					// 预估10倍最大节点
	EncodedFunctionNodeIndex.AllocateBuffer(TOTAL_NODEARRAY_MAX_COUNT);
	NodeAddressFull.AllocateBuffer(D_LEVEL_MAX_NODE);						// 最坏情况应该是8 * MAX_SURFEL_COUNT

	BaseAddressArray_Device.AllocateBuffer(Constants::maxDepth_Host + 1);

	NodeArrayDepthIndex.AllocateBuffer(TOTAL_NODEARRAY_MAX_COUNT);
	NodeArrayNodeCenter.AllocateBuffer(TOTAL_NODEARRAY_MAX_COUNT);

}

SparseSurfelFusion::BuildOctree::~BuildOctree()
{
	sampleOrientedPoints.ReleaseBuffer();
	perBlockMaxPoint.DeviceArray().release();
	perBlockMinPoint.DeviceArray().release();
	sortCode.ReleaseBuffer();

	keyLabel.ReleaseBuffer();
	uniqueCode.ReleaseBuffer();
	compactedVerticesOffset.ReleaseBuffer();

	nodeNums.ReleaseBuffer();

	uniqueNodeD.ReleaseBuffer();
	uniqueNodePrevious.ReleaseBuffer();

	nodeAddressD.ReleaseBuffer();
	nodeAddressPrevious.ReleaseBuffer();

	NodeArray.ReleaseBuffer();
	EncodedFunctionNodeIndex.ReleaseBuffer();
	NodeAddressFull.ReleaseBuffer();

	BaseAddressArray_Device.ReleaseBuffer();

	NodeArrayDepthIndex.ReleaseBuffer();
	NodeArrayNodeCenter.ReleaseBuffer();

}

void SparseSurfelFusion::BuildOctree::BuildNodesArray(DeviceArrayView<DepthSurfel> depthSurfel, pcl::PointCloud<pcl::PointXYZ>::Ptr cloud, pcl::PointCloud<pcl::Normal>::Ptr normals, cudaStream_t stream)
{
#ifdef CHECK_MESH_BUILD_TIME_COST
	auto start = std::chrono::high_resolution_clock::now();						// 记录开始时间点
#endif // CHECK_MESH_BUILD_TIME_COST



	size_t count = depthSurfel.Size();
	sampleOrientedPoints.ResizeArrayOrException(count);

	/***************************************** Step.1：计算BoundingBox *****************************************/
	Point3D<float> MaxPoint = Point3D<float>(float(-1e6), float(-1e6), float(-1e6));// 用于框定点云x,y,z最大值
	Point3D<float> MinPoint = Point3D<float>(float(1e6), float(1e6), float(1e6));	// 用于框定点云x,y,z最小值
	Point3D<float> center;		// BoundingBox的中心
	float maxEdge = 1.0f;		// BoundingBox的最长的边长
	float scaleFactor = 1.25f;	// 放缩尺寸【大于1：将所有点云在[0, 1]的范围压缩；小于1：点云将超出[0, 1]；等于1：x,y,z最大分量 == 1】

#ifdef CHECK_MESH_BUILD_TIME_COST
	auto time1 = std::chrono::high_resolution_clock::now();					// 记录开始时间点
#endif // CHECK_MESH_BUILD_TIME_COST


	// DenseSurfel转成Point3D
	getCoordinateAndNormal(depthSurfel, stream);		
	// 获得包围盒(归约算法)
	getBoundingBox(sampleOrientedPoints.ArrayView(), MaxPoint, MinPoint, stream);

	//// 包围盒可视化
	//BoundBoxVisualization(cloud, MaxPoint, MinPoint);

	// 调整根据scaleFactor调整各个点云坐标位置
	adjustPointsCoordinateAndNormal(sampleOrientedPoints, MaxPoint, MinPoint, maxEdge, scaleFactor, center, stream);

#ifdef CHECK_MESH_BUILD_TIME_COST
	CHECKCUDA(cudaStreamSynchronize(stream));
	auto time2 = std::chrono::high_resolution_clock::now();					// 记录结束时间点
	std::chrono::duration<double, std::milli> duration1 = time2 - time1;		// 计算执行时间（以ms为单位）
	std::cout << "调整坐标时间: " << duration1.count() << " ms" << std::endl;
#endif // CHECK_MESH_BUILD_TIME_COST

	/***************************************** Step.2：计算打乱的xyz键和排序编码 *****************************************/
	generateCode(sampleOrientedPoints, sortCode, count, stream);	// 计算编码
#ifdef CHECK_MESH_BUILD_TIME_COST
	CHECKCUDA(cudaStreamSynchronize(stream));
	auto time3 = std::chrono::high_resolution_clock::now();							// 记录结束时间点
	std::chrono::duration<double, std::milli> duration2 = time3 - time2;			// 计算执行时间（以ms为单位）
	std::cout << "编码时间: " << duration2.count() << " ms" << std::endl;
#endif // CHECK_MESH_BUILD_TIME_COST

	/***************************************** Step.3 && Step.4：对键进行排序 && Unique压缩 *****************************************/
	DeviceArray<OrientedPoint3D<float>> orientedPoints = sampleOrientedPoints.Array();	// 传入可暴露指针
	sortAndCompactVerticesKeys(orientedPoints, stream);									// 稠密点sampleOrientedPoints完成排序
	initUniqueNode(uniqueNodeD, uniqueCode, stream);
#ifdef CHECK_MESH_BUILD_TIME_COST
	CHECKCUDA(cudaStreamSynchronize(stream));
	auto time4 = std::chrono::high_resolution_clock::now();							// 记录结束时间点
	std::chrono::duration<double, std::milli> duration3 = time4 - time3;			// 计算执行时间（以ms为单位）
	std::cout << "初始化D层节点时间: " << duration3.count() << " ms" << std::endl;
#endif // CHECK_MESH_BUILD_TIME_COST

	/***************************************** Step.5 拓展UniqueNode,计算nodeNum和nodeAddress *****************************************/
	generateNodeNumsAndNodeAddress(uniqueCode, nodeNums, nodeAddressD, stream);
#ifdef CHECK_MESH_BUILD_TIME_COST
	CHECKCUDA(cudaStreamSynchronize(stream));
	auto time5 = std::chrono::high_resolution_clock::now();							// 记录结束时间点
	std::chrono::duration<double, std::milli> duration4 = time5 - time4;			// 计算执行时间（以ms为单位）
	std::cout << "生成NodeAddress时间: " << duration4.count() << " ms" << std::endl;
#endif // CHECK_MESH_BUILD_TIME_COST

	/***************************************** Step.6 创建Octree节点数组：NodeArrayD *****************************************/
	buildNodeArrayD(sampleOrientedPoints.ArrayView(), uniqueNodeD.ArrayView(), uniqueCode.ArrayView(), nodeAddressD, NodeAddressFull, Point2NodeArray, NodeArrays[Constants::maxDepth_Host], stream);

#ifdef CHECK_MESH_BUILD_TIME_COST
	CHECKCUDA(cudaStreamSynchronize(stream));
	auto time6 = std::chrono::high_resolution_clock::now();							// 记录结束时间点
	std::chrono::duration<double, std::milli> duration5 = time6 - time5;			// 计算执行时间（以ms为单位）
	std::cout << "构建D层NodeArray时间: " << duration5.count() << " ms" << std::endl;
#endif // CHECK_MESH_BUILD_TIME_COST

	/***************************************** Step.7 构建[0, D - 1]层 *****************************************/
	buildOtherDepthNodeArray(BaseAddressArray_Host, stream);
#ifdef CHECK_MESH_BUILD_TIME_COST
	CHECKCUDA(cudaStreamSynchronize(stream));
	auto time7 = std::chrono::high_resolution_clock::now();							// 记录结束时间点
	std::chrono::duration<double, std::milli> duration6 = time7 - time6;			// 计算执行时间（以ms为单位）
	std::cout << "构建[1, D - 1]层NodeArray时间: " << duration6.count() << " ms" << std::endl;
#endif // CHECK_MESH_BUILD_TIME_COST

	/***************************************** Step.8 更新每个节点的信息，重构NodeArray，同时计算节点Depth查找表，节点中心点Center查找表 *****************************************/
	updateNodeInfo(BaseAddressArray_Host, NodeArray, stream);
#ifdef CHECK_MESH_BUILD_TIME_COST
	CHECKCUDA(cudaStreamSynchronize(stream));
	auto time8 = std::chrono::high_resolution_clock::now();							// 记录结束时间点
	std::chrono::duration<double, std::milli> duration7 = time8 - time7;			// 计算执行时间（以ms为单位）
	std::cout << "更新整棵树时间: " << duration7.count() << " ms" << std::endl;
#endif // CHECK_MESH_BUILD_TIME_COST

	/***************************************** Step.9 构建每个节点的邻居节点 *****************************************/
	computeNodeNeighbor(NodeArray, stream);
	CHECKCUDA(cudaStreamSynchronize(stream));	// 此时需要同步，后续这里的参数要被两个不同流同时使用

	//printf("NodeArrayCount = %d\n", NodeArray.ArraySize());


#ifdef CHECK_MESH_BUILD_TIME_COST
	CHECKCUDA(cudaStreamSynchronize(stream));
	auto time9 = std::chrono::high_resolution_clock::now();							// 记录结束时间点
	std::chrono::duration<double, std::milli> duration8 = time9 - time8;			// 计算执行时间（以ms为单位）
	std::cout << "构建邻居节点时间: " << duration8.count() << " ms" << std::endl;

	auto end = std::chrono::high_resolution_clock::now();							// 记录结束时间点
	std::chrono::duration<double, std::milli> duration = end - start;				// 计算执行时间（以ms为单位）
	std::cout << "构建八叉树时间: " << duration.count() << " ms" << std::endl;			// 输出
	std::cout << std::endl;
	std::cout << "-----------------------------------------------------" << std::endl;	// 输出
	std::cout << std::endl;
#endif // CHECK_MESH_BUILD_TIME_COST

}

void SparseSurfelFusion::BuildOctree::BoundBoxVisualization(const pcl::PointCloud<pcl::PointXYZ>::Ptr cloud, Point3D<float> MaxPoint, Point3D<float> MinPoint)
{
	//----------------可视化--------------
	pcl::visualization::PCLVisualizer::Ptr viewer(new pcl::visualization::PCLVisualizer("Viewer"));	//viewer->initCameraParameters();//设置照相机参数，使用户从默认的角度和方向观察点云
	//设置背景颜色
	viewer->setBackgroundColor(0.3, 0.3, 0.3);
	//设置点云颜色
	pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> single_color(cloud, 0, 225, 0);
	//添加坐标系
	viewer->addCoordinateSystem(0.1);
	viewer->addPointCloud<pcl::PointXYZ>(cloud, single_color, "sample cloud");
	viewer->addCube(MinPoint.coords[0], MaxPoint.coords[0], MinPoint.coords[1], MaxPoint.coords[1], MinPoint.coords[2], MaxPoint.coords[2], 1.0f, 0, 0, "cube");
	while (!viewer->wasStopped())
	{
		viewer->spinOnce(100);
	}
	system("pause");
}
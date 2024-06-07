/*****************************************************************//**
 * \file   BuildOctree.cpp
 * \brief  ����˲�������ʵ��
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

	NodeArray.AllocateBuffer(TOTAL_NODEARRAY_MAX_COUNT);					// Ԥ��10�����ڵ�
	EncodedFunctionNodeIndex.AllocateBuffer(TOTAL_NODEARRAY_MAX_COUNT);
	NodeAddressFull.AllocateBuffer(D_LEVEL_MAX_NODE);						// ����Ӧ����8 * MAX_SURFEL_COUNT

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
	auto start = std::chrono::high_resolution_clock::now();						// ��¼��ʼʱ���
#endif // CHECK_MESH_BUILD_TIME_COST



	size_t count = depthSurfel.Size();
	sampleOrientedPoints.ResizeArrayOrException(count);

	/***************************************** Step.1������BoundingBox *****************************************/
	Point3D<float> MaxPoint = Point3D<float>(float(-1e6), float(-1e6), float(-1e6));// ���ڿ򶨵���x,y,z���ֵ
	Point3D<float> MinPoint = Point3D<float>(float(1e6), float(1e6), float(1e6));	// ���ڿ򶨵���x,y,z��Сֵ
	Point3D<float> center;		// BoundingBox������
	float maxEdge = 1.0f;		// BoundingBox����ı߳�
	float scaleFactor = 1.25f;	// �����ߴ硾����1�������е�����[0, 1]�ķ�Χѹ����С��1�����ƽ�����[0, 1]������1��x,y,z������ == 1��

#ifdef CHECK_MESH_BUILD_TIME_COST
	auto time1 = std::chrono::high_resolution_clock::now();					// ��¼��ʼʱ���
#endif // CHECK_MESH_BUILD_TIME_COST


	// DenseSurfelת��Point3D
	getCoordinateAndNormal(depthSurfel, stream);		
	// ��ð�Χ��(��Լ�㷨)
	getBoundingBox(sampleOrientedPoints.ArrayView(), MaxPoint, MinPoint, stream);

	//// ��Χ�п��ӻ�
	//BoundBoxVisualization(cloud, MaxPoint, MinPoint);

	// ��������scaleFactor����������������λ��
	adjustPointsCoordinateAndNormal(sampleOrientedPoints, MaxPoint, MinPoint, maxEdge, scaleFactor, center, stream);

#ifdef CHECK_MESH_BUILD_TIME_COST
	CHECKCUDA(cudaStreamSynchronize(stream));
	auto time2 = std::chrono::high_resolution_clock::now();					// ��¼����ʱ���
	std::chrono::duration<double, std::milli> duration1 = time2 - time1;		// ����ִ��ʱ�䣨��msΪ��λ��
	std::cout << "��������ʱ��: " << duration1.count() << " ms" << std::endl;
#endif // CHECK_MESH_BUILD_TIME_COST

	/***************************************** Step.2��������ҵ�xyz����������� *****************************************/
	generateCode(sampleOrientedPoints, sortCode, count, stream);	// �������
#ifdef CHECK_MESH_BUILD_TIME_COST
	CHECKCUDA(cudaStreamSynchronize(stream));
	auto time3 = std::chrono::high_resolution_clock::now();							// ��¼����ʱ���
	std::chrono::duration<double, std::milli> duration2 = time3 - time2;			// ����ִ��ʱ�䣨��msΪ��λ��
	std::cout << "����ʱ��: " << duration2.count() << " ms" << std::endl;
#endif // CHECK_MESH_BUILD_TIME_COST

	/***************************************** Step.3 && Step.4���Լ��������� && Uniqueѹ�� *****************************************/
	DeviceArray<OrientedPoint3D<float>> orientedPoints = sampleOrientedPoints.Array();	// ����ɱ�¶ָ��
	sortAndCompactVerticesKeys(orientedPoints, stream);									// ���ܵ�sampleOrientedPoints�������
	initUniqueNode(uniqueNodeD, uniqueCode, stream);
#ifdef CHECK_MESH_BUILD_TIME_COST
	CHECKCUDA(cudaStreamSynchronize(stream));
	auto time4 = std::chrono::high_resolution_clock::now();							// ��¼����ʱ���
	std::chrono::duration<double, std::milli> duration3 = time4 - time3;			// ����ִ��ʱ�䣨��msΪ��λ��
	std::cout << "��ʼ��D��ڵ�ʱ��: " << duration3.count() << " ms" << std::endl;
#endif // CHECK_MESH_BUILD_TIME_COST

	/***************************************** Step.5 ��չUniqueNode,����nodeNum��nodeAddress *****************************************/
	generateNodeNumsAndNodeAddress(uniqueCode, nodeNums, nodeAddressD, stream);
#ifdef CHECK_MESH_BUILD_TIME_COST
	CHECKCUDA(cudaStreamSynchronize(stream));
	auto time5 = std::chrono::high_resolution_clock::now();							// ��¼����ʱ���
	std::chrono::duration<double, std::milli> duration4 = time5 - time4;			// ����ִ��ʱ�䣨��msΪ��λ��
	std::cout << "����NodeAddressʱ��: " << duration4.count() << " ms" << std::endl;
#endif // CHECK_MESH_BUILD_TIME_COST

	/***************************************** Step.6 ����Octree�ڵ����飺NodeArrayD *****************************************/
	buildNodeArrayD(sampleOrientedPoints.ArrayView(), uniqueNodeD.ArrayView(), uniqueCode.ArrayView(), nodeAddressD, NodeAddressFull, Point2NodeArray, NodeArrays[Constants::maxDepth_Host], stream);

#ifdef CHECK_MESH_BUILD_TIME_COST
	CHECKCUDA(cudaStreamSynchronize(stream));
	auto time6 = std::chrono::high_resolution_clock::now();							// ��¼����ʱ���
	std::chrono::duration<double, std::milli> duration5 = time6 - time5;			// ����ִ��ʱ�䣨��msΪ��λ��
	std::cout << "����D��NodeArrayʱ��: " << duration5.count() << " ms" << std::endl;
#endif // CHECK_MESH_BUILD_TIME_COST

	/***************************************** Step.7 ����[0, D - 1]�� *****************************************/
	buildOtherDepthNodeArray(BaseAddressArray_Host, stream);
#ifdef CHECK_MESH_BUILD_TIME_COST
	CHECKCUDA(cudaStreamSynchronize(stream));
	auto time7 = std::chrono::high_resolution_clock::now();							// ��¼����ʱ���
	std::chrono::duration<double, std::milli> duration6 = time7 - time6;			// ����ִ��ʱ�䣨��msΪ��λ��
	std::cout << "����[1, D - 1]��NodeArrayʱ��: " << duration6.count() << " ms" << std::endl;
#endif // CHECK_MESH_BUILD_TIME_COST

	/***************************************** Step.8 ����ÿ���ڵ����Ϣ���ع�NodeArray��ͬʱ����ڵ�Depth���ұ��ڵ����ĵ�Center���ұ� *****************************************/
	updateNodeInfo(BaseAddressArray_Host, NodeArray, stream);
#ifdef CHECK_MESH_BUILD_TIME_COST
	CHECKCUDA(cudaStreamSynchronize(stream));
	auto time8 = std::chrono::high_resolution_clock::now();							// ��¼����ʱ���
	std::chrono::duration<double, std::milli> duration7 = time8 - time7;			// ����ִ��ʱ�䣨��msΪ��λ��
	std::cout << "����������ʱ��: " << duration7.count() << " ms" << std::endl;
#endif // CHECK_MESH_BUILD_TIME_COST

	/***************************************** Step.9 ����ÿ���ڵ���ھӽڵ� *****************************************/
	computeNodeNeighbor(NodeArray, stream);
	CHECKCUDA(cudaStreamSynchronize(stream));	// ��ʱ��Ҫͬ������������Ĳ���Ҫ��������ͬ��ͬʱʹ��

	//printf("NodeArrayCount = %d\n", NodeArray.ArraySize());


#ifdef CHECK_MESH_BUILD_TIME_COST
	CHECKCUDA(cudaStreamSynchronize(stream));
	auto time9 = std::chrono::high_resolution_clock::now();							// ��¼����ʱ���
	std::chrono::duration<double, std::milli> duration8 = time9 - time8;			// ����ִ��ʱ�䣨��msΪ��λ��
	std::cout << "�����ھӽڵ�ʱ��: " << duration8.count() << " ms" << std::endl;

	auto end = std::chrono::high_resolution_clock::now();							// ��¼����ʱ���
	std::chrono::duration<double, std::milli> duration = end - start;				// ����ִ��ʱ�䣨��msΪ��λ��
	std::cout << "�����˲���ʱ��: " << duration.count() << " ms" << std::endl;			// ���
	std::cout << std::endl;
	std::cout << "-----------------------------------------------------" << std::endl;	// ���
	std::cout << std::endl;
#endif // CHECK_MESH_BUILD_TIME_COST

}

void SparseSurfelFusion::BuildOctree::BoundBoxVisualization(const pcl::PointCloud<pcl::PointXYZ>::Ptr cloud, Point3D<float> MaxPoint, Point3D<float> MinPoint)
{
	//----------------���ӻ�--------------
	pcl::visualization::PCLVisualizer::Ptr viewer(new pcl::visualization::PCLVisualizer("Viewer"));	//viewer->initCameraParameters();//���������������ʹ�û���Ĭ�ϵĽǶȺͷ���۲����
	//���ñ�����ɫ
	viewer->setBackgroundColor(0.3, 0.3, 0.3);
	//���õ�����ɫ
	pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> single_color(cloud, 0, 225, 0);
	//�������ϵ
	viewer->addCoordinateSystem(0.1);
	viewer->addPointCloud<pcl::PointXYZ>(cloud, single_color, "sample cloud");
	viewer->addCube(MinPoint.coords[0], MaxPoint.coords[0], MinPoint.coords[1], MaxPoint.coords[1], MinPoint.coords[2], MaxPoint.coords[2], 1.0f, 0, 0, "cube");
	while (!viewer->wasStopped())
	{
		viewer->spinOnce(100);
	}
	system("pause");
}
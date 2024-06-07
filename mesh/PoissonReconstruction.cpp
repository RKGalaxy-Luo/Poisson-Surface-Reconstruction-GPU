/*****************************************************************//**
 * \file   PoissonReconstruction.cpp
 * \brief  GPU��Ⲵ�������ؽ�
 * 
 * \author LUOJIAXUAN
 * \date   May 4th 2024
 *********************************************************************/
#include "PoissonReconstruction.h"

SparseSurfelFusion::PoissonReconstruction::PoissonReconstruction()
{
	initCudaStream();	// ��ʼ��ִ��mesh�����cuda��
	
	DrawConstructedMesh = std::make_shared<DrawMesh>();

	cloud = std::make_shared<pcl::PointCloud<pcl::PointXYZ>>();
	normals = std::make_shared<pcl::PointCloud<pcl::Normal>>();

	OctreePtr = std::make_shared<BuildOctree>();
	VectorFieldPtr = std::make_shared<ComputeVectorField>(MeshStream[0]);	// ��ʼ����ʱ�򼴹��������
	NodeDivergencePtr = std::make_shared<ComputeNodesDivergence>();
	LaplacianSolverPtr = std::make_shared<LaplacianSolver>();
	MeshGeometryPtr = std::make_shared<BuildMeshGeometry>();
	TriangleIndicesPtr = std::make_shared<ComputeTriangleIndices>();

	DenseSurfel.AllocateBuffer(MAX_SURFEL_COUNT);
	PointNormalDevice.AllocateBuffer(MAX_SURFEL_COUNT);
	PointCloudDevice.AllocateBuffer(MAX_SURFEL_COUNT);
}

SparseSurfelFusion::PoissonReconstruction::~PoissonReconstruction()
{
	releaseCudaStream();
	DenseSurfel.ReleaseBuffer();
	PointNormalDevice.ReleaseBuffer();
	PointCloudDevice.ReleaseBuffer();
}

void SparseSurfelFusion::PoissonReconstruction::initCudaStream()
{
	for (int i = 0; i < MAX_MESH_STREAM; i++) {
		CHECKCUDA(cudaStreamCreate(&(MeshStream[i])));
	}
}

void SparseSurfelFusion::PoissonReconstruction::releaseCudaStream()
{
	for (int i = 0; i < MAX_MESH_STREAM; i++) {
		CHECKCUDA(cudaStreamDestroy(MeshStream[i]));
		MeshStream[i] = 0;
	}
}

void SparseSurfelFusion::PoissonReconstruction::synchronizeAllCudaStream()
{
	for (int i = 0; i < MAX_MESH_STREAM; i++) {
		CHECKCUDA(cudaStreamSynchronize(MeshStream[i]));
	}
}

void SparseSurfelFusion::PoissonReconstruction::readTXTFile(std::string path)
{
	FILE* file;
	if (fopen_s(&file, path.c_str(), "r") != 0) LOGGING(FATAL) << "���ƶ�ȡ����";	// ֻ��
	else {
		int count = 0;
		float num;
		while (fscanf_s(file, "%f", &num) == 1) {
			PointCloud.push_back(num);
			count++;
		}
		pointsNum = count / 3;
		std::cout << "�ܹ���ȡ���Ƹ�����" << pointsNum << std::endl;
		fclose(file);
	}
	PointCloudDevice.ResizeArrayOrException(pointsNum * 3);
	CHECKCUDA(cudaMemcpy(PointCloudDevice.Array().ptr(), PointCloud.data(), sizeof(float) * pointsNum * 3, cudaMemcpyHostToDevice));
	buildDenseSurfel(PointCloudDevice, DenseSurfel);
}

void SparseSurfelFusion::PoissonReconstruction::readPCDFile(std::string path)
{
	// ��ȡ PCD �ļ�

	pcl::io::loadPCDFile(path, *cloud);
	CalculatePointCloudNormal(cloud, normals);
	printf("�������� = %zu   �������� = %zu\n", cloud->size(), normals->size());
	std::cout << std::endl;
	std::cout << "-----------------------------------------------------" << std::endl;	// ���
	std::cout << std::endl;

	const unsigned int pointsNum = cloud->size();
	PointCloudDevice.ResizeArrayOrException(cloud->size());
	PointNormalDevice.ResizeArrayOrException(normals->size());
	DenseSurfel.ResizeArrayOrException(pointsNum);
	CHECKCUDA(cudaMemcpy(PointCloudDevice.Array().ptr(), cloud->data(), sizeof(pcl::PointXYZ) * pointsNum, cudaMemcpyHostToDevice));
	CHECKCUDA(cudaMemcpy(PointNormalDevice.Array().ptr(), normals->data(), sizeof(pcl::Normal) * pointsNum, cudaMemcpyHostToDevice));
	buildDenseSurfel(PointCloudDevice, PointNormalDevice, DenseSurfel);
	CHECKCUDA(cudaDeviceSynchronize());

	//// ���� PCL ���ӻ�����
	//pcl::visualization::PCLVisualizer::Ptr viewer(new pcl::visualization::PCLVisualizer("3D Viewer"));
	//// ��ӵ��Ƶ����ӻ�����
	//viewer->addPointCloud(cloud, "sample cloud");
	//// ��ʾ����
	//while (!viewer->wasStopped())
	//{
	//	viewer->spinOnce(100);
	//	std::this_thread::sleep_for(std::chrono::milliseconds(100));
	//}
	
}

void SparseSurfelFusion::PoissonReconstruction::readPLYFile(std::string path)
{
	// ����PCL���ӻ�����
	pcl::visualization::PCLVisualizer viewer("PLY Model Viewer");
	// ��ȡPLY�ļ�
	pcl::PolygonMesh mesh;
	pcl::io::loadPLYFile(path, mesh);
	// �������ģ��
	viewer.addPolygonMesh(mesh, "mesh");
	// ��ʾ���ӻ�����
	while (!viewer.wasStopped())
	{
		viewer.spinOnce();
	}
	system("pause");
	
}

void SparseSurfelFusion::PoissonReconstruction::CalculatePointCloudNormal(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud, pcl::PointCloud<pcl::Normal>::Ptr normals)
{

	// ��¼��ʼʱ���
	auto start = std::chrono::high_resolution_clock::now();
	//------------------���㷨��----------------------
	pcl::NormalEstimationOMP<pcl::PointXYZ, pcl::Normal> n;//OMP����
	//����kdtree�����н��ڵ㼯����
	pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>());
	n.setNumberOfThreads(100);//����openMP���߳���
	//n.setViewPoint(0,0,0);//�����ӵ㣬Ĭ��Ϊ��0��0��0��
	n.setInputCloud(cloud);
	n.setSearchMethod(tree);
	n.setKSearch(10);//���Ʒ������ʱ����Ҫ���ѵĽ��ڵ��С
	//n.setRadiusSearch(0.01);//�뾶����
	n.compute(*normals);//��ʼ���з����

	// ��¼����ʱ���
	auto end = std::chrono::high_resolution_clock::now();
	// ����ִ��ʱ�䣨��msΪ��λ��
	std::chrono::duration<double, std::milli> duration = end - start;
	// ���ִ��ʱ��
	std::cout << "���߼���ʱ��: " << duration.count() / 60000.0f << " ����" << std::endl;

	saveCloudWithNormal(cloud, normals);

	////----------------���ӻ�--------------
	//boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer(new pcl::visualization::PCLVisualizer("Normal viewer"));
	////viewer->initCameraParameters();//���������������ʹ�û���Ĭ�ϵĽǶȺͷ���۲����
	////���ñ�����ɫ
	//viewer->setBackgroundColor(0.3, 0.3, 0.3);
	//viewer->addText("NORMAL", 10, 10, "text");
	////���õ�����ɫ
	//pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> single_color(cloud, 0, 225, 0);
	////�������ϵ
	//viewer->addCoordinateSystem(0.1);
	//viewer->addPointCloud<pcl::PointXYZ>(cloud, single_color, "sample cloud");


	////�����Ҫ��ʾ�ĵ��Ʒ���cloudΪԭʼ����ģ�ͣ�normalΪ������Ϣ��20��ʾ��Ҫ��ʾ����ĵ��Ƽ������ÿ20������ʾһ�η���0.02��ʾ���򳤶ȡ�
	//viewer->addPointCloudNormals<pcl::PointXYZ, pcl::Normal>(cloud, normals, 1, 0.01, "normals");
	////���õ��ƴ�С
	//viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2, "sample cloud");
	//while (!viewer->wasStopped())
	//{
	//	viewer->spinOnce(100);
	//	boost::this_thread::sleep(boost::posix_time::microseconds(100000));
	//}

}



SparseSurfelFusion::DeviceArrayView<SparseSurfelFusion::DepthSurfel> SparseSurfelFusion::PoissonReconstruction::getDenseSurfel()
{
	return DenseSurfel.ArrayView();
}

void SparseSurfelFusion::PoissonReconstruction::SolvePoissionReconstructionMesh(DeviceArrayView<DepthSurfel> denseSurfel, CoredVectorMeshData& mesh)
{

	const unsigned int DenseSurfelCount = denseSurfel.Size();
	OctreePtr->BuildNodesArray(denseSurfel, cloud, normals, MeshStream[0]);						// ����Octree
	DeviceArrayView<OrientedPoint3D<float>> orientedPoints = OctreePtr->GetOrientedPoints();	// ����������
	DeviceArrayView<OctNode> OctreeNodeArray = OctreePtr->GetOctreeNodeArray();					// ��ð˲�����NodeArray
	const int* NodeArrayCount = OctreePtr->GetNodeArrayCount();									// ���ÿһ��ڵ����������һ��maxDepth��С������
	const int* BaseAddressArray = OctreePtr->GetBaseAddressArray();								// ���ÿ��ڵ��������е�ƫ��(ÿ���һ���ڵ��������е�λ��)
	DeviceArrayView<int> BaseAddressArrayDevice = OctreePtr->GetBaseAddressArrayDevice();
	// ����ڵ���롢����VectorField�����ɶ���Arrayͬʱ����
	OctreePtr->ComputeEncodedFunctionNodeIndex(MeshStream[0]);									// ����ڵ����������
	VectorFieldPtr->BuildVectorField(orientedPoints, OctreeNodeArray, NodeArrayCount, BaseAddressArray, MeshStream[1]);	// ����VectorField
	CHECKCUDA(cudaDeviceSynchronize());	// �����㷨��ɣ�ͬ������GPU��VectorFieldPtr�е�һЩ����������Ҫ����
	DeviceArrayView<unsigned int> NodeArrayDepthIndex = OctreePtr->GetNodeArrayDepthIndex();
	DeviceArrayView<Point3D<float>> NodeArrayNodeCenter = OctreePtr->GetNodeArrayNodeCenter();
	DeviceBufferArray<OctNode>& OctreeNodeArrayHandle = OctreePtr->GetOctreeNodeArrayHandle();
	//pool->AddTask([&]() { MeshGeometryPtr->GenerateVertexArray(OctreeNodeArrayHandle, NodeArrayDepthIndex, NodeArrayNodeCenter, MeshStream[2]); });
	//pool->AddTask([&]() { MeshGeometryPtr->GenerateEdgeArray(OctreeNodeArrayHandle, BaseAddressArray[Constants::maxDepth_Host], NodeArrayCount[Constants::maxDepth_Host], NodeArrayDepthIndex, NodeArrayNodeCenter, MeshStream[3]); });
	//pool->AddTask([&]() { MeshGeometryPtr->GenerateFaceArray(OctreeNodeArrayHandle, NodeArrayDepthIndex, NodeArrayNodeCenter, MeshStream[4]); });
	MeshGeometryPtr->GenerateVertexArray(OctreeNodeArrayHandle, NodeArrayDepthIndex, NodeArrayNodeCenter, MeshStream[2]);
	MeshGeometryPtr->GenerateEdgeArray(OctreeNodeArrayHandle, BaseAddressArray[Constants::maxDepth_Host], NodeArrayCount[Constants::maxDepth_Host], NodeArrayDepthIndex, NodeArrayNodeCenter, MeshStream[3]);
	MeshGeometryPtr->GenerateFaceArray(OctreeNodeArrayHandle, NodeArrayDepthIndex, NodeArrayNodeCenter, MeshStream[4]);

	DeviceArrayView<int> encodeNodeIndexInFunction = OctreePtr->GetEncodedFunctionNodeIndex();
	DeviceArrayView<Point3D<float>> vectorField = VectorFieldPtr->GetVectorField();
	DeviceArrayView<double> dot_F_dF = VectorFieldPtr->GetValueTable_Dot_F_dF();
	DeviceArrayView<double> dot_F_F = VectorFieldPtr->GetValueTable_Dot_F_F();
	DeviceArrayView<double> dot_F_d2F = VectorFieldPtr->GetValueTable_Dot_F_d2F();
	DeviceArrayView<ConfirmedPPolynomial<CONVTIMES + 1, CONVTIMES + 2>> baseFunctions = VectorFieldPtr->GetBaseFunction();
	//pool->AddTask([=]() { NodeDivergencePtr->CalculateNodesDivergence(BaseAddressArray, NodeArrayCount, BaseAddressArrayDevice, encodeNodeIndexInFunction, OctreeNodeArray, vectorField, dot_F_dF, MeshStream[0], MeshStream[1]); });
	NodeDivergencePtr->CalculateNodesDivergence(BaseAddressArray, NodeArrayCount, BaseAddressArrayDevice, encodeNodeIndexInFunction, OctreeNodeArray, vectorField, dot_F_dF, MeshStream[0], MeshStream[1]);
	CHECKCUDA(cudaDeviceSynchronize());	// �����㷨��ɣ�ͬ������GPU���˴���Ҫͬ������Ϊ������Ҫ����dot_F_dF��dot_F_F��dot_F_d2F��DivergencePtr
	float* DivergencePtr = NodeDivergencePtr->GetDivergenceRawPtr();
	DeviceArrayView<int> Point2NodeArray = OctreePtr->GetPoint2NodeArray();

	//pool->AddTask([&]() { LaplacianSolverPtr->LaplacianCGSolver(BaseAddressArray, NodeArrayCount, encodeNodeIndexInFunction, OctreeNodeArray, DivergencePtr, dot_F_F, dot_F_d2F, MeshStream[0]); });
	//pool->AddTask([&]() { LaplacianSolverPtr->CalculatePointsImplicitFunctionValue(orientedPoints, Point2NodeArray, OctreeNodeArray, encodeNodeIndexInFunction, baseFunctions, BaseAddressArray[Constants::maxDepth_Host], DenseSurfelCount, MeshStream[0]); });
	LaplacianSolverPtr->LaplacianCGSolver(BaseAddressArray, NodeArrayCount, encodeNodeIndexInFunction, OctreeNodeArray, DivergencePtr, dot_F_F, dot_F_d2F, MeshStream[0]);
	LaplacianSolverPtr->CalculatePointsImplicitFunctionValue(orientedPoints, Point2NodeArray, OctreeNodeArray, encodeNodeIndexInFunction, baseFunctions, BaseAddressArray[Constants::maxDepth_Host], DenseSurfelCount, MeshStream[0]);
	CHECKCUDA(cudaDeviceSynchronize());	// �����㷨��ɣ�ͬ������GPU

	DeviceArrayView<VertexNode> vertexArray = MeshGeometryPtr->GetVertexArray();
	DeviceArrayView<EdgeNode> edgeArray = MeshGeometryPtr->GetEdgeArray();
	DeviceArrayView<FaceNode> faceArray = MeshGeometryPtr->GetFaceArray();
	DeviceArrayView<float> dx = LaplacianSolverPtr->GetDx();
	const float isoValue = LaplacianSolverPtr->GetIsoValue();
	TriangleIndicesPtr->calculateTriangleIndices(vertexArray, edgeArray, faceArray, OctreeNodeArrayHandle, baseFunctions, dx, encodeNodeIndexInFunction, NodeArrayDepthIndex, NodeArrayNodeCenter, isoValue, BaseAddressArray[Constants::maxDepth_Host], NodeArrayCount[Constants::maxDepth_Host], mesh, MeshStream[0]);

	CHECKCUDA(cudaDeviceSynchronize());	// �����㷨��ɣ�ͬ������GPU
}


void SparseSurfelFusion::PoissonReconstruction::DrawRebuildMesh(CoredVectorMeshData& mesh)
{
	DrawConstructedMesh->CalculateMeshNormals(mesh);
	DrawConstructedMesh->DrawRenderedMesh(mesh);
}

void SparseSurfelFusion::PoissonReconstruction::saveCloudWithNormal(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud, pcl::PointCloud<pcl::Normal>::Ptr normals)
{
	// �������з��ߵĵ���
	pcl::PointCloud<pcl::PointNormal>::Ptr cloud_with_normals(new pcl::PointCloud<pcl::PointNormal>());
	for (size_t i = 0; i < cloud->points.size(); ++i) {
		pcl::PointNormal point_normal;
		point_normal.x = cloud->points[i].x;
		point_normal.y = cloud->points[i].y;
		point_normal.z = cloud->points[i].z;
		point_normal.normal_x = normals->points[i].normal_x;
		point_normal.normal_y = normals->points[i].normal_y;
		point_normal.normal_z = normals->points[i].normal_z;
		cloud_with_normals->points.push_back(point_normal);
	}

	// ���õ��ƿ�Ⱥ͸߶�
	cloud_with_normals->width = cloud_with_normals->points.size();
	cloud_with_normals->height = 1;
	cloud_with_normals->is_dense = true;

	// ���浽 .ply �ļ�
	pcl::PLYWriter writer;
	writer.write(PlySavePath, *cloud_with_normals, false); // true ��ʾ������ģʽ

	std::cout << "������Ʒ���ply�ļ�" << std::endl;
}
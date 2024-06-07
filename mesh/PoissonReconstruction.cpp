/*****************************************************************//**
 * \file   PoissonReconstruction.cpp
 * \brief  GPU求解泊松曲面重建
 * 
 * \author LUOJIAXUAN
 * \date   May 4th 2024
 *********************************************************************/
#include "PoissonReconstruction.h"

SparseSurfelFusion::PoissonReconstruction::PoissonReconstruction()
{
	initCudaStream();	// 初始化执行mesh任务的cuda流
	
	DrawConstructedMesh = std::make_shared<DrawMesh>();

	cloud = std::make_shared<pcl::PointCloud<pcl::PointXYZ>>();
	normals = std::make_shared<pcl::PointCloud<pcl::Normal>>();

	OctreePtr = std::make_shared<BuildOctree>();
	VectorFieldPtr = std::make_shared<ComputeVectorField>(MeshStream[0]);	// 初始化的时候即构建点积表
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
	if (fopen_s(&file, path.c_str(), "r") != 0) LOGGING(FATAL) << "点云读取错误";	// 只读
	else {
		int count = 0;
		float num;
		while (fscanf_s(file, "%f", &num) == 1) {
			PointCloud.push_back(num);
			count++;
		}
		pointsNum = count / 3;
		std::cout << "总共读取点云个数：" << pointsNum << std::endl;
		fclose(file);
	}
	PointCloudDevice.ResizeArrayOrException(pointsNum * 3);
	CHECKCUDA(cudaMemcpy(PointCloudDevice.Array().ptr(), PointCloud.data(), sizeof(float) * pointsNum * 3, cudaMemcpyHostToDevice));
	buildDenseSurfel(PointCloudDevice, DenseSurfel);
}

void SparseSurfelFusion::PoissonReconstruction::readPCDFile(std::string path)
{
	// 读取 PCD 文件

	pcl::io::loadPCDFile(path, *cloud);
	CalculatePointCloudNormal(cloud, normals);
	printf("点云数量 = %zu   法线数量 = %zu\n", cloud->size(), normals->size());
	std::cout << std::endl;
	std::cout << "-----------------------------------------------------" << std::endl;	// 输出
	std::cout << std::endl;

	const unsigned int pointsNum = cloud->size();
	PointCloudDevice.ResizeArrayOrException(cloud->size());
	PointNormalDevice.ResizeArrayOrException(normals->size());
	DenseSurfel.ResizeArrayOrException(pointsNum);
	CHECKCUDA(cudaMemcpy(PointCloudDevice.Array().ptr(), cloud->data(), sizeof(pcl::PointXYZ) * pointsNum, cudaMemcpyHostToDevice));
	CHECKCUDA(cudaMemcpy(PointNormalDevice.Array().ptr(), normals->data(), sizeof(pcl::Normal) * pointsNum, cudaMemcpyHostToDevice));
	buildDenseSurfel(PointCloudDevice, PointNormalDevice, DenseSurfel);
	CHECKCUDA(cudaDeviceSynchronize());

	//// 创建 PCL 可视化对象
	//pcl::visualization::PCLVisualizer::Ptr viewer(new pcl::visualization::PCLVisualizer("3D Viewer"));
	//// 添加点云到可视化对象
	//viewer->addPointCloud(cloud, "sample cloud");
	//// 显示点云
	//while (!viewer->wasStopped())
	//{
	//	viewer->spinOnce(100);
	//	std::this_thread::sleep_for(std::chrono::milliseconds(100));
	//}
	
}

void SparseSurfelFusion::PoissonReconstruction::readPLYFile(std::string path)
{
	// 创建PCL可视化对象
	pcl::visualization::PCLVisualizer viewer("PLY Model Viewer");
	// 读取PLY文件
	pcl::PolygonMesh mesh;
	pcl::io::loadPLYFile(path, mesh);
	// 添加网格模型
	viewer.addPolygonMesh(mesh, "mesh");
	// 显示可视化界面
	while (!viewer.wasStopped())
	{
		viewer.spinOnce();
	}
	system("pause");
	
}

void SparseSurfelFusion::PoissonReconstruction::CalculatePointCloudNormal(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud, pcl::PointCloud<pcl::Normal>::Ptr normals)
{

	// 记录开始时间点
	auto start = std::chrono::high_resolution_clock::now();
	//------------------计算法线----------------------
	pcl::NormalEstimationOMP<pcl::PointXYZ, pcl::Normal> n;//OMP加速
	//建立kdtree来进行近邻点集搜索
	pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>());
	n.setNumberOfThreads(100);//设置openMP的线程数
	//n.setViewPoint(0,0,0);//设置视点，默认为（0，0，0）
	n.setInputCloud(cloud);
	n.setSearchMethod(tree);
	n.setKSearch(10);//点云法向计算时，需要所搜的近邻点大小
	//n.setRadiusSearch(0.01);//半径搜素
	n.compute(*normals);//开始进行法向计

	// 记录结束时间点
	auto end = std::chrono::high_resolution_clock::now();
	// 计算执行时间（以ms为单位）
	std::chrono::duration<double, std::milli> duration = end - start;
	// 输出执行时间
	std::cout << "法线计算时间: " << duration.count() / 60000.0f << " 分钟" << std::endl;

	saveCloudWithNormal(cloud, normals);

	////----------------可视化--------------
	//boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer(new pcl::visualization::PCLVisualizer("Normal viewer"));
	////viewer->initCameraParameters();//设置照相机参数，使用户从默认的角度和方向观察点云
	////设置背景颜色
	//viewer->setBackgroundColor(0.3, 0.3, 0.3);
	//viewer->addText("NORMAL", 10, 10, "text");
	////设置点云颜色
	//pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> single_color(cloud, 0, 225, 0);
	////添加坐标系
	//viewer->addCoordinateSystem(0.1);
	//viewer->addPointCloud<pcl::PointXYZ>(cloud, single_color, "sample cloud");


	////添加需要显示的点云法向。cloud为原始点云模型，normal为法向信息，20表示需要显示法向的点云间隔，即每20个点显示一次法向，0.02表示法向长度。
	//viewer->addPointCloudNormals<pcl::PointXYZ, pcl::Normal>(cloud, normals, 1, 0.01, "normals");
	////设置点云大小
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
	OctreePtr->BuildNodesArray(denseSurfel, cloud, normals, MeshStream[0]);						// 构建Octree
	DeviceArrayView<OrientedPoint3D<float>> orientedPoints = OctreePtr->GetOrientedPoints();	// 获得有向点云
	DeviceArrayView<OctNode> OctreeNodeArray = OctreePtr->GetOctreeNodeArray();					// 获得八叉树的NodeArray
	const int* NodeArrayCount = OctreePtr->GetNodeArrayCount();									// 获得每一层节点的数量，是一个maxDepth大小的数组
	const int* BaseAddressArray = OctreePtr->GetBaseAddressArray();								// 获得每层节点在数组中的偏移(每层第一个节点在数组中的位置)
	DeviceArrayView<int> BaseAddressArrayDevice = OctreePtr->GetBaseAddressArrayDevice();
	// 计算节点编码、构建VectorField、生成顶点Array同时进行
	OctreePtr->ComputeEncodedFunctionNodeIndex(MeshStream[0]);									// 计算节点基函数索引
	VectorFieldPtr->BuildVectorField(orientedPoints, OctreeNodeArray, NodeArrayCount, BaseAddressArray, MeshStream[1]);	// 构建VectorField
	CHECKCUDA(cudaDeviceSynchronize());	// 所有算法完成，同步整个GPU，VectorFieldPtr中的一些参数后面需要调用
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
	CHECKCUDA(cudaDeviceSynchronize());	// 所有算法完成，同步整个GPU，此处需要同步，因为后面需要调用dot_F_dF、dot_F_F、dot_F_d2F、DivergencePtr
	float* DivergencePtr = NodeDivergencePtr->GetDivergenceRawPtr();
	DeviceArrayView<int> Point2NodeArray = OctreePtr->GetPoint2NodeArray();

	//pool->AddTask([&]() { LaplacianSolverPtr->LaplacianCGSolver(BaseAddressArray, NodeArrayCount, encodeNodeIndexInFunction, OctreeNodeArray, DivergencePtr, dot_F_F, dot_F_d2F, MeshStream[0]); });
	//pool->AddTask([&]() { LaplacianSolverPtr->CalculatePointsImplicitFunctionValue(orientedPoints, Point2NodeArray, OctreeNodeArray, encodeNodeIndexInFunction, baseFunctions, BaseAddressArray[Constants::maxDepth_Host], DenseSurfelCount, MeshStream[0]); });
	LaplacianSolverPtr->LaplacianCGSolver(BaseAddressArray, NodeArrayCount, encodeNodeIndexInFunction, OctreeNodeArray, DivergencePtr, dot_F_F, dot_F_d2F, MeshStream[0]);
	LaplacianSolverPtr->CalculatePointsImplicitFunctionValue(orientedPoints, Point2NodeArray, OctreeNodeArray, encodeNodeIndexInFunction, baseFunctions, BaseAddressArray[Constants::maxDepth_Host], DenseSurfelCount, MeshStream[0]);
	CHECKCUDA(cudaDeviceSynchronize());	// 所有算法完成，同步整个GPU

	DeviceArrayView<VertexNode> vertexArray = MeshGeometryPtr->GetVertexArray();
	DeviceArrayView<EdgeNode> edgeArray = MeshGeometryPtr->GetEdgeArray();
	DeviceArrayView<FaceNode> faceArray = MeshGeometryPtr->GetFaceArray();
	DeviceArrayView<float> dx = LaplacianSolverPtr->GetDx();
	const float isoValue = LaplacianSolverPtr->GetIsoValue();
	TriangleIndicesPtr->calculateTriangleIndices(vertexArray, edgeArray, faceArray, OctreeNodeArrayHandle, baseFunctions, dx, encodeNodeIndexInFunction, NodeArrayDepthIndex, NodeArrayNodeCenter, isoValue, BaseAddressArray[Constants::maxDepth_Host], NodeArrayCount[Constants::maxDepth_Host], mesh, MeshStream[0]);

	CHECKCUDA(cudaDeviceSynchronize());	// 所有算法完成，同步整个GPU
}


void SparseSurfelFusion::PoissonReconstruction::DrawRebuildMesh(CoredVectorMeshData& mesh)
{
	DrawConstructedMesh->CalculateMeshNormals(mesh);
	DrawConstructedMesh->DrawRenderedMesh(mesh);
}

void SparseSurfelFusion::PoissonReconstruction::saveCloudWithNormal(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud, pcl::PointCloud<pcl::Normal>::Ptr normals)
{
	// 创建带有法线的点云
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

	// 设置点云宽度和高度
	cloud_with_normals->width = cloud_with_normals->points.size();
	cloud_with_normals->height = 1;
	cloud_with_normals->is_dense = true;

	// 保存到 .ply 文件
	pcl::PLYWriter writer;
	writer.write(PlySavePath, *cloud_with_normals, false); // true 表示二进制模式

	std::cout << "保存点云法线ply文件" << std::endl;
}
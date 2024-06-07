/*****************************************************************//**
 * \file   ComputeTriangleIndices.cpp
 * \brief  �����޸��������񣬹�������
 * 
 * \author LUOJIAXUAN
 * \date   June 3rd 2024
 *********************************************************************/
#include "ComputeTriangleIndices.h"

SparseSurfelFusion::ComputeTriangleIndices::ComputeTriangleIndices()
{
	vvalue.AllocateBuffer(TOTAL_VERTEXARRAY_MAX_COUNT);
	vexNums.AllocateBuffer(TOTAL_EDGEARRAY_MAX_COUNT);
	vexAddress.AllocateBuffer(TOTAL_EDGEARRAY_MAX_COUNT);
	triNums.AllocateBuffer(D_LEVEL_MAX_NODE);
	triAddress.AllocateBuffer(D_LEVEL_MAX_NODE);
	cubeCatagory.AllocateBuffer(D_LEVEL_MAX_NODE);
	markValidVertex.AllocateBuffer(TOTAL_EDGEARRAY_MAX_COUNT);
	SubdivideNode.AllocateBuffer(TOTAL_NODEARRAY_MAX_COUNT - D_LEVEL_MAX_NODE);				// ��maxDepth������ڵ�����
	markValidSubdividedNode.AllocateBuffer(TOTAL_NODEARRAY_MAX_COUNT - D_LEVEL_MAX_NODE);	// ��maxDepth������ڵ�����
	SubdivideDepthBuffer.AllocateBuffer(TOTAL_NODEARRAY_MAX_COUNT - D_LEVEL_MAX_NODE);
	markValidSubdivideVertex.AllocateBuffer(int(1 << 21));									// �������Ϊ8^7
	markValidSubdivideEdge.AllocateBuffer(int(1 << 21));
	markValidSubdivedeVexNum.AllocateBuffer(int(1 << 21));
	markValidFinerVexArray.AllocateBuffer(int(1 << 22));
	markValidFinerEdge.AllocateBuffer(int(1 << 22));
	markValidFinerVexNum.AllocateBuffer(int(1 << 22));
}

SparseSurfelFusion::ComputeTriangleIndices::~ComputeTriangleIndices()
{
	vvalue.ReleaseBuffer();
	vexNums.ReleaseBuffer();
	vexAddress.ReleaseBuffer();
	triNums.ReleaseBuffer();
	triAddress.ReleaseBuffer();
	cubeCatagory.ReleaseBuffer();
	markValidVertex.ReleaseBuffer();
	SubdivideNode.ReleaseBuffer();
	markValidSubdividedNode.ReleaseBuffer();
	SubdivideDepthBuffer.DeviceArray().release();
	SubdivideDepthBuffer.HostArray().clear();
	markValidSubdivideVertex.ReleaseBuffer();
	markValidSubdivideEdge.ReleaseBuffer();
	markValidSubdivedeVexNum.ReleaseBuffer();
	markValidFinerVexArray.ReleaseBuffer();
	markValidFinerEdge.ReleaseBuffer();
	markValidFinerVexNum.ReleaseBuffer();
}

void SparseSurfelFusion::ComputeTriangleIndices::calculateTriangleIndices(DeviceArrayView<VertexNode> VertexArray, DeviceArrayView<EdgeNode> EdgeArray, DeviceArrayView<FaceNode> FaceArray, DeviceBufferArray<OctNode>& NodeArray, DeviceArrayView<ConfirmedPPolynomial<CONVTIMES + 1, CONVTIMES + 2>> BaseFunction, DeviceArrayView<float> dx, DeviceArrayView<int> encodeNodeIndexInFunction, DeviceArrayView<unsigned int> DepthBuffer, DeviceArrayView<Point3D<float>> CenterBuffer, const unsigned int isoValue, const unsigned int DLevelOffset, const unsigned int DLevelNodeCount, CoredVectorMeshData& mesh, cudaStream_t stream)
{
#ifdef CHECK_MESH_BUILD_TIME_COST
	auto time1 = std::chrono::high_resolution_clock::now();					// ��¼��ʼʱ���
#endif // CHECK_MESH_BUILD_TIME_COST

	//printf("VertexCount = %d   EdgeCount = %d   FaceCount = %d\n", VertexArray.Size(), EdgeArray.Size(), FaceArray.Size());

	/**************************** Step 1: ����˲����������ʽ����ֵ ****************************/
	ComputeVertexImplicitFunctionValue(VertexArray, NodeArray.ArrayView(), BaseFunction, dx, encodeNodeIndexInFunction, isoValue, stream);
#ifdef CHECK_MESH_BUILD_TIME_COST
	CHECKCUDA(cudaStreamSynchronize(stream));
	auto time2 = std::chrono::high_resolution_clock::now();					// ��¼����ʱ���
	std::chrono::duration<double, std::milli> duration1 = time2 - time1;		// ����ִ��ʱ�䣨��msΪ��λ��
	std::cout << "���㶥����ʽ����ֵ��ʱ��: " << duration1.count() << " ms" << std::endl;
#endif // CHECK_MESH_BUILD_TIME_COST

	/**************************** Step 2: ����ߵ������Լ�ƫ�� ****************************/
	generateVertexNumsAndVertexAddress(EdgeArray, NodeArray.ArrayView(), vvalue.ArrayView(), DLevelOffset, stream);
#ifdef CHECK_MESH_BUILD_TIME_COST
	CHECKCUDA(cudaStreamSynchronize(stream));
	auto time3 = std::chrono::high_resolution_clock::now();							// ��¼����ʱ���
	std::chrono::duration<double, std::milli> duration2 = time3 - time2;			// ����ִ��ʱ�䣨��msΪ��λ��
	std::cout << "���㶥�������Լ�����λ��ƫ�Ƶ�ʱ��: " << duration2.count() << " ms" << std::endl;
#endif // CHECK_MESH_BUILD_TIME_COST

	/**************************** Step 3: ���������������Լ���ַ ****************************/
	generateTriangleNumsAndTriangleAddress(NodeArray.ArrayView(), vvalue.ArrayView(), DLevelOffset, DLevelNodeCount, mesh, stream);
#ifdef CHECK_MESH_BUILD_TIME_COST
	CHECKCUDA(cudaStreamSynchronize(stream));
	auto time4 = std::chrono::high_resolution_clock::now();							// ��¼����ʱ���
	std::chrono::duration<double, std::milli> duration3 = time4 - time3;			// ����ִ��ʱ�䣨��msΪ��λ��
	std::cout << "���������ʷ������Լ�������ƫ��λ�õ�ʱ��: " << duration3.count() << " ms" << std::endl;
#endif // CHECK_MESH_BUILD_TIME_COST

	/**************************** Step 4 & 5: ���ɽڵ㼰���������� ****************************/
	generateVerticesAndTriangle(NodeArray, VertexArray, EdgeArray, FaceArray, DLevelOffset, DLevelNodeCount, mesh, stream);
#ifdef CHECK_MESH_BUILD_TIME_COST
	CHECKCUDA(cudaStreamSynchronize(stream));
	auto time5 = std::chrono::high_resolution_clock::now();							// ��¼����ʱ���
	std::chrono::duration<double, std::milli> duration4 = time5 - time4;			// ����ִ��ʱ�䣨��msΪ��λ��
	std::cout << "���ɽڵ������Mesh��ʱ��: " << duration4.count() << " ms" << std::endl;
#endif // CHECK_MESH_BUILD_TIME_COST

	/**************************** Step 6: ����ϸ�ֽڵ����鼰��ÿ��ϸ�ֽڵ�ƫ�ƺ�ϸ�ֽڵ����� ****************************/
	generateSubdivideNodeArrayCountAndAddress(NodeArray, DepthBuffer, DLevelOffset, stream);
#ifdef CHECK_MESH_BUILD_TIME_COST
	CHECKCUDA(cudaStreamSynchronize(stream));
	auto time6 = std::chrono::high_resolution_clock::now();							// ��¼����ʱ���
	std::chrono::duration<double, std::milli> duration5 = time6 - time5;			// ����ִ��ʱ�䣨��msΪ��λ��
	std::cout << "����ϸ�ֽڵ����鼰��ÿ��ƫ�ƺ�������ʱ��: " << duration5.count() << " ms" << std::endl;
#endif // CHECK_MESH_BUILD_TIME_COST

	/**************************** Step 7: ����ֽڵ�ϸ���ع�����Coarser��Finer���������������߳�������ִ�С� ****************************/
	CoarserSubdivideNodeAndRebuildMesh(NodeArray, DepthBuffer, CenterBuffer, BaseFunction, dx, encodeNodeIndexInFunction, isoValue, mesh, stream);
#ifdef CHECK_MESH_BUILD_TIME_COST	
	CHECKCUDA(cudaStreamSynchronize(stream));
	auto time7 = std::chrono::high_resolution_clock::now();							// ��¼����ʱ���
	std::chrono::duration<double, std::milli> duration6 = time7 - time6;			// ����ִ��ʱ�䣨��msΪ��λ��
	std::cout << "Coarser�ڵ�ϸ���ع������ʱ��: " << duration6.count() << " ms" << std::endl;
#endif // CHECK_MESH_BUILD_TIME_COST

	/**************************** Step 8: �����ڵ�ϸ���ع�����Coarser��Finer���������������߳�������ִ�С� ****************************/
	FinerSubdivideNodeAndRebuildMesh(NodeArray, DepthBuffer, CenterBuffer, BaseFunction, dx, encodeNodeIndexInFunction, isoValue, mesh, stream);
#ifdef CHECK_MESH_BUILD_TIME_COST		
	CHECKCUDA(cudaStreamSynchronize(stream));
	auto time8 = std::chrono::high_resolution_clock::now();							// ��¼����ʱ���
	std::chrono::duration<double, std::milli> duration7 = time8 - time7;			// ����ִ��ʱ�䣨��msΪ��λ��
	std::cout << "Finer�ڵ�ϸ���ع������ʱ��: " << duration7.count() << " ms" << std::endl;
#endif // CHECK_MESH_BUILD_TIME_COST

#ifdef CHECK_MESH_BUILD_TIME_COST
	CHECKCUDA(cudaStreamSynchronize(stream));	// ��ͬ��
	auto time9 = std::chrono::high_resolution_clock::now();						// ��¼����ʱ���
	std::chrono::duration<double, std::milli> duration8 = time9 - time1;			// ����ִ��ʱ�䣨��msΪ��λ��
	std::cout << "���������ʷ�������������ʱ��: " << duration8.count() << " ms" << std::endl;		// ���
	std::cout << std::endl;
	std::cout << "-----------------------------------------------------" << std::endl;	// ���
	std::cout << std::endl;
#endif // CHECK_MESH_BUILD_TIME_COST
}

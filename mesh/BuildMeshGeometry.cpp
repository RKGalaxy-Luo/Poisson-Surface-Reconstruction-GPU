/*****************************************************************//**
 * \file   BuildMeshGeometry.cpp
 * \brief  ���������㷨
 * 
 * \author LUOJIAXUAN
 * \date   June 1st 2024
 *********************************************************************/
#include "BuildMeshGeometry.h"

SparseSurfelFusion::BuildMeshGeometry::BuildMeshGeometry()
{
	VertexArray.AllocateBuffer(TOTAL_VERTEXARRAY_MAX_COUNT);	// NodeArray��С��8��(�������Ԫ��50w��˴��ķ�2.09G)		����ʵ�����й����У�����Ҫһ��previous���������յ�Array����˴˴���Ҫ��2������GPU���Դ��Ƿ������㷨�� 
	EdgeArray.AllocateBuffer(TOTAL_EDGEARRAY_MAX_COUNT);		// maxDepth��ڵ�������12��(�������Ԫ��50w��˴��ķ�1.07G)	����ʵ�����й����У�����Ҫһ��previous���������յ�Array����˴˴���Ҫ��2������GPU���Դ��Ƿ������㷨�� 
	FaceArray.AllocateBuffer(TOTAL_FACEARRAY_MAX_COUNT);		// NodeArray��С��6��(�������Ԫ��50w��˴��ķ�0.559G)		����ʵ�����й����У�����Ҫһ��previous���������յ�Array����˴˴���Ҫ��2������GPU���Դ��Ƿ������㷨�� 

	markValidVertexArray.AllocateBuffer(TOTAL_VERTEXARRAY_MAX_COUNT);
	markValidEdgeArray.AllocateBuffer(TOTAL_EDGEARRAY_MAX_COUNT);
	markValidFaceArray.AllocateBuffer(TOTAL_FACEARRAY_MAX_COUNT);
}

SparseSurfelFusion::BuildMeshGeometry::~BuildMeshGeometry()
{
	VertexArray.ReleaseBuffer();
	EdgeArray.ReleaseBuffer();
	FaceArray.ReleaseBuffer();

	markValidVertexArray.ReleaseBuffer();
	markValidEdgeArray.ReleaseBuffer();
	markValidFaceArray.ReleaseBuffer();
}

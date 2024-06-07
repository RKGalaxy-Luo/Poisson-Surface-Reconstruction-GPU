/*****************************************************************//**
 * \file   BuildMeshGeometry.cpp
 * \brief  构建网格算法
 * 
 * \author LUOJIAXUAN
 * \date   June 1st 2024
 *********************************************************************/
#include "BuildMeshGeometry.h"

SparseSurfelFusion::BuildMeshGeometry::BuildMeshGeometry()
{
	VertexArray.AllocateBuffer(TOTAL_VERTEXARRAY_MAX_COUNT);	// NodeArray大小的8倍(若最大面元是50w则此处耗费2.09G)		【在实际运行过程中，是需要一个previous来生成最终的Array，因此此处需要乘2来决定GPU的显存是否满足算法】 
	EdgeArray.AllocateBuffer(TOTAL_EDGEARRAY_MAX_COUNT);		// maxDepth层节点数量的12倍(若最大面元是50w则此处耗费1.07G)	【在实际运行过程中，是需要一个previous来生成最终的Array，因此此处需要乘2来决定GPU的显存是否满足算法】 
	FaceArray.AllocateBuffer(TOTAL_FACEARRAY_MAX_COUNT);		// NodeArray大小的6倍(若最大面元是50w则此处耗费0.559G)		【在实际运行过程中，是需要一个previous来生成最终的Array，因此此处需要乘2来决定GPU的显存是否满足算法】 

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

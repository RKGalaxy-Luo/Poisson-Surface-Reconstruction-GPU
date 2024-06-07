/*****************************************************************//**
 * \file   LaplacianSolver.cpp
 * \brief  拉普拉斯求解器
 * 
 * \author LUOJIAXUAN
 * \date   May 26th 2024
 *********************************************************************/
#include "LaplacianSolver.h"

SparseSurfelFusion::LaplacianSolver::LaplacianSolver()
{
	//pool = std::make_shared<ThreadPool>(Constants::maxDepth_Host);
	dx.AllocateBuffer(TOTAL_NODEARRAY_MAX_COUNT);
	DensePointsImplicitFunctionValue.AllocateBuffer(MAX_SURFEL_COUNT);
}

SparseSurfelFusion::LaplacianSolver::~LaplacianSolver()
{
	dx.ReleaseBuffer();
	DensePointsImplicitFunctionValue.ReleaseBuffer();
}



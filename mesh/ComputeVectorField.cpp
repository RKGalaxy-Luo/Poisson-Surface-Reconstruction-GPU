/*****************************************************************//**
 * \file   ComputeVectorField.cpp
 * \brief  计算向量场方法
 * 
 * \author LUOJIAXUAN
 * \date   May 15th 2024
 *********************************************************************/
#pragma once
#include "ComputeVectorField.h"

SparseSurfelFusion::ComputeVectorField::ComputeVectorField(cudaStream_t stream)
{
    AllocateBuffer();
    BuildInnerProductTable(stream);
}

SparseSurfelFusion::ComputeVectorField::~ComputeVectorField()
{
    dot_F_F.ReleaseBuffer();
    dot_F_DF.ReleaseBuffer();
    dot_F_D2F.ReleaseBuffer();
    baseFunctions_Device.ReleaseBuffer();
    BaseFunctionMaxDepth_Device.ReleaseBuffer();
    VectorField.ReleaseBuffer();
}

void SparseSurfelFusion::ComputeVectorField::AllocateBuffer()
{
    dot_F_F.AllocateBuffer(F_DATA_RES_SQUARE);
    dot_F_DF.AllocateBuffer(F_DATA_RES_SQUARE);
    dot_F_D2F.AllocateBuffer(F_DATA_RES_SQUARE);
    baseFunctions_Device.AllocateBuffer(F_DATA_RES);
    BaseFunctionMaxDepth_Device.AllocateBuffer(sizeof(ConfirmedPPolynomial<CONVTIMES, CONVTIMES + 2>));
    VectorField.AllocateBuffer(D_LEVEL_MAX_NODE);
}

void SparseSurfelFusion::ComputeVectorField::BuildVectorField(DeviceArrayView<OrientedPoint3D<float>> orientedPoints, DeviceArrayView<OctNode> NodeArray, const int* NodeArrayCount, const int* BaseAddressArray, cudaStream_t stream)
{
#ifdef CHECK_MESH_BUILD_TIME_COST
    auto start = std::chrono::high_resolution_clock::now();						// 记录开始时间点
#endif // CHECK_MESH_BUILD_TIME_COST


    ConfirmedPPolynomial<CONVTIMES, CONVTIMES + 2> BaseFunctionMaxDepth(ReconstructionFunction.scale(1.0 / (1 << Constants::maxDepth_Host)));
    
    //for (int i = 0; i < CONVTIMES + 2; i++) {
    //    printf("第 %d 个多项式系数：coefficients = %.5f   %.5f\n", i, BaseFunctionMaxDepth.polys[i].p.coefficients[0], BaseFunctionMaxDepth.polys[i].p.coefficients[1]);
    //}
    
    BaseFunctionMaxDepth_Device.ResizeArrayOrException(sizeof(BaseFunctionMaxDepth));
    CHECKCUDA(cudaMemcpyAsync(BaseFunctionMaxDepth_Device.Array().ptr(), &BaseFunctionMaxDepth, sizeof(BaseFunctionMaxDepth), cudaMemcpyHostToDevice, stream));
    VectorField.ResizeArrayOrException(NodeArrayCount[Constants::maxDepth_Host]);
    CalculateVectorField(BaseFunctionMaxDepth_Device, orientedPoints, NodeArray, BaseAddressArray[Constants::maxDepth_Host], NodeArrayCount[Constants::maxDepth_Host], VectorField, stream);


    ///** Check VectorField **/
    //CHECKCUDA(cudaStreamSynchronize(stream));
    //std::vector<Point3D<float>> vectorFieldTest;
    //VectorField.ArrayView().Download(vectorFieldTest);
    //for (int i = 0; i < vectorFieldTest.size(); i++) {
    //    if (i % 1000 == 0 || i < 1000)   printf("idx = %d   VectorField = (%.10f, %.10f, %.10f)\n", i, vectorFieldTest[i].coords[0], vectorFieldTest[i].coords[1], vectorFieldTest[i].coords[2]);
    //}

#ifdef CHECK_MESH_BUILD_TIME_COST
    CHECKCUDA(cudaStreamSynchronize(stream));
    auto end = std::chrono::high_resolution_clock::now();						// 记录结束时间点
    std::chrono::duration<double, std::milli> duration = end - start;			// 计算执行时间（以ms为单位）
    std::cout << "构建向量场时间: " << duration.count() << " ms" << std::endl;		// 输出
    std::cout << std::endl;
    std::cout << "-----------------------------------------------------" << std::endl;	// 输出
    std::cout << std::endl;
#endif // CHECK_MESH_BUILD_TIME_COST
}



void SparseSurfelFusion::ComputeVectorField::BuildInnerProductTable(cudaStream_t stream)
{
    auto start = std::chrono::high_resolution_clock::now();						// 记录开始时间点

    ReconstructionFunction = PPolynomial<CONVTIMES>::GaussianApproximation();
    FunctionData<CONVTIMES, double> fData;
    fData.set(Constants::maxDepth_Host, ReconstructionFunction, normalize, 0);
    fData.setDotTables(fData.DOT_FLAG | fData.D_DOT_FLAG | fData.D2_DOT_FLAG);
    PPolynomial<CONVTIMES>& F = ReconstructionFunction;
    switch (normalize) {
    case 2:
        F = F / sqrt((F * F).integral(F.polys[0].start, F.polys[F.polyCount - 1].start));
        break;
    case 1:
        F = F / F.integral(F.polys[0].start, F.polys[F.polyCount - 1].start);
        break;
    default:
        F = F / F(0);
    }

    dot_F_F.ResizeArrayOrException(fData.res * fData.res);
    CHECKCUDA(cudaMemcpyAsync(dot_F_F.Array().ptr(), fData.dotTable, sizeof(double) * fData.res * fData.res, cudaMemcpyHostToDevice, stream));
    dot_F_DF.ResizeArrayOrException(fData.res * fData.res);
    CHECKCUDA(cudaMemcpyAsync(dot_F_DF.Array().ptr(), fData.dDotTable, sizeof(double) * fData.res * fData.res, cudaMemcpyHostToDevice, stream));
    dot_F_D2F.ResizeArrayOrException(fData.res * fData.res);
    CHECKCUDA(cudaMemcpyAsync(dot_F_D2F.Array().ptr(), fData.d2DotTable, sizeof(double) * fData.res * fData.res, cudaMemcpyHostToDevice, stream));

    fData.clearDotTables(fData.DOT_FLAG | fData.D_DOT_FLAG | fData.D2_DOT_FLAG);

    std::vector<ConfirmedPPolynomial<CONVTIMES + 1, CONVTIMES + 2>> baseFunctions_Host;
    baseFunctions_Host.resize(fData.res);
    for (int i = 0; i < fData.res; i++) {
        baseFunctions_Host[i] = fData.baseFunctions[i];
    }
    CHECKCUDA(cudaStreamSynchronize(stream));
    CHECKCUDA(cudaMemcpyAsync(baseFunctions_Device.Array().ptr(), baseFunctions_Host.data(), sizeof(ConfirmedPPolynomial<CONVTIMES + 1, CONVTIMES + 2>) * fData.res, cudaMemcpyHostToDevice, stream));

    CHECKCUDA(cudaStreamSynchronize(stream));
    auto end = std::chrono::high_resolution_clock::now();						// 记录结束时间点
    std::chrono::duration<double, std::milli> duration = end - start;			// 计算执行时间（以ms为单位）
    std::cout << "预先计算点积表时间: " << duration.count() << " ms" << std::endl;		// 输出
    std::cout << std::endl;
    std::cout << "-----------------------------------------------------" << std::endl;	// 输出
    std::cout << std::endl;
}

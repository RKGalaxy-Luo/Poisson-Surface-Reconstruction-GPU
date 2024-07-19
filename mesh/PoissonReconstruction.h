/*****************************************************************//**
 * \file   PoissonReconstruction.h
 * \brief  GPU求解泊松曲面重建
 * 
 * \author LUOJIAXUAN
 * \date   May 4th 2024
 *********************************************************************/
#pragma once

#include <iostream>
#include <stdlib.h>
#include <stdio.h>

#include <pcl/io/ply_io.h>  // ply 文件读取头文件
#include <pcl/visualization/cloud_viewer.h>

#include <pcl/io/pcd_io.h>
#include <pcl/point_cloud.h>
#include <pcl/visualization/pcl_visualizer.h>

#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/features/normal_3d_omp.h>
#include <boost/thread/thread.hpp>

#include <random>
#include <chrono>
#include <thread>
#include <base/ThreadPool.h>
#include <curand_kernel.h>

#include "BuildOctree.h"
#include "BuildMeshGeometry.h"
#include "ComputeVectorField.h"
#include "ComputeNodesDivergence.h"
#include "solver/LaplacianSolver.h"
#include "ComputeTriangleIndices.h"

#include "DrawMesh.h"

namespace SparseSurfelFusion {
	namespace device {

		/**
		 * \brief 构建稠密面元.
		 * 
		 * \param coor 传入点云三维坐标
		 * \param surfel 坐标写入面元
		 * \param pointsNum 面元数量
		 */
		__global__ void buildDenseSurfelKernel(pcl::PointXYZ* coor, DepthSurfel* surfel, const unsigned int pointsNum);

		/**
		 * \brief 构建有法向量的稠密点云.
		 * 
		 * \param coor 传入点云三维坐标
		 * \param normal 法向量
		 * \param surfel 坐标写入面元
		 * \param cudaStates 用以生成随机数
		 * \param pointsNum 面元数量
		 */
		__global__ void buildOrientedDenseSurfelKernel(pcl::PointXYZ* coor, pcl::Normal* normal, DepthSurfel* surfel, curandState* cudaStates, const unsigned int pointsNum);
	}
	class PoissonReconstruction
	{
	public:
		PoissonReconstruction();

		~PoissonReconstruction();

	private:
		/**
		 * \brief 初始化mesh的所有cuda流.
		 * 
		 */
		void initCudaStream();

		/**
		 * \brief 释放cuda流.
		 * 
		 */
		void releaseCudaStream();

		/**
		 * \brief  同步所有cuda流.
		 * 
		 */
		void synchronizeAllCudaStream();

		BuildOctree::Ptr OctreePtr;							// 构建八叉树
		ComputeVectorField::Ptr VectorFieldPtr;				// 构建向量场(typename修饰告诉编译器这个是一个类)
		ComputeNodesDivergence::Ptr NodeDivergencePtr;		// 计算节点散度
		LaplacianSolver::Ptr LaplacianSolverPtr;			// Laplace求解器
		BuildMeshGeometry::Ptr MeshGeometryPtr;				// 网格构建顶点、边、面三种元素
		ComputeTriangleIndices::Ptr TriangleIndicesPtr;		// 三角剖分构建索引
		DrawMesh::Ptr DrawConstructedMesh;					// OpenGL绘制被构建的网格

	public:
		/**
		 * \brief 读取点云数据测试算法.
		 * 
		 * \param path 点云数据路径
		 */
		void readTXTFile(std::string path);

		/**
		 * \brief 读取pcd文件.
		 * 
		 * \param path 文件路径
		 */
		void readPCDFile(std::string path);

		/**
		 * \brief 读取ply文件.
		 * 
		 * \param path 文件路径
		 */
		void readPLYFile(std::string path);

		/**
		 * \brief 初始化八叉树的树结构.
		 *
		 * \param denseSurfel 传入稠密点
		 */
		void SolvePoissionReconstructionMesh(DeviceArrayView<DepthSurfel> denseSurfel);

		/**
		 * \brief OpenGL绘制重建的网格.
		 */
		void DrawRebuildMesh();

		DeviceArrayView<DepthSurfel> getDenseSurfel();

	private:

		std::shared_ptr<ThreadPool> pool;

		cudaStream_t MeshStream[MAX_MESH_STREAM];

		std::vector<float> PointCloud;
		unsigned int pointsNum = 0;
		DeviceBufferArray<pcl::PointXYZ> PointCloudDevice;
		DeviceBufferArray<pcl::Normal> PointNormalDevice;
		DeviceBufferArray<float3> PointCloudColor;
		DeviceBufferArray<DepthSurfel> DenseSurfel;

		pcl::PointCloud<pcl::PointXYZ>::Ptr cloud;
		pcl::PointCloud<pcl::Normal>::Ptr normals;

		curandState* cudaStates = NULL;		// 用以生成随机数


		/**
		 * \brief 传入点云计算法线.
		 *
		 * \param cloud 传入点云
		 * \param normals 计算得到法线
		 */
		void CalculatePointCloudNormal(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud, pcl::PointCloud<pcl::Normal>::Ptr normals);




		/**
		 * \brief .
		 * 
		 * \param PointCloudDevice
		 * \param DenseSurfel
		 * \param stream 
		 */
		void buildDenseSurfel(DeviceBufferArray<pcl::PointXYZ>& PointCloudDevice, DeviceBufferArray<DepthSurfel>& DenseSurfel, cudaStream_t stream = 0);

		/**
		 * \brief.
		 * 
		 * \param PointCloudDevice
		 * \param PointNormalDevice
		 * \param DenseSurfel
		 * \param stream
		 */
		void buildDenseSurfel(DeviceBufferArray<pcl::PointXYZ>& PointCloudDevice, DeviceBufferArray<pcl::Normal>& PointNormalDevice, DeviceBufferArray<DepthSurfel>& DenseSurfel, cudaStream_t stream = 0);


		void saveCloudWithNormal(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud, pcl::PointCloud<pcl::Normal>::Ptr normals);
	};
}



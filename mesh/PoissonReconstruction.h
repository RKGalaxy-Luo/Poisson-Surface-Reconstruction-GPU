/*****************************************************************//**
 * \file   PoissonReconstruction.h
 * \brief  GPU��Ⲵ�������ؽ�
 * 
 * \author LUOJIAXUAN
 * \date   May 4th 2024
 *********************************************************************/
#pragma once

#include <iostream>
#include <stdlib.h>
#include <stdio.h>

#include <pcl/io/ply_io.h>  // ply �ļ���ȡͷ�ļ�
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
		 * \brief ����������Ԫ.
		 * 
		 * \param coor ���������ά����
		 * \param surfel ����д����Ԫ
		 * \param pointsNum ��Ԫ����
		 */
		__global__ void buildDenseSurfelKernel(pcl::PointXYZ* coor, DepthSurfel* surfel, const unsigned int pointsNum);

		/**
		 * \brief �����з������ĳ��ܵ���.
		 * 
		 * \param coor ���������ά����
		 * \param normal ������
		 * \param surfel ����д����Ԫ
		 * \param cudaStates �������������
		 * \param pointsNum ��Ԫ����
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
		 * \brief ��ʼ��mesh������cuda��.
		 * 
		 */
		void initCudaStream();

		/**
		 * \brief �ͷ�cuda��.
		 * 
		 */
		void releaseCudaStream();

		/**
		 * \brief  ͬ������cuda��.
		 * 
		 */
		void synchronizeAllCudaStream();

		BuildOctree::Ptr OctreePtr;							// �����˲���
		ComputeVectorField::Ptr VectorFieldPtr;				// ����������(typename���θ��߱����������һ����)
		ComputeNodesDivergence::Ptr NodeDivergencePtr;		// ����ڵ�ɢ��
		LaplacianSolver::Ptr LaplacianSolverPtr;			// Laplace�����
		BuildMeshGeometry::Ptr MeshGeometryPtr;				// ���񹹽����㡢�ߡ�������Ԫ��
		ComputeTriangleIndices::Ptr TriangleIndicesPtr;		// �����ʷֹ�������
		DrawMesh::Ptr DrawConstructedMesh;					// OpenGL���Ʊ�����������

	public:
		/**
		 * \brief ��ȡ�������ݲ����㷨.
		 * 
		 * \param path ��������·��
		 */
		void readTXTFile(std::string path);

		/**
		 * \brief ��ȡpcd�ļ�.
		 * 
		 * \param path �ļ�·��
		 */
		void readPCDFile(std::string path);

		/**
		 * \brief ��ȡply�ļ�.
		 * 
		 * \param path �ļ�·��
		 */
		void readPLYFile(std::string path);

		/**
		 * \brief ��ʼ���˲��������ṹ.
		 *
		 * \param denseSurfel ������ܵ�
		 */
		void SolvePoissionReconstructionMesh(DeviceArrayView<DepthSurfel> denseSurfel);

		/**
		 * \brief OpenGL�����ؽ�������.
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

		curandState* cudaStates = NULL;		// �������������


		/**
		 * \brief ������Ƽ��㷨��.
		 *
		 * \param cloud �������
		 * \param normals ����õ�����
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



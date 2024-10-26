/*****************************************************************//**
 * \file   ComputeNodesDivergence.h
 * \brief  ����ڵ��ɢ��
 * 
 * \author LUOJIAXUAN
 * \date   May 24th 2024
 *********************************************************************/
#pragma once
#include <chrono>
#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>
#include <mesh/BuildOctree.h>
#include <core/AlgorithmTypes.h>
#include <mesh/Geometry.h>
#include <base/DeviceReadWrite/DeviceBufferArray.h>

namespace SparseSurfelFusion {

	namespace device {
		/**
		 * \brief ���㾫ϸ�ڵ��ɢ�Ⱥ˺���.
		 * 
		 * \param BaseAddressArray �ڵ�ƫ������
		 * \param encodeNodeIndexInFunction ����ڵ���ں���������
		 * \param NodeArray �˲���һά�ڵ�
		 * \param VectorField ������
		 * \param dot_F_DF �����
		 * \param begin ��NodeArrayƫ�ƿ�ʼ
		 * \param calculatedNodeNum ��Ҫ�������Ľڵ�����
		 * \param Divergence �ڵ�ɢ��
		 */
		__global__ void computeFinerNodesDivergenceKernel(DeviceArrayView<int> BaseAddressArray, DeviceArrayView<int> encodeNodeIndexInFunction, DeviceArrayView<OctNode> NodeArray, DeviceArrayView<Point3D<float>> VectorField, DeviceArrayView<double> dot_F_DF, const unsigned int begin, const unsigned int calculatedNodeNum, float* Divergence);

		/**
		 * \brief �����������.
		 * 
		 * \param p1 ����1
		 * \param p2 ����2
		 */
		__device__ float DotProduct(const Point3D<float>& p1, const Point3D<float>& p2);

		/**
		 * \brief ����ýڵ㼰���ھӽڵ������ǵ�maxDepth��ڵ������.
		 * 
		 * \param NodeArray �˲���һά�ڵ�
		 * \param index ��ǰ����NodeArray�е�λ��
		 * \param coverNums ���ǵĽڵ�����
		 */
		__global__ void computeCoverNums(DeviceArrayView<OctNode> NodeArray, const unsigned int index, unsigned int* coverNums);

		/**
		 * \brief ���ɣ�idx = [0, totalCoverNum)��D��ڵ㣬����Щ�ڵ�ӳ�䵽D���Ӧ��D_idx.
		 * 
		 * \param NodeArray �˲���һά�ڵ�
		 * \param index ��ǰ����NodeArray�е�λ��
		 * \param coverNums ���ǵĽڵ�����
		 * \param totalCoverNum ��ǰ�ڵ㼰���ھӽڵ㸲�ǵ�D��ڵ�Ľڵ�����
		 * \param DLevelIndexArray ӳ���ϵ����
		 */
		__global__ void generateDLevelIndexArrayKernel(DeviceArrayView<OctNode> NodeArray, const unsigned int index, const unsigned int* coverNums, const unsigned int totalCoverNum, unsigned int* DLevelIndexArray);

		/**
		 * \brief ���㵱ǰ�ڵ�ɢ�ȵĺ˺���.
		 * 
		 * \param BaseAddressArrayDevice �ڵ�ƫ������
		 * \param encodeNodeIndexInFunction ����ڵ���ں���������
		 * \param VectorField ������
		 * \param dot_F_DF �����
		 * \param index ��ǰ����Ҫ����Ľڵ���NodeArray�е�index
		 * \param DLevelIndexArray ӳ���ϵ����
		 * \param totalCoverNum ��ǰ�ڵ㼰���ھӽڵ㸲�ǵ�D��ڵ�Ľڵ�����
		 * \param divg ��Ҫ�����ɢ��ֵ�����ֵ
		 */
		__global__ void computeCoarserNodesDivergenceKernel(DeviceArrayView<int> BaseAddressArrayDevice, DeviceArrayView<int> encodeNodeIndexInFunction, DeviceArrayView<Point3D<float>> VectorField, DeviceArrayView<double> dot_F_DF, const unsigned int index, const unsigned int* DLevelIndexArray, const unsigned int totalCoverNum, float* divg);
	}
	/**
	 * \brief ����ڵ��ɢ��.
	 */
	class ComputeNodesDivergence
	{
	public:
		ComputeNodesDivergence();

		~ComputeNodesDivergence();

		using Ptr = std::shared_ptr<ComputeNodesDivergence>;


		/**
		 * \brief ����ڵ��ɢ�ȡ��������ֱ���ִ�У�ʱ�����44%��.
		 *
		 * \param BaseAddressArrayDevice �ڵ�ƫ������(Host)
		 * \param NodeArrayCount ÿһ��ڵ�����(Host)
		 * \param encodeNodeIndexInFunction ����ڵ���ں���������
		 * \param NodeArray �˲���һά�ڵ�
		 * \param VectorField ������
		 * \param dot_F_DF �����
		 * \param stream_1 cuda��1
		 * \param stream_2 cuda��2
		 */
		void CalculateNodesDivergence(const int* BaseAddressArray, const int* NodeArrayCount, DeviceArrayView<int> BaseAddressArrayDevice, DeviceArrayView<int> encodeNodeIndexInFunction, DeviceArrayView<OctNode> NodeArray, DeviceArrayView<Point3D<float>> VectorField, DeviceArrayView<double> dot_F_DF, cudaStream_t stream_1, cudaStream_t stream_2);

		/**
		 * \brief ��ýڵ�ɢ��(ֻ��).
		 * 
		 * \return NodeArray�нڵ��ɢ��
		 */
		DeviceArrayView<float> GetDivergenceView() { return Divergence.ArrayView(); }

		/**
		 * \brief ��ýڵ�ɢ��ָ��.
		 * 
		 * \return �ڵ�ɢ��ָ��
		 */
		float* GetDivergenceRawPtr() { return Divergence.Array().ptr(); }
	private:

		DeviceBufferArray<float> Divergence;			// �ڵ��ɢ��

		/**
		 * \brief ���㾫ϸ�ڵ��ɢ�ȡ�����[maxDepth - FinerLevelNum, maxDepth]��Ľڵ�ɢ�ȣ����������30%��������1�롿.
		 *
		 * \param BaseAddressArrayDevice �ڵ�ƫ������(GPU)
		 * \param encodeNodeIndexInFunction ����ڵ���ں���������
		 * \param NodeArray �˲���һά�ڵ�
		 * \param VectorField ������
		 * \param dot_F_DF �����
		 * \param left �������İ˲����ڵ��������߽�index  ���������ڵ�index�����䷶ΧΪ[left, right]��
		 * \param right �������İ˲����ڵ�������ұ߽�index ���������ڵ�index�����䷶ΧΪ[left, right]��
		 * \param cuda��
		 */
		void computeFinerNodesDivergence(DeviceArrayView<int> BaseAddressArrayDevice, DeviceArrayView<int> encodeNodeIndexInFunction, DeviceArrayView<OctNode> NodeArray, DeviceArrayView<Point3D<float>> VectorField, DeviceArrayView<double> dot_F_DF, const unsigned int left, const unsigned int right, cudaStream_t stream);

		/**
		 * \brief ����ֲڽڵ��ɢ�ȡ�����[1, CoarserLevelNum]��ڵ��ɢ�ȡ����������̡߳�.
		 *
		 * \param BaseAddressArrayDevice �ڵ�ƫ������(GPU)
		 * \param encodeNodeIndexInFunction ����ڵ���ں���������
		 * \param NodeArray �˲���һά�ڵ�
		 * \param VectorField ������
		 * \param dot_F_DF �����
		 * \param left �������İ˲����ڵ��������߽�index  ���������ڵ�index�����䷶ΧΪ[left, right]��
		 * \param right �������İ˲����ڵ�������ұ߽�index ���������ڵ�index�����䷶ΧΪ[left, right]��
		 * \param stream cuda��
		 */
		void computeCoarserNodesDivergence(const int* BaseAddressArray, DeviceArrayView<int> BaseAddressArrayDevice, DeviceArrayView<int> encodeNodeIndexInFunction, DeviceArrayView<OctNode> NodeArray, DeviceArrayView<Point3D<float>> VectorField, DeviceArrayView<double> dot_F_DF, const unsigned int left, const unsigned int right, cudaStream_t stream);
	};
}



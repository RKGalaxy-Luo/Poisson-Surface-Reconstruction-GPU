/*****************************************************************//**
 * \file   ComputeVectorField.h
 * \brief  ����������
 * 
 * \author LUOJIAXUAN
 * \date   May 15th 2024
 *********************************************************************/
#pragma once
#include <chrono>
#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>
#include <math/PPolynomial.h>
#include <base/Constants.h>
#include <base/GlobalConfigs.h>
#include <base/DeviceReadWrite/DeviceBufferArray.h>
#include "ConfirmedPPolynomial.h"
#include "FunctionData.h"
#include "Geometry.h"
#include "BuildOctree.h"
#include "BinaryNode.h"

namespace SparseSurfelFusion {
	namespace device {

		__device__ float FCenterWidthPoint(int idx, const ConfirmedPPolynomial<CONVTIMES, CONVTIMES + 2>& BaseFunctionMaxDepth_d, const Point3D<float>& center, const float& width, const Point3D<float>& point);

		__device__ void getFunctionIdxNode(const int& key, const int& maxDepth, int* idx);

		__global__ void CalculateVectorFieldKernel(ConfirmedPPolynomial<CONVTIMES, CONVTIMES + 2>* BaseFunctionMaxDepth_Device, DeviceArrayView<OrientedPoint3D<float>> DenseOrientedPoints, DeviceArrayView<OctNode> NodeArray, const unsigned int DLevelOffset, const unsigned int DLevelNodeNum, Point3D<float>* VectorField);
	}

	class ComputeVectorField
	{
	public:

		ComputeVectorField(cudaStream_t stream);

		~ComputeVectorField();

		using Ptr = std::shared_ptr<ComputeVectorField>;

		/**
		 * \brief ����VectorField����������.
		 * 
		 * \param orientedPoints ���������
		 * \param NodeArray �˲����ڵ�����
		 * \param NodeArrayCount ÿһ��ڵ������
		 * \param BaseAddressArray ÿһ���׽ڵ��ƫ��
		 * \param stream cuda��
		 */
		void BuildVectorField(DeviceArrayView<OrientedPoint3D<float>> orientedPoints, DeviceArrayView<OctNode> NodeArray, const int* NodeArrayCount, const int* BaseAddressArray, cudaStream_t stream);

		/**
		 * \brief ���������.
		 * 
		 * \return ������
		 */
		DeviceArrayView<Point3D<float>> GetVectorField() { return VectorField.ArrayView(); }
			
		/**
		 * \brief ��õ����<F, F>.
		 * 
		 * \return �����<F, F>.
		 */
		DeviceArrayView<double> GetValueTable_Dot_F_F() { return dot_F_F.ArrayView(); }

		/**
		 * \brief ���һ�׵��������<F, dF>.
		 * 
		 * \return һ�׵��������<F, dF>.
		 */
		DeviceArrayView<double> GetValueTable_Dot_F_dF() { return dot_F_DF.ArrayView(); }

		/**
		 * \brief ��ö��׵��������<F, d2F>.
		 * 
		 * \return ���׵��������<F, d2F>
		 */
		DeviceArrayView<double> GetValueTable_Dot_F_d2F() { return dot_F_D2F.ArrayView(); }
		/**
		 * \brief ��û�����.
		 * 
		 * \return ������
		 */
		DeviceArrayView<ConfirmedPPolynomial<CONVTIMES + 1, CONVTIMES + 2>> GetBaseFunction() { return baseFunctions_Device.ArrayView(); }
	private:

		const int normalize = NORMALIZE;

		PPolynomial<CONVTIMES> ReconstructionFunction;		// Ԥ�ȼ����ؽ�������

		void AllocateBuffer();

		/**
		 * \brief Ԥ�ȼ��㹹�������.
		 * 
		 * \param stream cuda��
		 */
		void BuildInnerProductTable(cudaStream_t stream);


		DeviceBufferArray<double> dot_F_F; 	   // �������ĵ����
		DeviceBufferArray<double> dot_F_DF;	   // ������һ�׵����ĵ����
		DeviceBufferArray<double> dot_F_D2F;   // ���������׵����ĵ����
		DeviceBufferArray<ConfirmedPPolynomial<CONVTIMES + 1, CONVTIMES + 2>> baseFunctions_Device;		// ������������GPU
		DeviceBufferArray<ConfirmedPPolynomial<CONVTIMES, CONVTIMES + 2>> BaseFunctionMaxDepth_Device;	// ����Ļ�������������ÿһ��[0, maxDepth]��������ÿһ��Ľڵ㶼����֮ƥ��Ļ�����ƥ�䡿
		DeviceBufferArray<Point3D<float>> VectorField;	// ������(��1������)

		/**
		 * \brief ����VectorField.
		 *
		 * \param BaseFunctionMaxDepth_Device ����Ļ�����
		 * \param DenseOrientedPoints ���������
		 * \param NodeArray �˲����ڵ�����
		 * \param DLevelOffset D���׽ڵ���NodeArray�е�ƫ��
		 * \param DLevelNodeNum D��ڵ�����
		 * \param VectorField ������
		 * \param stream cuda��
		 */
		void CalculateVectorField(ConfirmedPPolynomial<CONVTIMES, CONVTIMES + 2>* BaseFunctionMaxDepth_Device, DeviceArrayView<OrientedPoint3D<float>> DenseOrientedPoints, DeviceArrayView<OctNode> NodeArray, const unsigned int DLevelOffset, const unsigned int DLevelNodeNum, DeviceBufferArray<Point3D<float>>& VectorField, cudaStream_t stream);
	};
}



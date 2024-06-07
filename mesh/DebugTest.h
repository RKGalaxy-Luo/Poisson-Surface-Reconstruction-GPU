/*****************************************************************//**
 * \file   DebugTest.h
 * \brief  �����㷨��
 * 
 * \author LUOJIAXUAN
 * \date   May 1st 2024
 *********************************************************************/
#pragma once
#include <stdio.h>
#include <string>
#include <base/DeviceReadWrite/DeviceBufferArray.h>
namespace SparseSurfelFusion {
	class DebugTest
	{
	public:
		DebugTest() = default;
		~DebugTest() = default;

		/**
		 * \brief ���64λ��32λ�����Ʊ��룬all64�����64λȫ����high32�����ǰ32λ��low32�������32λ.
		 * 
		 * \param keys ��Ҫ����������
		 * \param stream ��ǰ�����cuda������Ҫͬ��
		 * \param Type ��Ҫ����Ĳ��֣�all64�����64λȫ����high32�����ǰ32λ��low32�������32λ
		 */
		void CheckKeysEncode(DeviceArrayView<long long> keys, cudaStream_t stream, std::string Type = "all64");
	private:

	};
}



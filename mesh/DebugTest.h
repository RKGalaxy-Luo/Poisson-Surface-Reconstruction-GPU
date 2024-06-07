/*****************************************************************//**
 * \file   DebugTest.h
 * \brief  测试算法类
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
		 * \brief 输出64位高32位二进制编码，all64：输出64位全部，high32：输出前32位，low32：输出后32位.
		 * 
		 * \param keys 需要解析的数组
		 * \param stream 当前运算的cuda流，需要同步
		 * \param Type 需要输出的部分，all64：输出64位全部，high32：输出前32位，low32：输出后32位
		 */
		void CheckKeysEncode(DeviceArrayView<long long> keys, cudaStream_t stream, std::string Type = "all64");
	private:

	};
}



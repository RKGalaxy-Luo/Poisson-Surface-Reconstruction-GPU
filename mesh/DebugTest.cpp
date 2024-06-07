/*****************************************************************//**
 * \file   DebugTest.cpp
 * \brief  测试算法类
 * 
 * \author LUOJIAXUAN
 * \date   May 1st 2024
 *********************************************************************/
#include "DebugTest.h"

void SparseSurfelFusion::DebugTest::CheckKeysEncode(DeviceArrayView<long long> keys, cudaStream_t stream, std::string Type)
{
	CHECKCUDA(cudaStreamSynchronize(stream));		// 先同步
	std::vector<long long> keysCodeCheck;
	keys.Download(keysCodeCheck);
	for (int index = 0; index < keys.Size(); index++) {
		long long n = keysCodeCheck[index];
		if (Type == "high32") {
			char binaryStr[33] = { '0' };
			if (n == 0) {
				binaryStr[0] = '0';
				binaryStr[1] = '\0';
				return;
			}
			int High32 = int(n >> 32);	// 前32位
			int i = 0;
			while (High32 > 0) {
				binaryStr[i] = (High32 % 2) + '0'; // 将数字转换为字符 '0' 或 '1'
				High32 = High32 / 2;
				i++;
			}
			int len = 31;
			for (; i < len; i++) {
				binaryStr[i] = '0';
			}
			binaryStr[len + 1] = '\0'; // 在字符串末尾添加空字符
			// 反转字符串
			for (int j = 0; j < len / 2; j++) {
				char temp = binaryStr[j];
				binaryStr[j] = binaryStr[len - j - 1];
				binaryStr[len - j - 1] = temp;
			}
			printf("index = %hu 二进制编码为 %s\n", index, binaryStr);
		}
		else if (Type == "low32") {
			char binaryStr[33] = { '0' };;
			if (n == 0) {
				binaryStr[0] = '0';
				binaryStr[1] = '\0';
				return;
			}
			int Low32 = int(n & ((1ll << 31) - 1));	// 后32位
			int DenseIndex = int(n & ((1ll << 31) - 1));
			int i = 0;
			while (Low32 > 0) {
				binaryStr[i] = (Low32 % 2) + '0'; // 将数字转换为字符 '0' 或 '1'
				Low32 = Low32 / 2;
				i++;
			}
			// 反转字符串
			int len = 31;
			for (; i < len; i++) {
				binaryStr[i] = '0';
			}
			binaryStr[len + 1] = '\0'; // 在字符串末尾添加空字符
			for (int j = 0; j < len / 2; j++) {
				char temp = binaryStr[j];
				binaryStr[j] = binaryStr[len - j - 1];
				binaryStr[len - j - 1] = temp;
			}
			printf("index = %hu 二进制编码为 %s  在稠密数组中的index = %d\n", index, binaryStr, DenseIndex);
		}
		else {
			char binaryStr[65] = { '0' };
			if (n == 0) {
				binaryStr[0] = '0';
				binaryStr[1] = '\0';
				return;
			}

			int i = 0;
			while (n > 0) {
				binaryStr[i] = (n % 2) + '0'; // 将数字转换为字符 '0' 或 '1'
				n = n / 2;
				i++;
			}
			int len = 63;
			for (; i < len; i++) {
				binaryStr[i] = '0';
			}
			binaryStr[len + 1] = '\0'; // 在字符串末尾添加空字符
			// 反转字符串
			for (int j = 0; j < len / 2; j++) {
				char temp = binaryStr[j];
				binaryStr[j] = binaryStr[len - j - 1];
				binaryStr[len - j - 1] = temp;
			}
			printf("index = %hu 二进制编码为 %s\n", index, binaryStr);
		}
	}


}

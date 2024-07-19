/*****************************************************************//**
 * \file   VectorUtils.h
 * \brief  向量运算工具包：主要解决一些常见的Vector的运算问题
 * 
 * \author LUO
 * \date   January 12th 2024
 *********************************************************************/
#pragma once
#include <base/CommonTypes.h>
#include <vector_functions.h> //GPU带的vector的相关函数

namespace SparseSurfelFusion {
	/**
	 * \brief 计算vec的模
	 * \param vec 传入对应类型，计算平方和
	 * \return 
	 */
	__host__ __device__ __forceinline__
		float squared_norm(const float4& vec) {
		return vec.x * vec.x + vec.y * vec.y + vec.z * vec.z + vec.w * vec.w;
	}
	/**
	 * \brief 计算vec的模
	 * \param vec 传入对应类型，计算平方和
	 * \return
	 */
	__host__ __device__ __forceinline__
		float squared_norm(const float3& vec) {
		return vec.x * vec.x + vec.y * vec.y + vec.z * vec.z;
	}
	/**
	 * \brief 计算vec的模
	 * \param vec 传入对应类型，计算平方和
	 * \return
	 */
	__host__ __device__ __forceinline__
		float squared_norm(const float2& vec) {
		return vec.x * vec.x + vec.y * vec.y;
	}
	/**
	 * \brief 计算vec的模
	 * \param vec 传入对应类型，计算平方和，只计算x,y,z分类
	 * \return
	 */
	__host__ __device__ __forceinline__
		float squared_norm_xyz(const float4& vec) {
		return vec.x * vec.x + vec.y * vec.y + vec.z * vec.z;
	}

	/**
	 * \brief 计算两个向量的欧式距离，只考虑xyz分量
	 * \param v3 float3类型向量
	 * \param v4 float4类型向量
	 * \return 
	 */
	__host__ __device__ __forceinline__
		float squared_distance(const float3& v3, const float4& v4) {
		return (v3.x - v4.x) * (v3.x - v4.x) + (v3.y - v4.y) * (v3.y - v4.y) + (v3.z - v4.z) * (v3.z - v4.z);
	}

	/**
	 * \brief 计算两个向量的欧式距离，只考虑xyz分量
	 * \param v1 float3类型向量
	 * \param v2 float3类型向量
	 * \return
	 */
	__host__ __device__ __forceinline__
		float squared_distance(const float3& v1, const float3& v2) {
		return (v1.x - v2.x) * (v1.x - v2.x) + (v1.y - v2.y) * (v1.y - v2.y) + (v1.z - v2.z) * (v1.z - v2.z);
	}

	/**
	 * \brief 计算两个点的距离.
	 * 
	 * \param v1 float3类型向量
	 * \param v2 float3类型向量
	 * \return 两个点的距离
	 */
	__host__ __device__ __forceinline__
		float points_distance(const float3& v1, const float3& v2) {
		return sqrtf(squared_distance(v1, v2));
	}

	/**
	 * \brief 计算vec的模
	 * \param vec 传入的向量
	 * \return 
	 */
	__host__ __device__ __forceinline__ float norm(const float4& vec) {
		return sqrtf(squared_norm(vec));
	}
	/**
	 * \brief 计算vec的模
	 * \param vec 传入的向量
	 * \return
	 */
	__host__ __device__ __forceinline__ float norm(const float3& vec) {
		return sqrtf(squared_norm(vec));
	}


#if defined(__CUDA_ARCH__)//CUDA
	/**
	 * \brief 计算向量vec模的倒数
	 * \param vec 输入向量
	 * \return
	 */
	__device__ __forceinline__ float norm_inversed(const float4& vec) {
		return rsqrt(squared_norm(vec));
	}
	/**
	 * \brief 计算向量vec模的倒数
	 * \param vec 输入向量
	 * \return
	 */
	__device__ __forceinline__ float norm_inversed(const float3& vec) {
		return rsqrt(squared_norm(vec));
	}
#else
	/**
	 * \brief 计算向量vec模的倒数
	 * \param vec 输入向量
	 * \return
	 */
	__host__ __forceinline__ float norm_inversed(const float4& vec) {
		return 1.0f / sqrt(squared_norm(vec));
	}
	/**
	 * \brief 计算向量vec模的倒数
	 * \param vec 输入向量
	 * \return
	 */
	__host__ __forceinline__ float norm_inversed(const float3& vec) {
		return 1.0f / sqrt(squared_norm(vec));
	}
#endif



#if defined(__CUDA_ARCH__) //CUDA
	/**
	 * \brief 向量归一化
	 * \param vec 将输入的向量归一化
	 * \return
	 */
	__device__ __forceinline__ void normalize(float4& vec) {
		const float inv_vecnorm = rsqrtf(squared_norm(vec));
		vec.x *= inv_vecnorm;
		vec.y *= inv_vecnorm;
		vec.z *= inv_vecnorm;
		vec.w *= inv_vecnorm;
	}
#else
	/**
	 * \brief 向量归一化，传入的是实参
	 * \param vec 将输入的向量归一化
	 * \return void
	 */
	__host__ __forceinline__ void normalize(float4& vec) {
		const float inv_vecnorm = 1.0f / sqrtf(squared_norm(vec));
		vec.x *= inv_vecnorm;
		vec.y *= inv_vecnorm;
		vec.z *= inv_vecnorm;
		vec.w *= inv_vecnorm;
	}
#endif


	/**
	 * \brief 返回归一化向量，并保持vec不变
	 * \param vec 输入需要归一化的向量
	 * \return 
	 */
	__host__ __device__ __forceinline__ float4 normalized(const float4& vec) {
		const float inv_vecnorm = norm_inversed(vec);
		const float4 normalized_vec = make_float4(
			vec.x * inv_vecnorm, vec.y * inv_vecnorm,
			vec.z * inv_vecnorm, vec.w * inv_vecnorm
		);
		return normalized_vec;
	}
	/**
	 * \brief 返回归一化向量，并保持vec不变
	 * \param vec 输入需要归一化的向量
	 * \return
	 */
	__host__ __device__ __forceinline__ float3 normalized(const float3& vec) {
		const float inv_vecnorm = norm_inversed(vec);
		const float3 normalized_vec = make_float3(
			vec.x * inv_vecnorm,
			vec.y * inv_vecnorm,
			vec.z * inv_vecnorm
		);
		return normalized_vec;
	}


	/**
	 * \brief 检查向量是否为0
	 * \param v 输入向量
	 * \return 
	 */
	__host__ __device__ __forceinline__ bool is_zero_vertex(const float4& v) {
		return fabsf(v.x) < 1e-3 && fabsf(v.y) < 1e-3 && fabsf(v.z) < 1e-3;
	}
	/**
	 * \brief 检查向量是否为0
	 * \param v 输入向量
	 * \return
	 */
	__host__ __device__ __forceinline__ bool is_zero_vertex(const float3& v) {
		return fabsf(v.x) < 1e-3 && fabsf(v.y) < 1e-3 && fabsf(v.z) < 1e-3;
	}


	/**
	 * \brief 重载“+”：向量+常数
	 * \param vec 输入向量
	 * \param scalar 输入常数
	 * \return 返回vec + scalar的值
	 */
	__host__ __device__ __forceinline__ float3 operator+(const float3& vec, const float& scalar)
	{
		return make_float3(vec.x + scalar, vec.y + scalar, vec.z + scalar);
	}
	/**
	 * \brief 重载“+”：常数+向量
	 * \param scalar 输入常数
	 * \param vec 输入向量
	 * \return 返回vec + scalar的值
	 */
	__host__ __device__ __forceinline__ float3 operator+(const float& scalar, const float3& vec)
	{
		return make_float3(vec.x + scalar, vec.y + scalar, vec.z + scalar);
	}
	/**
	 * \brief 重载“+”：向量+向量
	 * \param vec_0 输入向量0
	 * \param vec_1 输入向量1
	 * \return 返回vec_0 + vec_1的值
	 */
	__host__ __device__ __forceinline__ float3 operator+(const float3& vec_0, const float3& vec_1)
	{
		return make_float3(vec_0.x + vec_1.x, vec_0.y + vec_1.y, vec_0.z + vec_1.z);
	}


	/**
	 * \brief 重载“-”：向量0 - 向量1
	 * \param vec_0 输入向量0
	 * \param vec_1 输入向量1
	 * \return 返回vec_0 - vec_1
	 */
	__host__ __device__ __forceinline__ float3 operator-(const float3& vec_0, const float3& vec_1) {
		return make_float3(vec_0.x - vec_1.x, vec_0.y - vec_1.y, vec_0.z - vec_1.z);
	}
	/**
	 * \brief 重载“-”：向量0 - 向量1
	 * \param vec_0 输入向量0
	 * \param vec_1 输入向量1
	 * \return 返回vec_0 - vec_1
	 */
	__host__ __device__ __forceinline__ float4 operator-(const float4& vec_0, const float4& vec_1) {
		return make_float4(vec_0.x - vec_1.x, vec_0.y - vec_1.y, vec_0.z - vec_1.z, vec_0.w - vec_1.w);
	}


	/**
	 * \brief 重载“*”：常数v * 向量v1
	 * \param v 输入常数
	 * \param v1 输入向量
	 * \return 返回 v * v1的值
	 */
	__host__ __device__ __forceinline__ float3 operator*(const float& v, const float3& v1)
	{
		return make_float3(v * v1.x, v * v1.y, v * v1.z);
	}
	/**
	 * \brief 重载“*”：向量v1 * 常数v
	 * \param v1 输入向量v1
	 * \param v 输入常数v
	 * \return 返回 v1 * v的值
	 */
	__host__ __device__ __forceinline__ float3 operator*(const float3& v1, const float& v)
	{
		return make_float3(v1.x * v, v1.y * v, v1.z * v);
	}
	/**
	 * \brief 重载“*”：常数v * 向量v1
	 * \param v 输入常数v
	 * \param v1 输入向量v1
	 * \return 返回 v * v1的值
	 */
	__host__ __device__ __forceinline__ float2 operator*(const float& v, const float2& v1)
	{
		return make_float2(v * v1.x, v * v1.y);
	}
	/**
	 * \brief 重载“*”：向量v1 * 常数v
	 * \param v1 输入向量v1
	 * \param v 输入常数v
	 * \return 返回 v1 * v的值
	 */
	__host__ __device__ __forceinline__ float2 operator*(const float2& v1, const float& v)
	{
		return make_float2(v1.x * v, v1.y * v);
	}

	/**
	 * \brief 重载“*=”：计算3维向量vec *= 常数v，注意此时输入向量的值也变化了，传入的是实参
	 * \param vec 输入向量
	 * \param v 输入常数
	 * \return 返回vec的值：vec = vec * v
	 */
	__host__ __device__ __forceinline__ float3& operator*=(float3& vec, const float& v)
	{
		vec.x *= v;
		vec.y *= v;
		vec.z *= v;
		return vec;
	}

	/**
	 * \brief 重载“*=”：计算4维向量vec *= 常数v，注意此时输入向量vec的值也变化了，传入的是实参
	 * \param vec 输入的4维向量vec
	 * \param v 输入常数v
	 * \return 返回vec的值：vec = vec * v
	 */
	__host__ __device__ __forceinline__ float4& operator*=(float4& vec, const float& v)
	{
		vec.x *= v;
		vec.y *= v;
		vec.z *= v;
		vec.w *= v;
		return vec;
	}
	/**
	 * \brief 重载“+=”：向量vec_0 + 向量vec_1，注意此时vec_0值变了，传入的是实参
	 * \param vec_0 输入向量vec_0
	 * \param vec_1 输入向量vec_1
	 * \return 返回vec_0的值：vec_0 = vec_0 + vec_1
	 */
	__host__ __device__ __forceinline__ float3& operator+=(float3& vec_0, const float3& vec_1)
	{
		vec_0.x += vec_1.x;
		vec_0.y += vec_1.y;
		vec_0.z += vec_1.z;
		return vec_0;
	}

	/**
	 * \brief 向量取反
	 * \param vec 输入向量
	 * \return 输出反向向量
	 */
	__host__ __device__ __forceinline__ float3 operator-(const float3& vec)
	{
		float3 negative_vec;
		negative_vec.x = -vec.x;
		negative_vec.y = -vec.y;
		negative_vec.z = -vec.z;
		return negative_vec;
	}


	/**
	 * \brief 3维向量点乘
	 * \param v1 输入向量
	 * \param v2 输入向量
	 * \return 输出点乘结果
	 */
	__host__ __device__ __forceinline__ float dot(const float3& v1, const float3& v2) {
		return v1.x * v2.x + v1.y * v2.y + v1.z * v2.z;
	}
	/**
	 * \brief 4维向量点乘
	 * \param v1 输入向量
	 * \param v2 输入向量
	 * \return 输出点乘结果
	 */
	__host__ __device__ __forceinline__ float dot(const float4& v1, const float4& v2) {
		return v1.x * v2.x + v1.y * v2.y + v1.z * v2.z + v1.w * v2.w;
	}


	/**
	 * \brief 4维向量只点乘x,y,z三个分量
	 * \param v1 输入向量v1
	 * \param v2 输入向量v2
	 * \return 输入x,y,z分量的点乘
	 */
	__host__ __device__ __forceinline__ float dotxyz(const float4& v1, const float4& v2)
	{
		return v1.x * v2.x + v1.y * v2.y + v1.z * v2.z;
	}
	/**
	 * \brief 3维向量的点乘
	 * \param v1 输入向量v1
	 * \param v2 输入向量v2
	 * \return 输出v1·v2
	 */
	__host__ __device__ __forceinline__ float dotxyz(const float3& v1, const float4& v2)
	{
		return v1.x * v2.x + v1.y * v2.y + v1.z * v2.z;
	}

	/**
	 * \brief 3维向量叉乘
	 * \param v1 输入向量v1
	 * \param v2 输入向量v2
	 * \return 输出v1×v2
	 */
	__host__ __device__ __forceinline__ float3 cross(const float3& v1, const float3& v2)
	{
		return make_float3(v1.y * v2.z - v1.z * v2.y, v1.z * v2.x - v1.x * v2.z, v1.x * v2.y - v1.y * v2.x);
	}
	/**
	 * \brief 3维向量叉乘4维向量的x,y,z分量
	 * \param v1 输入3维向量
	 * \param v2 输入4维向量
	 * \return 输出3维向量叉乘4维向量的x,y,z分量的结果
	 */
	__host__ __device__ __forceinline__ float3 cross_xyz(const float3& v1, const float4& v2)
	{
		return make_float3(v1.y * v2.z - v1.z * v2.y, v1.z * v2.x - v1.x * v2.z, v1.x * v2.y - v1.y * v2.x);
	}

	/**
	 * \brief 3维向量的1-范数：|x| + |y| + |z|
	 * \param vec输入向量
	 * \return 返回向量vec的1-范数
	 */
	__host__ __device__ __forceinline__ float fabsf_sum(const float3& vec) {
		return fabsf(vec.x) + fabsf(vec.y) + fabsf(vec.z);
	}
	/**
	 * \brief 4维向量的1-范数：|x| + |y| + |z| + |w|
	 * \param vec输入向量
	 * \return 输出向量的1-范数
	 */
	__host__ __device__ __forceinline__ float fabsf_sum(const float4& vec) {
		return fabsf(vec.x) + fabsf(vec.y) + fabsf(vec.z) + fabsf(vec.w);
	}

	/**
	 * \brief 3维向量与4维向量(x,y,z分量)的平均绝对误差(MAS) L1 Error
	 * \param vec_0 输入的3维向量
	 * \param vec_1 输入的4维向量
	 * \return 返回向量的L1 Error
	 */
	__host__ __device__ __forceinline__ float fabsf_diff_xyz(const float3& vec_0, const float4& vec_1) {
		return fabsf(vec_0.x - vec_1.x) + fabsf(vec_0.y - vec_1.y) + fabsf(vec_0.z - vec_1.z);
	}
}

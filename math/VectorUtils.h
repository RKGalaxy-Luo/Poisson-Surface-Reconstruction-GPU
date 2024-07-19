/*****************************************************************//**
 * \file   VectorUtils.h
 * \brief  �������㹤�߰�����Ҫ���һЩ������Vector����������
 * 
 * \author LUO
 * \date   January 12th 2024
 *********************************************************************/
#pragma once
#include <base/CommonTypes.h>
#include <vector_functions.h> //GPU����vector����غ���

namespace SparseSurfelFusion {
	/**
	 * \brief ����vec��ģ
	 * \param vec �����Ӧ���ͣ�����ƽ����
	 * \return 
	 */
	__host__ __device__ __forceinline__
		float squared_norm(const float4& vec) {
		return vec.x * vec.x + vec.y * vec.y + vec.z * vec.z + vec.w * vec.w;
	}
	/**
	 * \brief ����vec��ģ
	 * \param vec �����Ӧ���ͣ�����ƽ����
	 * \return
	 */
	__host__ __device__ __forceinline__
		float squared_norm(const float3& vec) {
		return vec.x * vec.x + vec.y * vec.y + vec.z * vec.z;
	}
	/**
	 * \brief ����vec��ģ
	 * \param vec �����Ӧ���ͣ�����ƽ����
	 * \return
	 */
	__host__ __device__ __forceinline__
		float squared_norm(const float2& vec) {
		return vec.x * vec.x + vec.y * vec.y;
	}
	/**
	 * \brief ����vec��ģ
	 * \param vec �����Ӧ���ͣ�����ƽ���ͣ�ֻ����x,y,z����
	 * \return
	 */
	__host__ __device__ __forceinline__
		float squared_norm_xyz(const float4& vec) {
		return vec.x * vec.x + vec.y * vec.y + vec.z * vec.z;
	}

	/**
	 * \brief ��������������ŷʽ���룬ֻ����xyz����
	 * \param v3 float3��������
	 * \param v4 float4��������
	 * \return 
	 */
	__host__ __device__ __forceinline__
		float squared_distance(const float3& v3, const float4& v4) {
		return (v3.x - v4.x) * (v3.x - v4.x) + (v3.y - v4.y) * (v3.y - v4.y) + (v3.z - v4.z) * (v3.z - v4.z);
	}

	/**
	 * \brief ��������������ŷʽ���룬ֻ����xyz����
	 * \param v1 float3��������
	 * \param v2 float3��������
	 * \return
	 */
	__host__ __device__ __forceinline__
		float squared_distance(const float3& v1, const float3& v2) {
		return (v1.x - v2.x) * (v1.x - v2.x) + (v1.y - v2.y) * (v1.y - v2.y) + (v1.z - v2.z) * (v1.z - v2.z);
	}

	/**
	 * \brief ����������ľ���.
	 * 
	 * \param v1 float3��������
	 * \param v2 float3��������
	 * \return ������ľ���
	 */
	__host__ __device__ __forceinline__
		float points_distance(const float3& v1, const float3& v2) {
		return sqrtf(squared_distance(v1, v2));
	}

	/**
	 * \brief ����vec��ģ
	 * \param vec ���������
	 * \return 
	 */
	__host__ __device__ __forceinline__ float norm(const float4& vec) {
		return sqrtf(squared_norm(vec));
	}
	/**
	 * \brief ����vec��ģ
	 * \param vec ���������
	 * \return
	 */
	__host__ __device__ __forceinline__ float norm(const float3& vec) {
		return sqrtf(squared_norm(vec));
	}


#if defined(__CUDA_ARCH__)//CUDA
	/**
	 * \brief ��������vecģ�ĵ���
	 * \param vec ��������
	 * \return
	 */
	__device__ __forceinline__ float norm_inversed(const float4& vec) {
		return rsqrt(squared_norm(vec));
	}
	/**
	 * \brief ��������vecģ�ĵ���
	 * \param vec ��������
	 * \return
	 */
	__device__ __forceinline__ float norm_inversed(const float3& vec) {
		return rsqrt(squared_norm(vec));
	}
#else
	/**
	 * \brief ��������vecģ�ĵ���
	 * \param vec ��������
	 * \return
	 */
	__host__ __forceinline__ float norm_inversed(const float4& vec) {
		return 1.0f / sqrt(squared_norm(vec));
	}
	/**
	 * \brief ��������vecģ�ĵ���
	 * \param vec ��������
	 * \return
	 */
	__host__ __forceinline__ float norm_inversed(const float3& vec) {
		return 1.0f / sqrt(squared_norm(vec));
	}
#endif



#if defined(__CUDA_ARCH__) //CUDA
	/**
	 * \brief ������һ��
	 * \param vec �������������һ��
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
	 * \brief ������һ�����������ʵ��
	 * \param vec �������������һ��
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
	 * \brief ���ع�һ��������������vec����
	 * \param vec ������Ҫ��һ��������
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
	 * \brief ���ع�һ��������������vec����
	 * \param vec ������Ҫ��һ��������
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
	 * \brief ��������Ƿ�Ϊ0
	 * \param v ��������
	 * \return 
	 */
	__host__ __device__ __forceinline__ bool is_zero_vertex(const float4& v) {
		return fabsf(v.x) < 1e-3 && fabsf(v.y) < 1e-3 && fabsf(v.z) < 1e-3;
	}
	/**
	 * \brief ��������Ƿ�Ϊ0
	 * \param v ��������
	 * \return
	 */
	__host__ __device__ __forceinline__ bool is_zero_vertex(const float3& v) {
		return fabsf(v.x) < 1e-3 && fabsf(v.y) < 1e-3 && fabsf(v.z) < 1e-3;
	}


	/**
	 * \brief ���ء�+��������+����
	 * \param vec ��������
	 * \param scalar ���볣��
	 * \return ����vec + scalar��ֵ
	 */
	__host__ __device__ __forceinline__ float3 operator+(const float3& vec, const float& scalar)
	{
		return make_float3(vec.x + scalar, vec.y + scalar, vec.z + scalar);
	}
	/**
	 * \brief ���ء�+��������+����
	 * \param scalar ���볣��
	 * \param vec ��������
	 * \return ����vec + scalar��ֵ
	 */
	__host__ __device__ __forceinline__ float3 operator+(const float& scalar, const float3& vec)
	{
		return make_float3(vec.x + scalar, vec.y + scalar, vec.z + scalar);
	}
	/**
	 * \brief ���ء�+��������+����
	 * \param vec_0 ��������0
	 * \param vec_1 ��������1
	 * \return ����vec_0 + vec_1��ֵ
	 */
	__host__ __device__ __forceinline__ float3 operator+(const float3& vec_0, const float3& vec_1)
	{
		return make_float3(vec_0.x + vec_1.x, vec_0.y + vec_1.y, vec_0.z + vec_1.z);
	}


	/**
	 * \brief ���ء�-��������0 - ����1
	 * \param vec_0 ��������0
	 * \param vec_1 ��������1
	 * \return ����vec_0 - vec_1
	 */
	__host__ __device__ __forceinline__ float3 operator-(const float3& vec_0, const float3& vec_1) {
		return make_float3(vec_0.x - vec_1.x, vec_0.y - vec_1.y, vec_0.z - vec_1.z);
	}
	/**
	 * \brief ���ء�-��������0 - ����1
	 * \param vec_0 ��������0
	 * \param vec_1 ��������1
	 * \return ����vec_0 - vec_1
	 */
	__host__ __device__ __forceinline__ float4 operator-(const float4& vec_0, const float4& vec_1) {
		return make_float4(vec_0.x - vec_1.x, vec_0.y - vec_1.y, vec_0.z - vec_1.z, vec_0.w - vec_1.w);
	}


	/**
	 * \brief ���ء�*��������v * ����v1
	 * \param v ���볣��
	 * \param v1 ��������
	 * \return ���� v * v1��ֵ
	 */
	__host__ __device__ __forceinline__ float3 operator*(const float& v, const float3& v1)
	{
		return make_float3(v * v1.x, v * v1.y, v * v1.z);
	}
	/**
	 * \brief ���ء�*��������v1 * ����v
	 * \param v1 ��������v1
	 * \param v ���볣��v
	 * \return ���� v1 * v��ֵ
	 */
	__host__ __device__ __forceinline__ float3 operator*(const float3& v1, const float& v)
	{
		return make_float3(v1.x * v, v1.y * v, v1.z * v);
	}
	/**
	 * \brief ���ء�*��������v * ����v1
	 * \param v ���볣��v
	 * \param v1 ��������v1
	 * \return ���� v * v1��ֵ
	 */
	__host__ __device__ __forceinline__ float2 operator*(const float& v, const float2& v1)
	{
		return make_float2(v * v1.x, v * v1.y);
	}
	/**
	 * \brief ���ء�*��������v1 * ����v
	 * \param v1 ��������v1
	 * \param v ���볣��v
	 * \return ���� v1 * v��ֵ
	 */
	__host__ __device__ __forceinline__ float2 operator*(const float2& v1, const float& v)
	{
		return make_float2(v1.x * v, v1.y * v);
	}

	/**
	 * \brief ���ء�*=��������3ά����vec *= ����v��ע���ʱ����������ֵҲ�仯�ˣ��������ʵ��
	 * \param vec ��������
	 * \param v ���볣��
	 * \return ����vec��ֵ��vec = vec * v
	 */
	__host__ __device__ __forceinline__ float3& operator*=(float3& vec, const float& v)
	{
		vec.x *= v;
		vec.y *= v;
		vec.z *= v;
		return vec;
	}

	/**
	 * \brief ���ء�*=��������4ά����vec *= ����v��ע���ʱ��������vec��ֵҲ�仯�ˣ��������ʵ��
	 * \param vec �����4ά����vec
	 * \param v ���볣��v
	 * \return ����vec��ֵ��vec = vec * v
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
	 * \brief ���ء�+=��������vec_0 + ����vec_1��ע���ʱvec_0ֵ���ˣ��������ʵ��
	 * \param vec_0 ��������vec_0
	 * \param vec_1 ��������vec_1
	 * \return ����vec_0��ֵ��vec_0 = vec_0 + vec_1
	 */
	__host__ __device__ __forceinline__ float3& operator+=(float3& vec_0, const float3& vec_1)
	{
		vec_0.x += vec_1.x;
		vec_0.y += vec_1.y;
		vec_0.z += vec_1.z;
		return vec_0;
	}

	/**
	 * \brief ����ȡ��
	 * \param vec ��������
	 * \return �����������
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
	 * \brief 3ά�������
	 * \param v1 ��������
	 * \param v2 ��������
	 * \return �����˽��
	 */
	__host__ __device__ __forceinline__ float dot(const float3& v1, const float3& v2) {
		return v1.x * v2.x + v1.y * v2.y + v1.z * v2.z;
	}
	/**
	 * \brief 4ά�������
	 * \param v1 ��������
	 * \param v2 ��������
	 * \return �����˽��
	 */
	__host__ __device__ __forceinline__ float dot(const float4& v1, const float4& v2) {
		return v1.x * v2.x + v1.y * v2.y + v1.z * v2.z + v1.w * v2.w;
	}


	/**
	 * \brief 4ά����ֻ���x,y,z��������
	 * \param v1 ��������v1
	 * \param v2 ��������v2
	 * \return ����x,y,z�����ĵ��
	 */
	__host__ __device__ __forceinline__ float dotxyz(const float4& v1, const float4& v2)
	{
		return v1.x * v2.x + v1.y * v2.y + v1.z * v2.z;
	}
	/**
	 * \brief 3ά�����ĵ��
	 * \param v1 ��������v1
	 * \param v2 ��������v2
	 * \return ���v1��v2
	 */
	__host__ __device__ __forceinline__ float dotxyz(const float3& v1, const float4& v2)
	{
		return v1.x * v2.x + v1.y * v2.y + v1.z * v2.z;
	}

	/**
	 * \brief 3ά�������
	 * \param v1 ��������v1
	 * \param v2 ��������v2
	 * \return ���v1��v2
	 */
	__host__ __device__ __forceinline__ float3 cross(const float3& v1, const float3& v2)
	{
		return make_float3(v1.y * v2.z - v1.z * v2.y, v1.z * v2.x - v1.x * v2.z, v1.x * v2.y - v1.y * v2.x);
	}
	/**
	 * \brief 3ά�������4ά������x,y,z����
	 * \param v1 ����3ά����
	 * \param v2 ����4ά����
	 * \return ���3ά�������4ά������x,y,z�����Ľ��
	 */
	__host__ __device__ __forceinline__ float3 cross_xyz(const float3& v1, const float4& v2)
	{
		return make_float3(v1.y * v2.z - v1.z * v2.y, v1.z * v2.x - v1.x * v2.z, v1.x * v2.y - v1.y * v2.x);
	}

	/**
	 * \brief 3ά������1-������|x| + |y| + |z|
	 * \param vec��������
	 * \return ��������vec��1-����
	 */
	__host__ __device__ __forceinline__ float fabsf_sum(const float3& vec) {
		return fabsf(vec.x) + fabsf(vec.y) + fabsf(vec.z);
	}
	/**
	 * \brief 4ά������1-������|x| + |y| + |z| + |w|
	 * \param vec��������
	 * \return ���������1-����
	 */
	__host__ __device__ __forceinline__ float fabsf_sum(const float4& vec) {
		return fabsf(vec.x) + fabsf(vec.y) + fabsf(vec.z) + fabsf(vec.w);
	}

	/**
	 * \brief 3ά������4ά����(x,y,z����)��ƽ���������(MAS) L1 Error
	 * \param vec_0 �����3ά����
	 * \param vec_1 �����4ά����
	 * \return ����������L1 Error
	 */
	__host__ __device__ __forceinline__ float fabsf_diff_xyz(const float3& vec_0, const float4& vec_1) {
		return fabsf(vec_0.x - vec_1.x) + fabsf(vec_0.y - vec_1.y) + fabsf(vec_0.z - vec_1.z);
	}
}

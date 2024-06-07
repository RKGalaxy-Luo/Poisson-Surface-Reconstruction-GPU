/*****************************************************************//**
 * \file   Vector.cu
 * \brief  向量计算方法实现
 * 
 * \author LUOJIAXUAN
 * \date   May 22nd 2024
 *********************************************************************/
#pragma once
#include "Vector.h"
////////////
// Vector //
////////////
template<class T>
SparseSurfelFusion::Vector<T>::Vector()
{
    m_N = 0;
    m_pV = 0;
}
template<class T>
SparseSurfelFusion::Vector<T>::Vector(const SparseSurfelFusion::Vector<T>& V)
{
    m_N = 0;
    m_pV = 0;
    Resize(V.m_N);
    memcpy(m_pV, V.m_pV, m_N * sizeof(T));
}
template<class T>
SparseSurfelFusion::Vector<T>::Vector(int N)
{
    m_N = 0;
    m_pV = 0;
    Resize(N);
}
template<class T>
void SparseSurfelFusion::Vector<T>::Resize(int N)
{
    if (m_N != N) {
        if (m_N) { delete[] m_pV; }
        m_pV = NULL;
        m_N = N;
        if (N)
        {
            m_pV = new T[N];
        }
    }
    memset(m_pV, 0, N * sizeof(T));
}
template<class T>
SparseSurfelFusion::Vector<T>::Vector(int N, T* pV)
{
    Resize(N);
    memcpy(m_pV, pV, N * sizeof(T));
}
template<class T>
SparseSurfelFusion::Vector<T>::~Vector() { Resize(0); }
template<class T>
SparseSurfelFusion::Vector<T>& SparseSurfelFusion::Vector<T>::operator = (const Vector& V)
{
    Resize(V.m_N);
    memcpy(m_pV, V.m_pV, m_N * sizeof(T));
    return *this;
}
template<class T>
__host__ __device__  int SparseSurfelFusion::Vector<T>::Dimensions() const { return m_N; }
template<class T>
__host__ __device__ void SparseSurfelFusion::Vector<T>::SetZero(void) { for (int i = 0; i < m_N; i++) { m_pV[i] = T(0); } }
template<class T>
__host__ __device__ const T& SparseSurfelFusion::Vector<T>::operator () (int i) const
{
    Assert(i < m_N);
    return m_pV[i];
}
template<class T>
__host__ __device__ T& SparseSurfelFusion::Vector<T>::operator () (int i)
{
    return m_pV[i];
}
template<class T>
__host__ __device__ const T& SparseSurfelFusion::Vector<T>::operator [] (int i) const
{
    return m_pV[i];
}
template<class T>
__host__ __device__ T& SparseSurfelFusion::Vector<T>::operator [] (int i)
{
    return m_pV[i];
}
template<class T>
SparseSurfelFusion::Vector<T> SparseSurfelFusion::Vector<T>::operator * (const T& A) const
{
    Vector V(*this);
    for (int i = 0; i < m_N; i++)
        V.m_pV[i] *= A;
    return V;
}
template<class T>
__host__ __device__ SparseSurfelFusion::Vector<T>& SparseSurfelFusion::Vector<T>::operator *= (const T& A)
{
    for (int i = 0; i < m_N; i++)
        m_pV[i] *= A;
    return *this;
}
template<class T>
SparseSurfelFusion::Vector<T> SparseSurfelFusion::Vector<T>::operator / (const T& A) const
{
    Vector V(*this);
    for (int i = 0; i < m_N; i++)
        V.m_pV[i] /= A;
    return V;
}
template<class T>
__host__ __device__ SparseSurfelFusion::Vector<T>& SparseSurfelFusion::Vector<T>::operator /= (const T& A)
{
    for (int i = 0; i < m_N; i++)
        m_pV[i] /= A;
    return *this;
}
template<class T>
SparseSurfelFusion::Vector<T> SparseSurfelFusion::Vector<T>::operator + (const Vector<T>& V0) const
{
    Vector<T> V(m_N);
    for (int i = 0; i < m_N; i++)
        V.m_pV[i] = m_pV[i] + V0.m_pV[i];

    return V;
}
template<class T>
__host__ __device__ SparseSurfelFusion::Vector<T>& SparseSurfelFusion::Vector<T>::AddScaled(const Vector<T>& V, const T& scale)
{
    for (int i = 0; i < m_N; i++)
        m_pV[i] += V.m_pV[i] * scale;

    return *this;
}
template<class T>
__host__ __device__ SparseSurfelFusion::Vector<T>& SparseSurfelFusion::Vector<T>::SubtractScaled(const Vector<T>& V, const T& scale)
{
    for (int i = 0; i < m_N; i++)
        m_pV[i] -= V.m_pV[i] * scale;

    return *this;
}
template<class T>
__host__ __device__ void SparseSurfelFusion::Vector<T>::Add(const Vector<T>& V1, const T& scale1, const Vector<T>& V2, const T& scale2, Vector<T>& Out) {
    for (int i = 0; i < V1.m_N; i++)
        Out.m_pV[i] = V1.m_pV[i] * scale1 + V2.m_pV[i] * scale2;
}
template<class T>
__host__ __device__ void SparseSurfelFusion::Vector<T>::Add(const Vector<T>& V1, const T& scale1, const Vector<T>& V2, Vector<T>& Out) {
    for (int i = 0; i < V1.m_N; i++)
        Out.m_pV[i] = V1.m_pV[i] * scale1 + V2.m_pV[i];
}
template<class T>
__host__ __device__ SparseSurfelFusion::Vector<T>& SparseSurfelFusion::Vector<T>::operator += (const Vector<T>& V)
{
    for (int i = 0; i < m_N; i++)
        m_pV[i] += V.m_pV[i];

    return *this;
}
template<class T>
SparseSurfelFusion::Vector<T> SparseSurfelFusion::Vector<T>::operator - (const Vector<T>& V0) const
{
    Vector<T> V(m_N);
    for (int i = 0; i < m_N; i++)
        V.m_pV[i] = m_pV[i] - V0.m_pV[i];

    return V;
}
template<class T>
SparseSurfelFusion::Vector<T> SparseSurfelFusion::Vector<T>::operator - (void) const
{
    Vector<T> V(m_N);

    for (int i = 0; i < m_N; i++)
        V.m_pV[i] = -m_pV[i];

    return V;
}
template<class T>
__host__ __device__ SparseSurfelFusion::Vector<T>& SparseSurfelFusion::Vector<T>::operator -= (const Vector<T>& V)
{
    for (int i = 0; i < m_N; i++)
        m_pV[i] -= V.m_pV[i];

    return *this;
}
template<class T>
__host__ __device__ T SparseSurfelFusion::Vector<T>::Norm(int Ln) const
{
    T N = T();
    for (int i = 0; i < m_N; i++)
        N += pow(m_pV[i], (T)Ln);
    return pow(N, (T)1.0 / Ln);
}
template<class T>
__host__ __device__ void SparseSurfelFusion::Vector<T>::Normalize()
{
    T N = 1.0f / Norm(2);
    for (int i = 0; i < m_N; i++)
        m_pV[i] *= N;
}
template<class T>
__host__ __device__ T SparseSurfelFusion::Vector<T>::Length() const
{
    T N = T();
    for (int i = 0; i < m_N; i++)
        N += m_pV[i] * m_pV[i];
    return sqrt(N);
}
template<class T>
__host__ __device__ T SparseSurfelFusion::Vector<T>::Dot(const Vector<T>& V) const
{
    T V0 = T();
    for (int i = 0; i < m_N; i++)
        V0 += m_pV[i] * V.m_pV[i];

    return V0;
}



/////////////
// NVector //
/////////////
template<class T, int Dim>
SparseSurfelFusion::NVector<T, Dim>::NVector()
{
    m_N = 0;
    m_pV = 0;
}
template<class T, int Dim>
SparseSurfelFusion::NVector<T, Dim>::NVector(const NVector<T, Dim>& V)
{
    m_N = 0;
    m_pV = 0;
    Resize(V.m_N);
    memcpy(m_pV, V.m_pV, m_N * sizeof(T) * Dim);
}
template<class T, int Dim>
SparseSurfelFusion::NVector<T, Dim>::NVector(int N)
{
    m_N = 0;
    m_pV = 0;
    Resize(N);
}
template<class T, int Dim>
void SparseSurfelFusion::NVector<T, Dim>::Resize(int N)
{
    if (m_N != N) {
        if (m_N) { delete[] m_pV; }
        m_pV = NULL;
        m_N = N;
        if (N) { m_pV = new T[Dim * N]; }
    }
    memset(m_pV, 0, N * sizeof(T) * Dim);
}
template<class T, int Dim>
SparseSurfelFusion::NVector<T, Dim>::NVector(int N, T* pV)
{
    Resize(N);
    memcpy(m_pV, pV, N * sizeof(T) * Dim);
}
template<class T, int Dim>
SparseSurfelFusion::NVector<T, Dim>::~NVector() { Resize(0); }
template<class T, int Dim>
SparseSurfelFusion::NVector<T, Dim>& SparseSurfelFusion::NVector<T, Dim>::operator = (const NVector& V)
{
    Resize(V.m_N);
    memcpy(m_pV, V.m_pV, m_N * sizeof(T) * Dim);
    return *this;
}
template<class T, int Dim>
int SparseSurfelFusion::NVector<T, Dim>::Dimensions() const { return m_N; }
template<class T, int Dim>
void SparseSurfelFusion::NVector<T, Dim>::SetZero(void) { for (int i = 0; i < m_N * Dim; i++) { m_pV[i] = T(0); } }
template<class T, int Dim>
const T* SparseSurfelFusion::NVector<T, Dim>::operator () (int i) const
{
    Assert(i < m_N);
    return &m_pV[i * Dim];
}
template<class T, int Dim>
T* SparseSurfelFusion::NVector<T, Dim>::operator () (int i)
{
    return &m_pV[i * Dim];
}
template<class T, int Dim>
const T* SparseSurfelFusion::NVector<T, Dim>::operator [] (int i) const
{
    return &m_pV[i * Dim];
}
template<class T, int Dim>
T* SparseSurfelFusion::NVector<T, Dim>::operator [] (int i)
{
    return &m_pV[i * Dim];
}
template<class T, int Dim>
SparseSurfelFusion::NVector<T, Dim> SparseSurfelFusion::NVector<T, Dim>::operator * (const T& A) const
{
    NVector<T, Dim> V(*this);
    for (int i = 0; i < m_N * Dim; i++)
        V.m_pV[i] *= A;
    return V;
}
template<class T, int Dim>
SparseSurfelFusion::NVector<T, Dim>& SparseSurfelFusion::NVector<T, Dim>::operator *= (const T& A)
{
    for (int i = 0; i < m_N * Dim; i++)
        m_pV[i] *= A;
    return *this;
}
template<class T, int Dim>
SparseSurfelFusion::NVector<T, Dim> SparseSurfelFusion::NVector<T, Dim>::operator / (const T& A) const
{
    NVector<T, Dim> V(*this);
    for (int i = 0; i < m_N * Dim; i++)
        V.m_pV[i] /= A;
    return V;
}
template<class T, int Dim>
SparseSurfelFusion::NVector<T, Dim>& SparseSurfelFusion::NVector<T, Dim>::operator /= (const T& A)
{
    for (int i = 0; i < m_N * Dim; i++)
        m_pV[i] /= A;
    return *this;
}
template<class T, int Dim>
SparseSurfelFusion::NVector<T, Dim> SparseSurfelFusion::NVector<T, Dim>::operator + (const NVector<T, Dim>& V0) const
{
    NVector<T, Dim> V(m_N);
    for (int i = 0; i < m_N * Dim; i++)
        V.m_pV[i] = m_pV[i] + V0.m_pV[i];

    return V;
}
template<class T, int Dim>
SparseSurfelFusion::NVector<T, Dim>& SparseSurfelFusion::NVector<T, Dim>::AddScaled(const NVector<T, Dim>& V, const T& scale)
{
    for (int i = 0; i < m_N * Dim; i++)
        m_pV[i] += V.m_pV[i] * scale;

    return *this;
}
template<class T, int Dim>
SparseSurfelFusion::NVector<T, Dim>& SparseSurfelFusion::NVector<T, Dim>::SubtractScaled(const NVector<T, Dim>& V, const T& scale)
{
    for (int i = 0; i < m_N * Dim; i++)
        m_pV[i] -= V.m_pV[i] * scale;

    return *this;
}
template<class T, int Dim>
void SparseSurfelFusion::NVector<T, Dim>::Add(const NVector<T, Dim>& V1, const T& scale1, const NVector<T, Dim>& V2, const T& scale2, NVector<T, Dim>& Out) {
    for (int i = 0; i < V1.m_N * Dim; i++)
        Out.m_pV[i] = V1.m_pV[i] * scale1 + V2.m_pV[i] * scale2;
}
template<class T, int Dim>
void SparseSurfelFusion::NVector<T, Dim>::Add(const NVector<T, Dim>& V1, const T& scale1, const NVector<T, Dim>& V2, NVector<T, Dim>& Out) {
    for (int i = 0; i < V1.m_N * Dim; i++)
        Out.m_pV[i] = V1.m_pV[i] * scale1 + V2.m_pV[i];
}
template<class T, int Dim>
SparseSurfelFusion::NVector<T, Dim>& SparseSurfelFusion::NVector<T, Dim>::operator += (const NVector<T, Dim>& V)
{
    for (int i = 0; i < m_N * Dim; i++)
        m_pV[i] += V.m_pV[i];

    return *this;
}
template<class T, int Dim>
SparseSurfelFusion::NVector<T, Dim> SparseSurfelFusion::NVector<T, Dim>::operator - (const NVector<T, Dim>& V0) const
{
    NVector<T, Dim> V(m_N);
    for (int i = 0; i < m_N * Dim; i++)
        V.m_pV[i] = m_pV[i] - V0.m_pV[i];

    return V;
}
template<class T, int Dim>
SparseSurfelFusion::NVector<T, Dim> SparseSurfelFusion::NVector<T, Dim>::operator - (void) const
{
    NVector<T, Dim> V(m_N);

    for (int i = 0; i < m_N * Dim; i++)
        V.m_pV[i] = -m_pV[i];

    return V;
}
template<class T, int Dim>
SparseSurfelFusion::NVector<T, Dim>& SparseSurfelFusion::NVector<T, Dim>::operator -= (const NVector<T, Dim>& V)
{
    for (int i = 0; i < m_N * Dim; i++)
        m_pV[i] -= V.m_pV[i];

    return *this;
}
template<class T, int Dim>
T SparseSurfelFusion::NVector<T, Dim>::Norm(int Ln) const
{
    T N = T();
    for (int i = 0; i < m_N * Dim; i++)
        N += pow(m_pV[i], (T)Ln);
    return pow(N, (T)1.0 / Ln);
}
template<class T, int Dim>
void SparseSurfelFusion::NVector<T, Dim>::Normalize()
{
    T N = 1.0f / Norm(2);
    for (int i = 0; i < m_N * 3; i++)
        m_pV[i] *= N;
}
template<class T, int Dim>
T SparseSurfelFusion::NVector<T, Dim>::Length() const
{
    T N = T();
    for (int i = 0; i < m_N * Dim; i++)
        N += m_pV[i] * m_pV[i];
    return sqrt(N);
}
template<class T, int Dim>
T SparseSurfelFusion::NVector<T, Dim>::Dot(const NVector<T, Dim>& V) const
{
    T V0 = T();
    for (int i = 0; i < m_N * Dim; i++)
        V0 += m_pV[i] * V.m_pV[i];

    return V0;
}

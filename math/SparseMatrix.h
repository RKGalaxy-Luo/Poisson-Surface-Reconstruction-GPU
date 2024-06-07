/*****************************************************************//**
 * \file   SparseMatrix.h
 * \brief  稀疏矩阵计算方法实现
 * 
 * \author LUOJIAXUAN
 * \date   May 22nd 2024
 *********************************************************************/
#pragma once
#include "Vector.h"

namespace SparseSurfelFusion {
    template <class T>
    struct MatrixEntry;		// 前向声明

    template <class T, int Dim>
    struct NMatrixEntry;	// 前向声明


    /**
     *  数据保存在Allocator<MatrixEntry<T> >.memory中
     *  m_ppElements[i]是指向第i行起始地址的指针
     */
    template<class T> class SparseMatrix
    {
    public:
        int rows = 0;
        int* rowSizes = NULL;
        MatrixEntry<T>** m_ppElements = NULL;

        SparseMatrix() = default;
        SparseMatrix(int rows) { this->rows = 0; Resize(rows); }
        SparseMatrix(const SparseMatrix& M) {
            Resize(M.rows);
            for (int i = 0; i < rows; i++) {
                SetRowSize(i, M.rowSizes[i]);
                for (int j = 0; j < rowSizes[i]; j++) { m_ppElements[i][j] = M.m_ppElements[i][j]; }
            }
        }
        ~SparseMatrix() { Resize(0); }

        void Resize(int r);

        /**
         * set m_ppElements[row] size be $count
         * rowSizes[row] = count
         * @param row
         * @param count
         */
        void SetRowSize(int row, int count);

        /**
         * @return sum of rowSizes[0...rows]
         */
        int Entries(void);



        void SetZero();
        void SetIdentity();

        SparseMatrix<T>& operator = (const SparseMatrix<T>& M);
        SparseMatrix<T> operator * (const T& V) const;
        SparseMatrix<T>& operator *= (const T& V);
        SparseMatrix<T> operator * (const SparseMatrix<T>& M) const;
        template<class T2>
        Vector<T2> operator * (const Vector<T2>& V) const;

        SparseMatrix<T> Multiply(const SparseMatrix<T>& M) const;
        template<class T2>
        Vector<T2> Multiply(const Vector<T2>& V) const;
        template<class T2>
        void Multiply(const Vector<T2>& In, Vector<T2>& Out) const;
        SparseMatrix<T> MultiplyTranspose(const SparseMatrix<T>& Mt) const;

        SparseMatrix<T> Transpose() const;

        static int Solve(const SparseMatrix<T>& M, const Vector<T>& b, const int& iters, Vector<T>& solution, const T eps = 1e-8);

        template<class T2>
        static int SolveSymmetric(const SparseMatrix<T>& M, const Vector<T2>& b, const int& iters, Vector<T2>& solution, const T2 eps = 1e-8, const int& reset = 1);

    };
    template<class T, int Dim> class SparseNMatrix
    {
    public:
        int rows = 0;
        int* rowSizes = NULL;
        NMatrixEntry<T, Dim>** m_ppElements = NULL;

        SparseNMatrix() = default;
        SparseNMatrix(int rows) { Resize(rows); }
        SparseNMatrix(const SparseNMatrix& M) {
            Resize(M.rows);
            for (int i = 0; i < rows; i++) {
                SetRowSize(i, M.rowSizes[i]);
                for (int j = 0; j < rowSizes[i]; j++) { m_ppElements[i][j] = M.m_ppElements[i][j]; }
            }
        }
        ~SparseNMatrix() { Resize(0); }

        void Resize(int r);
        void SetRowSize(int row, int count);
        int Entries(void);

        SparseNMatrix& operator = (const SparseNMatrix& M);

        SparseNMatrix  operator *  (const T& V) const;
        SparseNMatrix& operator *= (const T& V);

        template<class T2>
        NVector<T2, Dim> operator * (const Vector<T2>& V) const;
        template<class T2>
        Vector<T2> operator * (const NVector<T2, Dim>& V) const;
    };



    template <class T>
    class SparseSymmetricMatrix : public SparseMatrix<T> {
    public:

        template<class T2>
        Vector<T2> operator * (const Vector<T2>& V) const { return Multiply(V); }
        template<class T2>
        Vector<T2> Multiply(const Vector<T2>& V) const;
        /**     $Out = this.Dot(Vector $In)                                     */
        template<class T2>
        void Multiply(const Vector<T2>& In, Vector<T2>& Out) const;

        /**     Solve for x s.t. M(x)=b by solving for x s.t. M^tM(x)=M^t(b)
         *      return the iter times
         *      optimal x is saved in $solution                                 */
        template<class T2>
        static int Solve(const SparseSymmetricMatrix<T>& M, const Vector<T2>& b, const int& iters, Vector<T2>& solution, const T2 eps = 1e-8, const int& reset = 1);

        template<class T2>
        static int Solve(const SparseSymmetricMatrix<T>& M, const Vector<T>& diagonal, const Vector<T2>& b, const int& iters, Vector<T2>& solution, const T2 eps = 1e-8, const int& reset = 1);

        template <class T>
        struct MatrixEntry
        {
            MatrixEntry(void) { N = -1; Value = 0; }
            MatrixEntry(int i) { N = i; Value = 0; }
            int N;
            T Value;
        };
        template <class T, int Dim>
        struct NMatrixEntry
        {
            NMatrixEntry(void) { N = -1; memset(Value, 0, sizeof(T) * Dim); }
            NMatrixEntry(int i) { N = i; memset(Value, 0, sizeof(T) * Dim); }
            int N;
            T Value[Dim];
        };
    };
}



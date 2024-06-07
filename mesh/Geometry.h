/*****************************************************************//**
 * \file   Geometry.h
 * \brief  �㷨���ݽṹ
 * 
 * \author LUOJIAXUAN
 * \date   May 2nd 2024
 *********************************************************************/
#pragma once
#include <math.h>
#include <vector>
#include <device_launch_parameters.h>
#include <cuda_runtime_api.h>

namespace SparseSurfelFusion {
    /**
     * \brief ������ά�����ģ�庯��Ϊ��������(int, float, double).
     */
    template<class T>
    struct Point3D {
        T coords[3]; // x,y,z����
        /**
         * \brief ����[]�������ֵ(��const).
         *
         * \param i i��{0, 1, 2}
         * \return �����������
         */
        inline T& operator[] (int i) { return coords[i]; }
        /**
         * \brief ����[]�������ֵ(constֵ).
         *
         * \param i i��{0, 1, 2}
         * \return �����������
         */
        inline const T& operator[] (int i) const { return coords[i]; }
        /**
         * \brief ��ʼ�����캯�����������Ϊ0.
         *
         */
        __host__ __device__ Point3D() {
            coords[0] = 0;
            coords[1] = 0;
            coords[2] = 0;
        }
        /**
         * \brief �������Point3D�������ÿ���(ʵ�δ���)������ֵconst.
         *
         * \param cpy ��Ҫ���������ֵ
         */
        __host__ __device__ Point3D(const Point3D<T>& cpy) {
            coords[0] = cpy.coords[0];
            coords[1] = cpy.coords[1];
            coords[2] = cpy.coords[2];
        }

        /**
         * \brief ��3D�㸳ֵ.
         * 
         * \param x �����x����
         * \param y �����y����
         * \param z �����z����
         */
        __host__ __device__ Point3D(const T& x, const T& y, const T& z) {
            coords[0] = x;
            coords[1] = y;
            coords[2] = z;
        }

        /**
         * \brief ���صȺţ���Point3D��ֵ��ֵ���Ⱥ���ߵ�ֵ��const���ô���.
         *
         * \param cpy �ұ߸�ֵ������ֵ
         * \return �ұߵ�Point3D����
         */
        __host__ __device__ Point3D<T>& operator = (const Point3D<T>& cpy) {
            coords[0] = cpy.coords[0];
            coords[1] = cpy.coords[1];
            coords[2] = cpy.coords[2];
            return *this;
        }
    };
    template<class T>
    struct OrientedPoint3D {
        Point3D<T> point;   // ���������
        Point3D<T> normal;  // ����㷨��
    };
    /**
     * \brief OP��������ƽ�� = x^2 + y^2 + z^2.
     *
     * \param P����ά����
     * \return ����double����OP��������ƽ��
     */
    template<class T>
    __host__ __device__ double SquareLength(const Point3D<T>& p);

    /**
     * \brief OP�������� = sqrt(x^2 + y^2 + z^2).
     *
     * \param P����ά����
     * \return ����double����OP��������ƽ��
     */
    template<class T>
    double Length(const Point3D<T>& p);

    /**
     * \brief ��������֮�����.
     *
     * \param p1 ��p1����
     * \param p2 ��p2����
     * \return ����double��������֮��ľ���
     */
    template<class T>
    double Distance(const Point3D<T>& p1, const Point3D<T>& p2);

    /**
     * \brief ����֮��ľ���ƽ��.
     *
     * \param p1 ��p1����
     * \param p2 ��p2����
     * \return ����double��������֮��ľ���ƽ��
     */
    template<class T>
    __host__ __device__ double SquareDistance(const Point3D<T>& p1, const Point3D<T>& p2);

    /**
     * \brief OP1��OP2���������Ĳ�˽��.
     *
     * \param p1 ��p1����(const����)
     * \param p2 ��p2����(const����)
     * \param p ��˺����������
     */
    template <class T>
    void CrossProduct(const Point3D<T>& p1, const Point3D<T>& p2, Point3D<T>& p);


    /**
     * \brief 2D�߱�ʾ���ݽṹ�������е�Fig.2.(a)
     *     p1 = (p[0][0], p[0][1])
     *     p2 = (p[1][0], p[1][1])
     */
    class Edge {
    public:
        double p[2][2]; // ��¼������������
        /**
         * \brief ��ñߵĳ���.
         *
         * \return �ߵĳ���(double)
         */
        double Length(void) const {
            double d[2];
            d[0] = p[0][0] - p[1][0];   // p1.x - p2.x
            d[1] = p[0][1] - p[1][1];   // p1.y - p2.y
            return sqrt(d[0] * d[0] + d[1] * d[1]);
        }
    };

    /**
     * \brief 3D�����α�ʾ���ݽṹ
     *     x = (p[0][0], p[0][1], p[0][2])
     *     y = (p[1][0], p[1][1], p[1][2])
     *     z = (p[2][0], p[2][1], p[2][2])
     */
    class Triangle {
    public:
        double p[3][3];

        /**
         * \brief �������������.
         *
         * \return �������������
         */
        double Area(void) const {
            double v1[3], v2[3], v[3];
            for (int d = 0; d < 3; d++) {
                v1[d] = p[1][d] - p[0][d];
                v2[d] = p[2][d] - p[0][d];
            }
            v[0] = v1[1] * v2[2] - v1[2] * v2[1];
            v[1] = -v1[0] * v2[2] + v1[2] * v2[0];
            v[2] = v1[0] * v2[1] - v1[1] * v2[0];
            return sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]) / 2;
        }

        /**
         * \brief �������ݺ�� AspectRatio = abc/[8(s - a)(s - b)(s - c)], ���� s = (a + b + c) / 2.
         *
         * \return �������ݺ��
         */
        double AspectRatio(void) const {
            double v0[3] = { 0 };   double v1[3] = { 0 };   double v2[3] = { 0 };   // ��¼������
            for (int i = 0; i < 3; i++) {
                v0[i] = p[0][i] - p[2][i];
                v1[i] = p[1][i] - p[0][i];
                v2[i] = p[2][i] - p[1][i];
            }
            double Va = 0;  double Vb = 0;  double Vc = 0;  // ����߳�
            for (int i = 0; i < 3; i++) {
                Va += v0[i] * v0[i];
                Vb += v1[i] * v1[i];
                Vc += v2[i] * v2[i];
            }
            Va = sqrt(Va);  Vb = sqrt(Vb);  Vc = sqrt(Vc);  // �߳�����
            return (Va * Vb * Vc) / ((Va + Vc) * (Vb + Vc) * (Va + Vb));
        }
    };

    class CoredPointIndex {
    public:
        int index;
        char inCore;
        /**
         * \brief ���صȺ�"==".
         *
         * \param cpi
         * \return
         */
        int operator == (const CoredPointIndex& cpi) const {
            return (index == cpi.index) && (inCore == cpi.inCore);
        };
    };

    class EdgeIndex {
    public:
        int idx[2];
    };

    class CoredEdgeIndex {
    public:
        CoredPointIndex idx[2];
    };

    class TriangleIndex {
    public:
        int idx[3];
    };

    template<class T>
    void EdgeCollapse(const T& edgeRatio, std::vector<TriangleIndex>& triangles, std::vector< Point3D<T> >& positions, std::vector<Point3D<T> >* normals);

    template<class T>
    void TriangleCollapse(const T& edgeRatio, std::vector<TriangleIndex>& triangles, std::vector<Point3D<T> >& positions, std::vector<Point3D<T> >* normals);

    class CoredMeshData {
    public:
        std::vector<Point3D<float> > inCorePoints;  // ���ĵ㣬Ӧ���ǹ�������ĵ�
        const static int IN_CORE_FLAG[3];
        virtual void resetIterator(void) = 0;

        virtual int addOutOfCorePoint(const Point3D<float>& p) = 0;
        virtual int addTriangle(const TriangleIndex& t, const int& icFlag = (IN_CORE_FLAG[0] | IN_CORE_FLAG[1] | IN_CORE_FLAG[2])) = 0;

        virtual int nextOutOfCorePoint(Point3D<float>& p) = 0;
        virtual int nextTriangle(TriangleIndex& t, int& inCoreFlag) = 0;

        virtual int outOfCorePointCount(void) = 0;
        virtual int triangleCount(void) = 0;
    };

    class CoredVectorMeshData : public CoredMeshData {
        std::vector<Point3D<float> > oocPoints;     // �Ǻ��ĵ㣬Ӧ����ƫ������ĵ㣬��Ӧ�û���������
        std::vector<TriangleIndex> triangles;       // ������Mesh������
        /**     use as iterator     **/
        int oocPointIndex, triangleIndex;
    public:
        CoredVectorMeshData(void);

        void resetIterator(void);

        int addOutOfCorePoint(const Point3D<float>& p);
        int addTriangle(const TriangleIndex& t, const int& inCoreFlag = (CoredMeshData::IN_CORE_FLAG[0] | CoredMeshData::IN_CORE_FLAG[1] | CoredMeshData::IN_CORE_FLAG[2]));

        int nextOutOfCorePoint(Point3D<float>& p);
        int nextTriangle(TriangleIndex& t, int& inCoreFlag);

        /**     Count all, include iterated     */
        int outOfCorePointCount(void);
        int InCorePointsCount(void);
        int triangleCount(void);

        bool GetTriangleIndices(std::vector<unsigned int>& triangleIndices);        // �����Ҫ���Ƶ�������index
        bool GetTriangleIndices(std::vector<TriangleIndex>& triangleIndices);       // �����Ҫ���Ƶ�������index

        bool GetVertexArray(std::vector<float>& vertexArray);                       // ��ö�������
        bool GetVertexArray(std::vector<Point3D<float>>& vertexArray);              // ��ö�������

        void clearAllContainer();   // �����������
    };
}

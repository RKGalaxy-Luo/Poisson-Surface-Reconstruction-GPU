/*****************************************************************//**
 * \file   Geometry.h
 * \brief  算法数据结构
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
     * \brief 构建三维点对象，模板函数为精度设置(int, float, double).
     */
    template<class T>
    struct Point3D {
        T coords[3]; // x,y,z坐标
        /**
         * \brief 重载[]获得坐标值(非const).
         *
         * \param i i∈{0, 1, 2}
         * \return 返回坐标分量
         */
        inline T& operator[] (int i) { return coords[i]; }
        /**
         * \brief 重载[]获得坐标值(const值).
         *
         * \param i i∈{0, 1, 2}
         * \return 返回坐标分量
         */
        inline const T& operator[] (int i) const { return coords[i]; }
        /**
         * \brief 初始化构造函数，点坐标均为0.
         *
         */
        __host__ __device__ Point3D() {
            coords[0] = 0;
            coords[1] = 0;
            coords[2] = 0;
        }
        /**
         * \brief 将传入的Point3D进行引用拷贝(实参传递)，常量值const.
         *
         * \param cpy 需要构造的坐标值
         */
        __host__ __device__ Point3D(const Point3D<T>& cpy) {
            coords[0] = cpy.coords[0];
            coords[1] = cpy.coords[1];
            coords[2] = cpy.coords[2];
        }

        /**
         * \brief 给3D点赋值.
         * 
         * \param x 传入的x坐标
         * \param y 传入的y坐标
         * \param z 传入的z坐标
         */
        __host__ __device__ Point3D(const T& x, const T& y, const T& z) {
            coords[0] = x;
            coords[1] = y;
            coords[2] = z;
        }

        /**
         * \brief 重载等号，将Point3D的值赋值给等号左边的值，const引用传入.
         *
         * \param cpy 右边赋值的坐标值
         * \return 右边的Point3D数据
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
        Point3D<T> point;   // 有向点坐标
        Point3D<T> normal;  // 有向点法线
    };
    /**
     * \brief OP向量长度平方 = x^2 + y^2 + z^2.
     *
     * \param P点三维坐标
     * \return 返回double类型OP向量长度平方
     */
    template<class T>
    __host__ __device__ double SquareLength(const Point3D<T>& p);

    /**
     * \brief OP向量长度 = sqrt(x^2 + y^2 + z^2).
     *
     * \param P点三维坐标
     * \return 返回double类型OP向量长度平方
     */
    template<class T>
    double Length(const Point3D<T>& p);

    /**
     * \brief 计算两点之间距离.
     *
     * \param p1 点p1坐标
     * \param p2 点p2坐标
     * \return 返回double类型两点之间的距离
     */
    template<class T>
    double Distance(const Point3D<T>& p1, const Point3D<T>& p2);

    /**
     * \brief 两点之间的距离平方.
     *
     * \param p1 点p1坐标
     * \param p2 点p2坐标
     * \return 返回double类型两点之间的距离平方
     */
    template<class T>
    __host__ __device__ double SquareDistance(const Point3D<T>& p1, const Point3D<T>& p2);

    /**
     * \brief OP1和OP2两个向量的叉乘结果.
     *
     * \param p1 点p1坐标(const常量)
     * \param p2 点p2坐标(const常量)
     * \param p 叉乘后的向量坐标
     */
    template <class T>
    void CrossProduct(const Point3D<T>& p1, const Point3D<T>& p2, Point3D<T>& p);


    /**
     * \brief 2D边表示数据结构，论文中的Fig.2.(a)
     *     p1 = (p[0][0], p[0][1])
     *     p2 = (p[1][0], p[1][1])
     */
    class Edge {
    public:
        double p[2][2]; // 记录边上两点坐标
        /**
         * \brief 获得边的长度.
         *
         * \return 边的长度(double)
         */
        double Length(void) const {
            double d[2];
            d[0] = p[0][0] - p[1][0];   // p1.x - p2.x
            d[1] = p[0][1] - p[1][1];   // p1.y - p2.y
            return sqrt(d[0] * d[0] + d[1] * d[1]);
        }
    };

    /**
     * \brief 3D三角形表示数据结构
     *     x = (p[0][0], p[0][1], p[0][2])
     *     y = (p[1][0], p[1][1], p[1][2])
     *     z = (p[2][0], p[2][1], p[2][2])
     */
    class Triangle {
    public:
        double p[3][3];

        /**
         * \brief 计算三角形面积.
         *
         * \return 返回三角形面积
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
         * \brief 三角形纵横比 AspectRatio = abc/[8(s - a)(s - b)(s - c)], 其中 s = (a + b + c) / 2.
         *
         * \return 三角形纵横比
         */
        double AspectRatio(void) const {
            double v0[3] = { 0 };   double v1[3] = { 0 };   double v2[3] = { 0 };   // 记录边向量
            for (int i = 0; i < 3; i++) {
                v0[i] = p[0][i] - p[2][i];
                v1[i] = p[1][i] - p[0][i];
                v2[i] = p[2][i] - p[1][i];
            }
            double Va = 0;  double Vb = 0;  double Vc = 0;  // 计算边长
            for (int i = 0; i < 3; i++) {
                Va += v0[i] * v0[i];
                Vb += v1[i] * v1[i];
                Vc += v2[i] * v2[i];
            }
            Va = sqrt(Va);  Vb = sqrt(Vb);  Vc = sqrt(Vc);  // 边长长度
            return (Va * Vb * Vc) / ((Va + Vc) * (Vb + Vc) * (Va + Vb));
        }
    };

    class CoredPointIndex {
    public:
        int index;
        char inCore;
        /**
         * \brief 重载等号"==".
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
        std::vector<Point3D<float> > inCorePoints;  // 核心点，应该是构造网格的点
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
        std::vector<Point3D<float> > oocPoints;     // 非核心点，应该是偏离网格的点，不应该画在网格中
        std::vector<TriangleIndex> triangles;       // 三角形Mesh的索引
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

        bool GetTriangleIndices(std::vector<unsigned int>& triangleIndices);        // 获得需要绘制的三角形index
        bool GetTriangleIndices(std::vector<TriangleIndex>& triangleIndices);       // 获得需要绘制的三角形index

        bool GetVertexArray(std::vector<float>& vertexArray);                       // 获得顶点数组
        bool GetVertexArray(std::vector<Point3D<float>>& vertexArray);              // 获得顶点数组

        void clearAllContainer();   // 清空所有容器
    };
}

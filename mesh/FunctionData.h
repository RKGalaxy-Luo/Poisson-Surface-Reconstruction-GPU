/*****************************************************************//**
 * \file   FunctionData.h
 * \brief  基函数方法及模板方法实现
 * 
 * \author LUOJIAXUAN
 * \date   May 15th 2024
 *********************************************************************/
#pragma once
#include <math/PPolynomial.h>
#include "BinaryNode.h"
namespace SparseSurfelFusion {

	template<class Real>
	class FunctionValueTable;	// 前向声明

	template<int Degree, class Real>
	class FunctionData
	{
        /**
         * 是否缩放dDotTable, d2DotTable与dotTable 
         */
        int useDotRatios;   // 是否进行点积比的计算【true：计算完一阶或者二阶导函数点积后除以原始函数的点积；false：直接计算函数点积】
        int normalize;      // 函数点积归一化的类型【0：不进行归一化；1：进行L1-范数归一化；2：进行L2-范数归一化
    public:
        const static int     DOT_FLAG = 1;  // 是否记录基函数的点积表
        const static int   D_DOT_FLAG = 2;  // 是否记录基函数一阶导数的点积表
        const static int  D2_DOT_FLAG = 4;  // 是否记录基函数二阶导数的点积表
        const static int   VALUE_FLAG = 1;  // 是否记录基函数值
        const static int D_VALUE_FLAG = 2;  // 是否记录基函数导数值

        /**
         * Res是分辨率
         */
        int res = 0;                    // 记录基函数点积表的大小
        int depth, res2;
        double* dotTable = NULL;        // 基函数的函数内积表
        double* dDotTable = NULL;       // 基函数一阶导数的函数内积表
        double* d2DotTable = NULL;      // 基函数二阶导数的函数内积表
        double* valueTables = NULL;     // 基函数值的表
        double* dValueTables = NULL;    // 基函数导数值的表
        PPolynomial<Degree> baseFunction;
        /**
         * baseFunction的导函数
         */
        PPolynomial<Degree - 1> dBaseFunction;
        /**
         * baseFunction的原函数.
         */
        PPolynomial<Degree + 1>* baseFunctions;

        FunctionData(void) = default;
        ~FunctionData(void) {
            if (res) {
                delete[] dotTable;
                delete[] dDotTable;
                delete[] d2DotTable;
                delete[] valueTables;
                delete[] dValueTables;
            }
            dotTable = dDotTable = d2DotTable = NULL;
            valueTables = dValueTables = NULL;
            res = 0;
        }

        /**     假设maxDepth为2，假设REAL为一维
          *     the center and width of $index is
          *     [0] 0.500000, 1.000000
          *     [1] 0.250000, 0.500000
          *     [2] 0.750000, 0.500000
          *     [3] 0.125000, 0.250000
          *     [4] 0.375000, 0.250000
          *     [5] 0.625000, 0.250000
          *     [6] 0.875000, 0.250000
          *     $normalize scale the $F function and assign it to $baseFunction, so that it has:
          *     [0] Value 1 at 0
          *     [1] Integral equal to 1
          *     [2] Square integral equal to 1
          *     scale, shift and normalize the $baseFunction by center and width of $index,
          *     then assign it to baseFunctions[$index]
          *     baseFunctions[$index].start = (start * width) + center
          *     e.g :    start in [-1, 1]
          *              will be reflected into [center - width, center + width]    */
        void set(const int& maxDepth, const PPolynomial<Degree>& F, const int& normalize, const int& useDotRatios = 1) {
            this->normalize = normalize;
            this->useDotRatios = useDotRatios;

            this->depth = maxDepth;
            this->res = BinaryNode<double>::CumulativeCenterCount(this->depth);
            this->res2 = (1 << (this->depth + 1)) + 1;
            this->baseFunctions = new PPolynomial<Degree + 1>[res];
            // Scale the function so that it has:
            // 0] Value 1 at 0
            // 1] Integral equal to 1
            // 2] Square integral equal to 1
            switch (normalize) {
            case 2:
                baseFunction = F / sqrt((F * F).integral(F.polys[0].start, F.polys[F.polyCount - 1].start));
                break;
            case 1:
                baseFunction = F / F.integral(F.polys[0].start, F.polys[F.polyCount - 1].start);
                break;
            default:
                baseFunction = F / F(0);
            }
            dBaseFunction = baseFunction.derivative();
            double c1, w1;
            for (int i = 0; i < res; i++) {
                BinaryNode<double>::CenterAndWidth(i, c1, w1);
                // printf("original start:%lf\n",baseFunction.polys[1].start);
                // baseFunction.printnl();
                // printf("%lf,%lf\n",c1,w1);
                baseFunctions[i] = baseFunction.scale(w1).shift(c1);
                // printf("next start:%lf\n",baseFunctions[i].polys[1].start);
                // baseFunctions[i].printnl();
                // Scale the function so that it has L2-norm equal to one
                switch (normalize) {
                case 2:
                    baseFunctions[i] /= sqrt(w1);
                    break;
                case 1:
                    baseFunctions[i] /= w1;
                    break;
                }
            }
        }


        /**     if   (flags &   DOT_FLAG)  为真, 设置dotTable  
          *          (flags & D_DOT_FLAG)  为真, 设置dDotTable 
          *          (flags & D2_DOT_FLAG) 为真, 设置d2DotTable
          *    数据表包含baseFunctions的内积
          *    这些数组的大小为[res * res]                      
          */
        void setDotTables(const int& flags) {
            clearDotTables(flags);
            if (flags & DOT_FLAG) {
                dotTable = new double[res * res];
                memset(dotTable, 0, sizeof(double) * res * res);
            }
            if (flags & D_DOT_FLAG) {
                dDotTable = new double[res * res];
                memset(dDotTable, 0, sizeof(double) * res * res);
            }
            if (flags & D2_DOT_FLAG) {
                d2DotTable = new double[res * res];
                memset(d2DotTable, 0, sizeof(double) * res * res);
            }

            double t1, t2;
            t1 = baseFunction.polys[0].start;
            t2 = baseFunction.polys[baseFunction.polyCount - 1].start;
            for (int i = 0; i < res; i++) {
                double c1, c2, w1, w2;
                BinaryNode<double>::CenterAndWidth(i, c1, w1);
                // 将0点的函数中心映射到它的实际位置
                double start1 = t1 * w1 + c1;
                double end1 = t2 * w1 + c1;
                for (int j = 0; j <= i; j++) {
                    BinaryNode<double>::CenterAndWidth(j, c2, w2);
                    int idx1 = i + res * j;
                    int idx2 = j + res * i;

                    double start = t1 * w2 + c2;
                    double end = t2 * w2 + c2;

                    if (start < start1) { start = start1; }
                    if (end > end1) { end = end1; }
                    if (start >= end) { continue; }

                    BinaryNode<double>::CenterAndWidth(j, c2, w2);
                    double dot = dotProduct(c1, w1, c2, w2);
                    if (fabs(dot) < 1e-15) { continue; }
                    if (flags & DOT_FLAG) { dotTable[idx1] = dotTable[idx2] = dot; }
                    if (useDotRatios) {
                        if (flags & D_DOT_FLAG) {
                            dDotTable[idx1] = dDotProduct(c1, w1, c2, w2) / dot;
                            dDotTable[idx2] = -dDotTable[idx1];
                        }
                        if (flags & D2_DOT_FLAG) { d2DotTable[idx1] = d2DotTable[idx2] = d2DotProduct(c1, w1, c2, w2) / dot; }
                    }
                    else {
                        if (flags & D_DOT_FLAG) {
                            dDotTable[idx1] = dDotProduct(c1, w1, c2, w2);
                            dDotTable[idx2] = -dDotTable[idx1];
                        }
                        if (flags & D2_DOT_FLAG) { d2DotTable[idx1] = d2DotTable[idx2] = d2DotProduct(c1, w1, c2, w2); }
                    }
                }
            }
        }
        /**
         * \brief 清空点积查询表.
         * 
         * \param flags 标志位
         */
        void clearDotTables(const int& flags) {
            if ((flags & DOT_FLAG) && dotTable) {
                delete[] dotTable;
                dotTable = NULL;
            }
            if ((flags & D_DOT_FLAG) && dDotTable) {
                delete[] dDotTable;
                dDotTable = NULL;
            }
            if ((flags & D2_DOT_FLAG) && d2DotTable) {
                delete[] d2DotTable;
                d2DotTable = NULL;
            }
        }

        /**     如果   (flags &   VALUE_FLAG) 是 True, 计算设置 valueTables
          *     如果   (flags & D_VALUE_FLAG) 是 True, 计算设置 dValueTables
          *     valueTables[i*res2 -- i*res2+res2-1] 是被平滑的基函数 baseFunctions[i]
          *     discrete value from [0, 1]
          *     so is dValueTables.
          *     size of these array is all [res * res2]                     */
        void setValueTables(const int& flags, const double& smooth = 0) {
            clearValueTables();
            if (flags & VALUE_FLAG) { valueTables = new double[res * res2]; }
            if (flags & D_VALUE_FLAG) { dValueTables = new double[res * res2]; }
            PPolynomial<Degree + 1> function;
            PPolynomial<Degree>  dFunction;
            for (int i = 0; i < res; i++) {
                if (smooth > 0) {
                    function = baseFunctions[i].MovingAverage(smooth);
                    dFunction = baseFunctions[i].derivative().MovingAverage(smooth);
                }
                else {
                    function = baseFunctions[i];
                    dFunction = baseFunctions[i].derivative();
                }
                for (int j = 0; j < res2; j++) {
                    double x = double(j) / (res2 - 1);
                    if (flags & VALUE_FLAG) { valueTables[i * res2 + j] = function(x); }
                    if (flags & D_VALUE_FLAG) { dValueTables[i * res2 + j] = dFunction(x); }
                }
            }
        }
        void clearValueTables(void) {
            if (valueTables) {
                delete[] valueTables;
                valueTables = NULL;
            }
            if (dValueTables) {
                delete[] dValueTables;
                dValueTables = NULL;
            }
        }



        /**     <F1, F2> inner product      */
        /**
         * \brief 计算基函数1和基函数2之间的点积<F, F>【点积表示了两个函数在给定区间上的相似程度。如果点积结果为非零，则说明这两个函数在该区间上有一定的相似性】.
         * 
         * \param center1 基函数1的中心值
         * \param width1 基函数1的宽度
         * \param center2 基函数2的中心值
         * \param width2 基函数2的中心值
         * \return 两个基函数的点积
         */
        double dotProduct(const double& center1, const double& width1, const double& center2, const double& width2) const {
            double r = fabs(baseFunction.polys[0].start);
            switch (normalize) {
            case 2:
                return (baseFunction * baseFunction.scale(width2 / width1).shift((center2 - center1) / width1)).integral(-2 * r, 2 * r) * width1 / sqrt(width1 * width2);
            case 1:
                return (baseFunction * baseFunction.scale(width2 / width1).shift((center2 - center1) / width1)).integral(-2 * r, 2 * r) * width1 / (width1 * width2);
            default:
                return (baseFunction * baseFunction.scale(width2 / width1).shift((center2 - center1) / width1)).integral(-2 * r, 2 * r) * width1;
            }
        }
        /**
         * \brief 计算基函数一阶导数的点积<F, dF>【点积表示了两个函数在给定区间上的相似程度。如果点积结果为非零，则说明这两个函数在该区间上有一定的相似性】.
         * 
         * \param center1 基函数1的中心值
         * \param width1 基函数1的宽度
         * \param center2 基函数2的中心值
         * \param width2 基函数2的中心值
         * \return 两个基函数导数的点积
         */
        double  dDotProduct(const double& center1, const double& width1, const double& center2, const double& width2) const {
            double r = fabs(baseFunction.polys[0].start);
            switch (normalize) {
            case 2:
                return (dBaseFunction * baseFunction.scale(width2 / width1).shift((center2 - center1) / width1)).integral(-2 * r, 2 * r) / sqrt(width1 * width2);
            case 1:
                return (dBaseFunction * baseFunction.scale(width2 / width1).shift((center2 - center1) / width1)).integral(-2 * r, 2 * r) / (width1 * width2);
            default:
                return (dBaseFunction * baseFunction.scale(width2 / width1).shift((center2 - center1) / width1)).integral(-2 * r, 2 * r);
            }
        }
        /**
         * \brief 计算基函数二阶导数的点积<F, d2F>【点积表示了两个函数在给定区间上的相似程度。如果点积结果为非零，则说明这两个函数在该区间上有一定的相似性】.
         *
         * \param center1 基函数1的中心值
         * \param width1 基函数1的宽度
         * \param center2 基函数2的中心值
         * \param width2 基函数2的中心值
         * \return 两个基函数导数的点积
         */
        double d2DotProduct(const double& center1, const double& width1, const double& center2, const double& width2) const {
            double r = fabs(baseFunction.polys[0].start);
            switch (normalize) {
            case 2:
                return (dBaseFunction * dBaseFunction.scale(width2 / width1).shift((center2 - center1) / width1)).integral(-2 * r, 2 * r) / width2 / sqrt(width1 * width2);
            case 1:
                return (dBaseFunction * dBaseFunction.scale(width2 / width1).shift((center2 - center1) / width1)).integral(-2 * r, 2 * r) / width2 / (width1 * width2);
            default:
                return (dBaseFunction * dBaseFunction.scale(width2 / width1).shift((center2 - center1) / width1)).integral(-2 * r, 2 * r) / width2;
            }
        }
	};



    template<class Real>
    class FunctionValueTable {
        int start = -1;
        int size = 0;
        Real * values = NULL;
    public:
        FunctionValueTable(void) = default;
        ~FunctionValueTable(void) {
            if (this->values) { delete[] this->values; }
            start = -1;
            size = 0;
            values = NULL;
        }

        /**     返回poly(idx/res)，如果value未定义则返回0                            */
        inline Real operator[] (const int& idx) {
            int i = idx - start;
            if (i < 0 || i >= size) { return 0; }
            else { return this->values[i]; }
        }

        /**     res为分辨率，[0,1]中poly(i/ Res)的离散值保存为values   */
        template<int Degree>
        int setValues(const PPolynomial<Degree>& ppoly, const int& res) {
            int j;
            if (values) { delete[] values; }
            start = -1;
            size = 0;
            values = NULL;
            for (j = 0; j < res; j++) {
                double x = double(j) / (res - 1);
                if (x > ppoly.polys[0].start && x < ppoly.polys[ppoly.polyCount - 1].start) {
                    if (start == -1) { start = j; }
                    size = j + 1 - start;
                }
            }
            if (size) {
                values = new Real[size];
                for (j = 0; j < size; j++) {
                    double x = double(j + start) / (res - 1);
                    values[j] = Real(ppoly(x));
                }
            }
            return size;
        }

    };
}



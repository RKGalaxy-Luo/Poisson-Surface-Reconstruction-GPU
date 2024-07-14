/*****************************************************************//**
 * \file   PPolynomial.h
 * \brief  多项式求解
 * 
 * \author LUOJIAXUAN
 * \date   May 21st 2024
 *********************************************************************/
#pragma once
#include "Polynomial.h"
#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>
namespace SparseSurfelFusion {

    /**
     * \brief 带有起点的多项式，start就是该多项式定义域的左边界，即p(x) = f(x)  x∈[start, +∞).
     */
    template<int Degree>
    class StartingPolynomial
    {
    public:
        Polynomial<Degree> p;   // 多项式
        float start;            // 多项式的起点

        /**	    return a StartingPolynomial
          *	    new start is the bigger start
          *	    polynomials are multiplied  */
        /**
         * \brief 多项式乘法，其多项式为两个多项式的乘积，起始点为两个起始点中较大的一个.
         * 
         * \param p 多项式
         * \return 相乘后的结果
         */
        template<int Degree2>
        __host__ __device__ StartingPolynomial<Degree + Degree2>  operator * (const StartingPolynomial<Degree2>& p) const {
            StartingPolynomial<Degree + Degree2> sp;
            if (start > p.start) { sp.start = start; }
            else { sp.start = p.start; }
            sp.p = this->p * p.p;
            return sp;
        }

        /**
         * \brief 重载定义了赋值操作，使得可以将一个 StartingPolynomial 对象的值赋给另一个对象.
         * 
         * \param sp 赋值者
         * \return 被赋值对象
         */
        __host__ __device__ StartingPolynomial& operator = (const StartingPolynomial& sp) {
            start = sp.start;
            for (int i = 0; i <= Degree; ++i)
                p.coefficients[i] = sp.p.coefficients[i];
            return *this;
        }

        /**
         * \brief 起始点和多项式都按比例 s 缩放.
         * 
         * \param s 缩放比例
         * \return 缩放后的多项式
         */
        __host__ __device__ StartingPolynomial scale(const float& s) const {
            StartingPolynomial q;
            q.start = start * s;
            q.p = p.scale(s);
            return q;
        }

        /**
         * \brief 起点平移【start = start + t】，多项式也平移【f(x) -> f(x-t)】.
         * 
         * \param t 平移量
         * \return 平移后的多项式
         */
        __host__ __device__ StartingPolynomial shift(const float& t) const {
            StartingPolynomial q;
            q.start = start + t;
            q.p = p.shift(t);
            return q;
        }

        /**
         * \brief 比较哪个多项式的起点更小，如果当前对象的起始点小于 sp 的起始点，则返回 1，否则返回 0.
         * 
         * \param sp 比较对象
         * \return 是否更小
         */
        __host__ __device__ int operator < (const StartingPolynomial& sp) const {
            if (start < sp.start) {
                return 1;
            }
            else {
                return 0;
            }
        }

        /**     v1 > v2 , return 1
          *     v1 < v2 , return -1
          *     v1 = v2 , return 0  */
        /**
         * \brief 比较多项式的起点大小：v1 > v2 , return 1；v1 < v2 , return -1；v1 = v2 , return 0.
         * 
         * \param v1 多项式v1
         * \param v2 多项式v2
         * \return 多项式比较的结果
         */
        __host__ __device__ static int Compare(const void* v1, const void* v2) {
            float d = ((StartingPolynomial*)(v1))->start - ((StartingPolynomial*)(v2))->start;
            if (d < 0) { return -1; }
            else if (d > 0) { return  1; }
            else { return  0; }
        }
    };

    /**
     * \brief 多项式集合(构成B样条函数的基函数集合).
     */
    template<int Degree>
	class PPolynomial
	{
	public:
        size_t polyCount = 0;                       // 多项式集合元素的数量
        StartingPolynomial<Degree>* polys = NULL;   // 带起点(有定义域的)的多项式(组)

        PPolynomial(void) = default;
        /**
         * \brief 将多项式p拷贝赋值给当前对象.
         * 
         * \param 多项式p 
         */
        PPolynomial(const PPolynomial<Degree>& p) {
            set(p.polyCount);   // 分配缓存
            memcpy(polys, p.polys, sizeof(StartingPolynomial<Degree>) * p.polyCount);
        }
        ~PPolynomial(void) {
            if (polyCount) { free(polys); }
            polyCount = 0;
            polys = NULL;
        }

        /**
         * \brief 获得当前多项式集合的byte大小.
         * 
         * \return 返回当前多项式集合的byte大小
         */
        int size(void) const {
            return int(sizeof(StartingPolynomial<Degree>) * polyCount);
        }

        /**
         * \param 释放之前的多项式，并重新分配当前多项式的内存.
         * 
         * \param size 多项式的项数
         */
        void set(const size_t& size) {
            if (polyCount) {
                free(polys);
            }
            polyCount = 0;
            polys = NULL;
            polyCount = size;
            if (size) {
                polys = (StartingPolynomial<Degree>*)malloc(sizeof(StartingPolynomial<Degree>) * size);
                memset(polys, 0, sizeof(StartingPolynomial<Degree>) * size);
            }
        }


        /**
         * \brief 对输入的 StartingPolynomial 对象数组进行升序排序，并将具有相同起点的多项式合并在一起.
         * 
         * \param sps 起点多项式
         * \param count 多项式数量
         */
        void set(StartingPolynomial<Degree>* sps, const int& count) {
            int i = 0;  // 起点多项式数组的index
            int c = 0;  // 记录不同起点的多项式的偏移
            set(count);
            // 根据StartingPolynomial<Degree>::Compare的方式进行排序：即根据start的大小
            qsort(sps, count, sizeof(StartingPolynomial<Degree>), StartingPolynomial<Degree>::Compare);
            for (i = 0; i < count; i++) {
                if (!c || sps[i].start != polys[c - 1].start) {
                    polys[c++] = sps[i];
                }
                else {
                    polys[c - 1].p += sps[i].p;
                }
            }
            reset(c);
        }

        /**
         * \brief 重新分配扩展多项式的内存大小，其中旧多项式的数据会被保留.
         * 
         * \param newSize 新多项式的大小
         */
        void reset(const size_t& newSize) {
            polyCount = newSize;
            polys = (StartingPolynomial<Degree>*)realloc(polys, sizeof(StartingPolynomial<Degree>) * newSize);
        }


        /**     assume that StartPolynomial is sorted by set() function
          *     calculate f0(t) + f1(t) + f2(t) + ... + fn(t)
          *     StartPolynomial n+1's start >= t    */
        /**
         * \brief 【假设起始多项式集合被set函数排序】获得多项式集合中满足start < t的多项式，这些多项式在t点处的数值和(f0(t) + f1(t) + f2(t) + ... + fn(t)).
         * 
         * \param t 自变量t，阈值t
         * \return 返回多项式集合中满足start < t的多项式的和
         */
        float operator()(const float& t) const {
            float v = 0;
            for (int i = 0; i < int(polyCount) && t > polys[i].start; i++)  { v += polys[i].p(t); }
            return v;
        }


        /**     calculate the definite integral, integral start from the p[i].start, not min(tMin, tMax)
          *     let end = max(tMin, tMax)
          *     p[0].start / end [f0(x)dx] + p[1].start / end [f1(x)dx] + ... + p[n].start / end [fn(x)dx]
          *     p[n+1].start >= min(tMin, tMax)
          *     tMin can be bigger than tMax    */

        /**
         * \brief 这个 integral 函数用于计算在给定区间 [tMin,tMax] 内的多个 StartingPolynomial 对象的定积分的总和。
         * 
         * \param tMin 积分区间的起点
         * \param tMax 积分区间的终点
         * \return 积分的结果
         */
        float integral(const float& tMin, const float& tMax) const {
            int m = 1;  // 定积分符号
            float start, end, s, v = 0;
            start = tMin;
            end = tMax;
            if (tMin > tMax) {  // 定积分上下界确定符号
                m = -1;
                start = tMax;
                end = tMin;
            }
            for (int i = 0; i < int(polyCount) && polys[i].start < end; i++) {
                if (start < polys[i].start) { s = polys[i].start; } // 还未到达下界，将多项式的起点赋值给s
                else { s = start; }
                v += polys[i].p.integral(s, end);   // 在定义域范围内的积分
            }
            return v * m;
        }


        /**
         * \brief 这里应该是计算函数集合中所有元素非公共区间的积分【前提是函数已经排序】.
         * 
         * \return 函数集合中所有元素非公共区间的积分【前提是函数已经排序】
         */
        float Integral(void) const {
            return integral(polys[0].start, polys[polyCount - 1].start);
        }

        PPolynomial& operator = (const PPolynomial& p) {
            set(p.polyCount);
            memcpy(polys, p.polys, sizeof(StartingPolynomial<Degree>) * p.polyCount);
            return *this;
        }

        template<int Degree2>
        PPolynomial<Degree>& operator = (const PPolynomial<Degree2>& p) {
            set(p.polyCount);
            for (int i = 0; i<int(polyCount); i++) {
                polys[i].start = p.polys[i].start;
                polys[i].p = p.polys[i].p;
            }
            return *this;
        }

        PPolynomial  operator + (const PPolynomial& p) const {
            PPolynomial q;
            int i, j;
            size_t idx = 0;
            q.set(polyCount + p.polyCount);
            i = j = -1;

            while (idx < q.polyCount) {
                if (j >= int(p.polyCount) - 1) { q.polys[idx] = polys[++i]; }
                else if (i >= int(polyCount) - 1) { q.polys[idx] = p.polys[++j]; }
                else if (polys[i + 1].start < p.polys[j + 1].start) { q.polys[idx] = polys[++i]; }
                else { q.polys[idx] = p.polys[++j]; }
                //		if(idx && polys[idx].start==polys[idx-1].start)	{polys[idx-1].p+=polys[idx].p;}
                //		else{idx++;}
                idx++;
            }
            return q;
        }

        PPolynomial  operator - (const PPolynomial& p) const {
            PPolynomial q;
            int i, j;
            size_t idx = 0;
            q.set(polyCount + p.polyCount);
            i = j = -1;

            while (idx < q.polyCount) {
                if (j >= int(p.polyCount) - 1) { q.polys[idx] = polys[++i]; }
                else if (i >= int(polyCount) - 1) { q.polys[idx].start = p.polys[++j].start; q.polys[idx].p = p.polys[j].p * (-1.0); }
                else if (polys[i + 1].start < p.polys[j + 1].start) { q.polys[idx] = polys[++i]; }
                else { q.polys[idx].start = p.polys[++j].start; q.polys[idx].p = p.polys[j].p * (-1.0); }
                //		if(idx && polys[idx].start==polys[idx-1].start)	{polys[idx-1].p+=polys[idx].p;}
                //		else{idx++;}
                idx++;
            }
            return q;
        }

        /**     remain the start
          *     multiply every polynomial by p  */
        template<int Degree2>
        PPolynomial<Degree + Degree2> operator * (const Polynomial<Degree2>& p) const {
            PPolynomial<Degree + Degree2> q;
            q.set(polyCount);
            for (int i = 0; i<int(polyCount); i++) {
                q.polys[i].start = polys[i].start;
                q.polys[i].p = polys[i].p * p;
            }
            return q;
        }

        /**     for i in *this.polys
          *         for j in p.polys
          *              new.polys = i * j      */
        template<int Degree2>
        PPolynomial<Degree + Degree2> operator * (const PPolynomial<Degree2>& p) const {
            PPolynomial<Degree + Degree2> q;
            StartingPolynomial<Degree + Degree2>* sp;
            int i, j, spCount = int(polyCount * p.polyCount);

            sp = (StartingPolynomial<Degree + Degree2>*)malloc(sizeof(StartingPolynomial<Degree + Degree2>) * spCount);
            for (i = 0; i < int(polyCount); i++) {
                for (j = 0; j < int(p.polyCount); j++) {
                    sp[i * p.polyCount + j] = polys[i] * p.polys[j];
                }
            }
            q.set(sp, spCount);
            free(sp);
            return q;
        }


        PPolynomial& operator += (const float& s) { polys[0].p += s; }
        PPolynomial& operator -= (const float& s) { polys[0].p -= s; }
        PPolynomial& operator *= (const float& s) { for (int i = 0; i < int(polyCount); i++) { polys[i].p *= s; }   return *this; }
        PPolynomial& operator /= (const float& s) { for (size_t i = 0; i < polyCount; i++) { polys[i].p /= s; }     return *this; }
        PPolynomial  operator +  (const float& s) const { PPolynomial q = *this;    q += s;     return q; }
        PPolynomial  operator -  (const float& s) const { PPolynomial q = *this;    q -= s;     return q; }
        PPolynomial  operator *  (const float& s) const { PPolynomial q = *this;    q *= s;     return q; }
        PPolynomial  operator /  (const float& s) const { PPolynomial q = *this;    q /= s;     return q; }


        /**     merge the *this and scale*poly
          *     poly with the same start will be added together */
        PPolynomial& addScaled(const PPolynomial& p, const float& scale) {
            int i, j;
            StartingPolynomial<Degree>* oldPolys = polys;
            size_t idx = 0, cnt = 0, oldPolyCount = polyCount;
            polyCount = 0;
            polys = NULL;
            set(oldPolyCount + p.polyCount);
            i = j = -1;
            while (cnt < polyCount) {
                // no remain p.polys
                if (j >= int(p.polyCount) - 1) { polys[idx] = oldPolys[++i]; }
                // no remain old.polys
                else if (i >= int(oldPolyCount) - 1) { polys[idx].start = p.polys[++j].start; polys[idx].p = p.polys[j].p * scale; }
                // take poly with smaller start
                else if (oldPolys[i + 1].start < p.polys[j + 1].start) { polys[idx] = oldPolys[++i]; }
                else { polys[idx].start = p.polys[++j].start; polys[idx].p = p.polys[j].p * scale; }
                // poly with the same start will be added together
                if (idx && polys[idx].start == polys[idx - 1].start) { polys[idx - 1].p += polys[idx].p; }
                else { idx++; }
                cnt++;
            }
            free(oldPolys);
            reset(idx);
            return *this;
        }

        /**     缩放*this中的每个多项式
          *     每一个start都为start * s   */
        PPolynomial scale(const float& s) const {
            PPolynomial q;
            q.set(polyCount);
            for (size_t i = 0; i < polyCount; i++) { q.polys[i] = polys[i].scale(s); }
            return q;
        }

        /**     shift every poly in *this
          *     every start + t                 */
        PPolynomial shift(const float& t) const {
            PPolynomial q;
            q.set(polyCount);
            for (size_t i = 0; i < polyCount; i++) { q.polys[i] = polys[i].shift(t); }
            return q;
        }

        /**     polys.start remain the same
          *     polys are derived               */
        PPolynomial<Degree - 1> derivative(void) const {
            PPolynomial<Degree - 1> q;
            q.set(polyCount);
            for (size_t i = 0; i < polyCount; i++) {
                q.polys[i].start = polys[i].start;
                q.polys[i].p = polys[i].p.derivative();
            }
            return q;
        }

        /**     polys.start remain the same
          *     definite integral function
          *     polys[i].start / x [fi(t)dt]
          *     Code:
          *     q.polys[i].p=polys[i].p.integral();
          *     q.polys[i].p-=q.polys[i].p(q.polys[i].start);  */
        PPolynomial<Degree + 1> integral(void) const {
            int i;
            PPolynomial<Degree + 1> q;
            q.set(polyCount);
            for (i = 0; i<int(polyCount); i++) {
                q.polys[i].start = polys[i].start;
                q.polys[i].p = polys[i].p.integral();
                q.polys[i].p -= q.polys[i].p(q.polys[i].start);

            }
            return q;
        }

        /**     polys with $start < min are added together, get a new poly
          *     solve
          *     a0 x^0 + a1 x^1 + ... + an x^n = c
          *     save all solution accord with ( min < root < max )  */
        void getSolutions(const float& c, std::vector<float>& roots, const float& EPS, const float& min = -DBL_MAX, const float& max = DBL_MAX) const {
            Polynomial<Degree> p;
            std::vector<float> tempRoots;
            p.setZero();
            for (size_t i = 0; i < polyCount; i++) {
                p += polys[i].p;
                if (polys[i].start > max) { break; }
                if (i < polyCount - 1 && polys[i + 1].start < min) { continue; }
                p.getSolutions(c, tempRoots, EPS);
                for (size_t j = 0; j < tempRoots.size(); j++) {
                    if (tempRoots[j] > polys[i].start && (i + 1 == polyCount || tempRoots[j] <= polys[i + 1].start)) {
                        if (tempRoots[j] > min && tempRoots[j] < max) { roots.push_back(tempRoots[j]); }

                    }
                }
            }
        }

        void printnl(void) const {
            Polynomial<Degree> p;
            if (!polyCount) {
                Polynomial<Degree> p;
                printf("[-Infinity,Infinity]\n");
            }
            else {
                for (size_t i = 0; i < polyCount; i++) {
                    printf("[");
                    if (polys[i].start == DBL_MAX) { printf("Infinity,"); }
                    else if (polys[i].start == -DBL_MAX) { printf("-Infinity,"); }
                    else { printf("%f,", polys[i].start); }
                    if (i + 1 == polyCount) { printf("Infinity]\t"); }
                    else if (polys[i + 1].start == DBL_MAX) { printf("Infinity]\t"); }
                    else if (polys[i + 1].start == -DBL_MAX) { printf("-Infinity]\t"); }
                    else { printf("%f]\t", polys[i + 1].start); }
                    p = p + polys[i].p;
                    p.printnl();
                }
            }
            printf("\n");
        }

        /**
         * \brief 获得2个初始多项式【常函数】，多项式f0 = 1 x∈[-radius, +∞)，多项式f1 = -1 x∈[radius, +∞)
         * 
         * \param radius 区间半径
         * \return 初始多项式
         */
        static PPolynomial ConstantFunction(const float& radius = 0.5) {
            if (Degree < 0) {
                fprintf(stderr, "不能将阶数为 %d 的多项式设置为常数\n", Degree);
                exit(0);
            }
            PPolynomial q;
            q.set(2);   // 开辟空间

            q.polys[0].start = -radius;             // 多项式0起始坐标在-radius
            q.polys[1].start = radius;              // 多项式1起始坐标在radius

            q.polys[0].p.coefficients[0] = 1.0;     // 多项式0常数为1
            q.polys[1].p.coefficients[0] = -1.0;    // 多项式1常数为-1
            return q;
        }

        /**
         * \brief 生成高斯平滑的近似.
         * 
         * \param width 平滑的半径
         * \return 平滑的多项式
         */
        static PPolynomial GaussianApproximation(const float& width = 0.5);

        /**
         * \brief 用于计算多项式的移动平均值。它的作用是对给定半径范围内的多项式进行积分和平均，从而平滑原始多项式数据.
         * 
         * \param radius 平滑的半径
         * \return 平滑的多项式
         */
        PPolynomial<Degree + 1> MovingAverage(const float& radius) {
            PPolynomial<Degree + 1> A;              // 多项式集合对象(二次B样条函数)
            Polynomial<Degree + 1> p;               // 多项式对象
            StartingPolynomial<Degree + 1>* sps;    // 带有起始位置的多项式，主要是记录多项式函数在区间段的平移，从而求出[-radius, radius]的积分
            sps = (StartingPolynomial<Degree + 1>*)malloc(sizeof(StartingPolynomial<Degree + 1>) * polyCount * 2);
            for (int i = 0; i < int(polyCount); i++)  {
                sps[2 * i].start = polys[i].start - radius;       // 多项式定义域起点(左边界)偏移
                sps[2 * i + 1].start = polys[i].start + radius;   // 多项式定义域起点(左边界)偏移
                // p = 多项式i的原函数 - 多项式i的原函数在start处的值   -->   确保多项式原函数在start处的值为0
                p = polys[i].p.integral() - polys[i].p.integral()(polys[i].start);
                sps[2 * i].p = p.shift(-radius);                  // 多项式偏移得到新的多项式
                sps[2 * i + 1].p = p.shift(radius) * -1;          // 多项式偏移得到新的多项式
                //if (int(Degree) == 0) {
                //    printf("Degree = %d  Poly[%d] = %.3fx + %.3f\n", int(Degree), i, p.coefficients[1], p.coefficients[0]);
                //    printf("            sps [%d] = %.3fx + %.3f  x ∈ [%.3f, ∞)\n            sps [%d] = %.3fx + %.3f  x ∈ [%.3f, ∞)\n",
                //        2 * i, sps[2 * i].p.coefficients[1], sps[2 * i].p.coefficients[0], sps[2 * i].start,
                //        2 * i + 1, sps[2 * i + 1].p.coefficients[1], sps[2 * i + 1].p.coefficients[0], sps[2 * i + 1].start);
                //}
                //if (int(Degree) == 1) {
                //    printf("Degree = %d  Poly[%d] = %.3fx^2 + %.3fx + %.3f\n", int(Degree), i, p.coefficients[2], p.coefficients[1], p.coefficients[0]);
                //    printf("            sps [%d] = %.3fx^2 + %.3fx + %.3f  x ∈ [%.3f, ∞)\n            sps [%d] = %.3fx^2 + %.3fx + %.3f  x ∈ [%.3f, ∞)\n",
                //        2 * i, sps[2 * i].p.coefficients[2], sps[2 * i].p.coefficients[1], sps[2 * i].p.coefficients[0], sps[2 * i].start,
                //        2 * i + 1, sps[2 * i + 1].p.coefficients[2], sps[2 * i + 1].p.coefficients[1], sps[2 * i + 1].p.coefficients[0], sps[2 * i + 1].start);
                //}
                //else if (int(Degree) == 2) {
                //    printf("Degree = %d  Poly[%d] = %.3fx^3 + %.3fx^2 + %.3fx + %.3f\n", int(Degree), i, p.coefficients[3], p.coefficients[2], p.coefficients[1], p.coefficients[0]);
                //    printf("            sps [%d] = %.3fx^3 + %.3fx^2 + %.3fx + %.3f  x ∈ [%.3f, ∞)\n            sps [%d] = %.3fx^3 + %.3fx^2 + %.3fx + %.3f  x ∈ [%.3f, ∞)\n",
                //        2 * i, sps[2 * i].p.coefficients[3], sps[2 * i].p.coefficients[2], sps[2 * i].p.coefficients[1], sps[2 * i].p.coefficients[0], sps[2 * i].start,
                //        2 * i + 1, sps[2 * i + 1].p.coefficients[3], sps[2 * i + 1].p.coefficients[2], sps[2 * i + 1].p.coefficients[1], sps[2 * i + 1].p.coefficients[0], sps[2 * i + 1].start);
                //}
            }
            A.set(sps, int(polyCount * 2)); // 合并偏移的多项式
            free(sps);

            return A * 1.0 / (2 * radius);  // 多项式在[-radius, radius]区间内求平均面积，本身求的也是多项式在区间内的平均值
        }

        void write(FILE* fp, const int& samples, const float& min, const float& max) const {
            fwrite(&samples, sizeof(int), 1, fp);
            for (int i = 0; i < samples; i++) {
                float x = min + i * (max - min) / (samples - 1);
                float v = (*this)(x);
                fwrite(&v, sizeof(float), 1, fp);
            }
        }
	};

    /**
     * \brief 【通用版本的 GaussianApproximation 函数实现】用在某个区间上的平均值近似的作为高斯分布.
     * 
     * \param radius 区间的宽度/2
     * \return (近似)高斯平滑后的多项式
     */
    template<int Degree>
    PPolynomial<Degree> PPolynomial<Degree>::GaussianApproximation(const float& radius) {
        return PPolynomial<Degree - 1>::GaussianApproximation().MovingAverage(radius);
    }

    /**
     * \brief 【特化版本的 GaussianApproximation 函数实现】迭代的最后一层，初始多项式【常数多项式】.
     * 
     * \param radius 区间的宽度/2
     * \return 初始多项式【常数多项式】
     */
    template<>
    inline PPolynomial<0> PPolynomial<0>::GaussianApproximation(const float& radius) {
        return ConstantFunction(radius);
    }


    template<int Degree>
    __host__ void copySinglePPolynomialHostToDevice(PPolynomial<Degree>* pp_h, PPolynomial<Degree>*& pp_d) {
        cudaMalloc((PPolynomial<Degree> **) & pp_d, sizeof(PPolynomial<Degree>));
        StartingPolynomial<Degree>* AddrPPolynomialDevice = NULL;
        StartingPolynomial<Degree>* AddrPPolynomialHost = pp_h->polys;
        int nByte = sizeof(StartingPolynomial<Degree>) * pp_h->polyCount;
        cudaMalloc((StartingPolynomial<Degree> **) & AddrPPolynomialDevice, nByte);
        cudaMemcpy(AddrPPolynomialDevice, pp_h->polys, nByte, cudaMemcpyHostToDevice);
        pp_h->polys = AddrPPolynomialDevice;

        cudaMemcpy(pp_d, pp_h, sizeof(PPolynomial<Degree>), cudaMemcpyHostToDevice);
        pp_h->polys = AddrPPolynomialHost;

    }

    template<int Degree>
    __host__ void copyWholePPolynomialHostToDevice(PPolynomial<Degree>* pp_h, PPolynomial<Degree>*& pp_d, int size) {
        cudaMalloc((PPolynomial<Degree> **) & pp_d, sizeof(PPolynomial<Degree>) * size);
        std::vector<StartingPolynomial<Degree>*> host_pointer_v;
        for (int i = 0; i < size; ++i) {
            StartingPolynomial<Degree>* d_addr = NULL;
            int nByte = sizeof(StartingPolynomial<Degree>) * pp_h[i].polyCount;
            host_pointer_v.push_back(pp_h[i].polys);

            cudaMalloc((StartingPolynomial<Degree> **) & d_addr, nByte);
            cudaMemcpy(d_addr, pp_h[i].polys, nByte, cudaMemcpyHostToDevice);
            pp_h[i].polys = d_addr;
        }
        cudaMemcpy(pp_d, pp_h, sizeof(PPolynomial<Degree>) * size, cudaMemcpyHostToDevice);

        for (int i = 0; i < size; ++i) {
            pp_h[i].polys = host_pointer_v[i];
        }
    }


    template<int Degree>
    __host__ __device__ void scale(PPolynomial<Degree>* pp, const float& scale) {
        for (int i = 0; i < pp->polyCount; ++i) {
            pp->polys[i].start *= scale;
            float s2 = 1.0;
            for (int j = 0; j <= Degree; ++j) {
                pp->polys[i].p.coefficients[j] *= s2;
                s2 /= scale;
            }
        }
    }
}
/*****************************************************************//**
 * \file   PPolynomial.h
 * \brief  ����ʽ���
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
     * \brief �������Ķ���ʽ��start���Ǹö���ʽ���������߽磬��p(x) = f(x)  x��[start, +��).
     */
    template<int Degree>
    class StartingPolynomial
    {
    public:
        Polynomial<Degree> p;   // ����ʽ
        float start;            // ����ʽ�����

        /**	    return a StartingPolynomial
          *	    new start is the bigger start
          *	    polynomials are multiplied  */
        /**
         * \brief ����ʽ�˷��������ʽΪ��������ʽ�ĳ˻�����ʼ��Ϊ������ʼ���нϴ��һ��.
         * 
         * \param p ����ʽ
         * \return ��˺�Ľ��
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
         * \brief ���ض����˸�ֵ������ʹ�ÿ��Խ�һ�� StartingPolynomial �����ֵ������һ������.
         * 
         * \param sp ��ֵ��
         * \return ����ֵ����
         */
        __host__ __device__ StartingPolynomial& operator = (const StartingPolynomial& sp) {
            start = sp.start;
            for (int i = 0; i <= Degree; ++i)
                p.coefficients[i] = sp.p.coefficients[i];
            return *this;
        }

        /**
         * \brief ��ʼ��Ͷ���ʽ�������� s ����.
         * 
         * \param s ���ű���
         * \return ���ź�Ķ���ʽ
         */
        __host__ __device__ StartingPolynomial scale(const float& s) const {
            StartingPolynomial q;
            q.start = start * s;
            q.p = p.scale(s);
            return q;
        }

        /**
         * \brief ���ƽ�ơ�start = start + t��������ʽҲƽ�ơ�f(x) -> f(x-t)��.
         * 
         * \param t ƽ����
         * \return ƽ�ƺ�Ķ���ʽ
         */
        __host__ __device__ StartingPolynomial shift(const float& t) const {
            StartingPolynomial q;
            q.start = start + t;
            q.p = p.shift(t);
            return q;
        }

        /**
         * \brief �Ƚ��ĸ�����ʽ������С�������ǰ�������ʼ��С�� sp ����ʼ�㣬�򷵻� 1�����򷵻� 0.
         * 
         * \param sp �Ƚ϶���
         * \return �Ƿ��С
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
         * \brief �Ƚ϶���ʽ������С��v1 > v2 , return 1��v1 < v2 , return -1��v1 = v2 , return 0.
         * 
         * \param v1 ����ʽv1
         * \param v2 ����ʽv2
         * \return ����ʽ�ȽϵĽ��
         */
        __host__ __device__ static int Compare(const void* v1, const void* v2) {
            float d = ((StartingPolynomial*)(v1))->start - ((StartingPolynomial*)(v2))->start;
            if (d < 0) { return -1; }
            else if (d > 0) { return  1; }
            else { return  0; }
        }
    };

    /**
     * \brief ����ʽ����(����B���������Ļ���������).
     */
    template<int Degree>
	class PPolynomial
	{
	public:
        size_t polyCount = 0;                       // ����ʽ����Ԫ�ص�����
        StartingPolynomial<Degree>* polys = NULL;   // �����(�ж������)�Ķ���ʽ(��)

        PPolynomial(void) = default;
        /**
         * \brief ������ʽp������ֵ����ǰ����.
         * 
         * \param ����ʽp 
         */
        PPolynomial(const PPolynomial<Degree>& p) {
            set(p.polyCount);   // ���仺��
            memcpy(polys, p.polys, sizeof(StartingPolynomial<Degree>) * p.polyCount);
        }
        ~PPolynomial(void) {
            if (polyCount) { free(polys); }
            polyCount = 0;
            polys = NULL;
        }

        /**
         * \brief ��õ�ǰ����ʽ���ϵ�byte��С.
         * 
         * \return ���ص�ǰ����ʽ���ϵ�byte��С
         */
        int size(void) const {
            return int(sizeof(StartingPolynomial<Degree>) * polyCount);
        }

        /**
         * \param �ͷ�֮ǰ�Ķ���ʽ�������·��䵱ǰ����ʽ���ڴ�.
         * 
         * \param size ����ʽ������
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
         * \brief ������� StartingPolynomial ������������������򣬲���������ͬ���Ķ���ʽ�ϲ���һ��.
         * 
         * \param sps ������ʽ
         * \param count ����ʽ����
         */
        void set(StartingPolynomial<Degree>* sps, const int& count) {
            int i = 0;  // ������ʽ�����index
            int c = 0;  // ��¼��ͬ���Ķ���ʽ��ƫ��
            set(count);
            // ����StartingPolynomial<Degree>::Compare�ķ�ʽ�������򣺼�����start�Ĵ�С
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
         * \brief ���·�����չ����ʽ���ڴ��С�����оɶ���ʽ�����ݻᱻ����.
         * 
         * \param newSize �¶���ʽ�Ĵ�С
         */
        void reset(const size_t& newSize) {
            polyCount = newSize;
            polys = (StartingPolynomial<Degree>*)realloc(polys, sizeof(StartingPolynomial<Degree>) * newSize);
        }


        /**     assume that StartPolynomial is sorted by set() function
          *     calculate f0(t) + f1(t) + f2(t) + ... + fn(t)
          *     StartPolynomial n+1's start >= t    */
        /**
         * \brief ��������ʼ����ʽ���ϱ�set�������򡿻�ö���ʽ����������start < t�Ķ���ʽ����Щ����ʽ��t�㴦����ֵ��(f0(t) + f1(t) + f2(t) + ... + fn(t)).
         * 
         * \param t �Ա���t����ֵt
         * \return ���ض���ʽ����������start < t�Ķ���ʽ�ĺ�
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
         * \brief ��� integral �������ڼ����ڸ������� [tMin,tMax] �ڵĶ�� StartingPolynomial ����Ķ����ֵ��ܺ͡�
         * 
         * \param tMin ������������
         * \param tMax ����������յ�
         * \return ���ֵĽ��
         */
        float integral(const float& tMin, const float& tMax) const {
            int m = 1;  // �����ַ���
            float start, end, s, v = 0;
            start = tMin;
            end = tMax;
            if (tMin > tMax) {  // ���������½�ȷ������
                m = -1;
                start = tMax;
                end = tMin;
            }
            for (int i = 0; i < int(polyCount) && polys[i].start < end; i++) {
                if (start < polys[i].start) { s = polys[i].start; } // ��δ�����½磬������ʽ����㸳ֵ��s
                else { s = start; }
                v += polys[i].p.integral(s, end);   // �ڶ�����Χ�ڵĻ���
            }
            return v * m;
        }


        /**
         * \brief ����Ӧ���Ǽ��㺯������������Ԫ�طǹ�������Ļ��֡�ǰ���Ǻ����Ѿ�����.
         * 
         * \return ��������������Ԫ�طǹ�������Ļ��֡�ǰ���Ǻ����Ѿ�����
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

        /**     ����*this�е�ÿ������ʽ
          *     ÿһ��start��Ϊstart * s   */
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
         * \brief ���2����ʼ����ʽ����������������ʽf0 = 1 x��[-radius, +��)������ʽf1 = -1 x��[radius, +��)
         * 
         * \param radius ����뾶
         * \return ��ʼ����ʽ
         */
        static PPolynomial ConstantFunction(const float& radius = 0.5) {
            if (Degree < 0) {
                fprintf(stderr, "���ܽ�����Ϊ %d �Ķ���ʽ����Ϊ����\n", Degree);
                exit(0);
            }
            PPolynomial q;
            q.set(2);   // ���ٿռ�

            q.polys[0].start = -radius;             // ����ʽ0��ʼ������-radius
            q.polys[1].start = radius;              // ����ʽ1��ʼ������radius

            q.polys[0].p.coefficients[0] = 1.0;     // ����ʽ0����Ϊ1
            q.polys[1].p.coefficients[0] = -1.0;    // ����ʽ1����Ϊ-1
            return q;
        }

        /**
         * \brief ���ɸ�˹ƽ���Ľ���.
         * 
         * \param width ƽ���İ뾶
         * \return ƽ���Ķ���ʽ
         */
        static PPolynomial GaussianApproximation(const float& width = 0.5);

        /**
         * \brief ���ڼ������ʽ���ƶ�ƽ��ֵ�����������ǶԸ����뾶��Χ�ڵĶ���ʽ���л��ֺ�ƽ�����Ӷ�ƽ��ԭʼ����ʽ����.
         * 
         * \param radius ƽ���İ뾶
         * \return ƽ���Ķ���ʽ
         */
        PPolynomial<Degree + 1> MovingAverage(const float& radius) {
            PPolynomial<Degree + 1> A;              // ����ʽ���϶���(����B��������)
            Polynomial<Degree + 1> p;               // ����ʽ����
            StartingPolynomial<Degree + 1>* sps;    // ������ʼλ�õĶ���ʽ����Ҫ�Ǽ�¼����ʽ����������ε�ƽ�ƣ��Ӷ����[-radius, radius]�Ļ���
            sps = (StartingPolynomial<Degree + 1>*)malloc(sizeof(StartingPolynomial<Degree + 1>) * polyCount * 2);
            for (int i = 0; i < int(polyCount); i++)  {
                sps[2 * i].start = polys[i].start - radius;       // ����ʽ���������(��߽�)ƫ��
                sps[2 * i + 1].start = polys[i].start + radius;   // ����ʽ���������(��߽�)ƫ��
                // p = ����ʽi��ԭ���� - ����ʽi��ԭ������start����ֵ   -->   ȷ������ʽԭ������start����ֵΪ0
                p = polys[i].p.integral() - polys[i].p.integral()(polys[i].start);
                sps[2 * i].p = p.shift(-radius);                  // ����ʽƫ�Ƶõ��µĶ���ʽ
                sps[2 * i + 1].p = p.shift(radius) * -1;          // ����ʽƫ�Ƶõ��µĶ���ʽ
                //if (int(Degree) == 0) {
                //    printf("Degree = %d  Poly[%d] = %.3fx + %.3f\n", int(Degree), i, p.coefficients[1], p.coefficients[0]);
                //    printf("            sps [%d] = %.3fx + %.3f  x �� [%.3f, ��)\n            sps [%d] = %.3fx + %.3f  x �� [%.3f, ��)\n",
                //        2 * i, sps[2 * i].p.coefficients[1], sps[2 * i].p.coefficients[0], sps[2 * i].start,
                //        2 * i + 1, sps[2 * i + 1].p.coefficients[1], sps[2 * i + 1].p.coefficients[0], sps[2 * i + 1].start);
                //}
                //if (int(Degree) == 1) {
                //    printf("Degree = %d  Poly[%d] = %.3fx^2 + %.3fx + %.3f\n", int(Degree), i, p.coefficients[2], p.coefficients[1], p.coefficients[0]);
                //    printf("            sps [%d] = %.3fx^2 + %.3fx + %.3f  x �� [%.3f, ��)\n            sps [%d] = %.3fx^2 + %.3fx + %.3f  x �� [%.3f, ��)\n",
                //        2 * i, sps[2 * i].p.coefficients[2], sps[2 * i].p.coefficients[1], sps[2 * i].p.coefficients[0], sps[2 * i].start,
                //        2 * i + 1, sps[2 * i + 1].p.coefficients[2], sps[2 * i + 1].p.coefficients[1], sps[2 * i + 1].p.coefficients[0], sps[2 * i + 1].start);
                //}
                //else if (int(Degree) == 2) {
                //    printf("Degree = %d  Poly[%d] = %.3fx^3 + %.3fx^2 + %.3fx + %.3f\n", int(Degree), i, p.coefficients[3], p.coefficients[2], p.coefficients[1], p.coefficients[0]);
                //    printf("            sps [%d] = %.3fx^3 + %.3fx^2 + %.3fx + %.3f  x �� [%.3f, ��)\n            sps [%d] = %.3fx^3 + %.3fx^2 + %.3fx + %.3f  x �� [%.3f, ��)\n",
                //        2 * i, sps[2 * i].p.coefficients[3], sps[2 * i].p.coefficients[2], sps[2 * i].p.coefficients[1], sps[2 * i].p.coefficients[0], sps[2 * i].start,
                //        2 * i + 1, sps[2 * i + 1].p.coefficients[3], sps[2 * i + 1].p.coefficients[2], sps[2 * i + 1].p.coefficients[1], sps[2 * i + 1].p.coefficients[0], sps[2 * i + 1].start);
                //}
            }
            A.set(sps, int(polyCount * 2)); // �ϲ�ƫ�ƵĶ���ʽ
            free(sps);

            return A * 1.0 / (2 * radius);  // ����ʽ��[-radius, radius]��������ƽ��������������Ҳ�Ƕ���ʽ�������ڵ�ƽ��ֵ
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
     * \brief ��ͨ�ð汾�� GaussianApproximation ����ʵ�֡�����ĳ�������ϵ�ƽ��ֵ���Ƶ���Ϊ��˹�ֲ�.
     * 
     * \param radius ����Ŀ��/2
     * \return (����)��˹ƽ����Ķ���ʽ
     */
    template<int Degree>
    PPolynomial<Degree> PPolynomial<Degree>::GaussianApproximation(const float& radius) {
        return PPolynomial<Degree - 1>::GaussianApproximation().MovingAverage(radius);
    }

    /**
     * \brief ���ػ��汾�� GaussianApproximation ����ʵ�֡����������һ�㣬��ʼ����ʽ����������ʽ��.
     * 
     * \param radius ����Ŀ��/2
     * \return ��ʼ����ʽ����������ʽ��
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
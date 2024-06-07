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

    template<int Degree>
    class StartingPolynomial
    {
    public:
        Polynomial<Degree> p;
        float start;

        /**	    return a StartingPolynomial
          *	    new start is the bigger start
          *	    polynomials are multiplied  */
        template<int Degree2>
        __host__ __device__ StartingPolynomial<Degree + Degree2>  operator * (const StartingPolynomial<Degree2>& p) const {
            StartingPolynomial<Degree + Degree2> sp;
            if (start > p.start) { sp.start = start; }
            else { sp.start = p.start; }
            sp.p = this->p * p.p;
            return sp;
        }

        __host__ __device__ StartingPolynomial& operator = (const StartingPolynomial& sp) {
            start = sp.start;
            for (int i = 0; i <= Degree; ++i)
                p.coefficients[i] = sp.p.coefficients[i];
            return *this;
        }
        /**     start = start * s
          *     polynomial is scaled by s   */
        __host__ __device__ StartingPolynomial scale(const float& s) const {
            StartingPolynomial q;
            q.start = start * s;
            q.p = p.scale(s);
            return q;
        }

        /**     start = start + t
          *     polynomial f(x) -> f(x-t)   */
        __host__ __device__ StartingPolynomial shift(const float& t) const {
            StartingPolynomial q;
            q.start = start + t;
            q.p = p.shift(t);
            return q;
        }

        /**     big start is bigger */
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
        __host__ __device__ static int Compare(const void* v1, const void* v2) {
            float d = ((StartingPolynomial*)(v1))->start - ((StartingPolynomial*)(v2))->start;
            if (d < 0) { return -1; }
            else if (d > 0) { return  1; }
            else { return  0; }
        }
    };

    template<int Degree>
	class PPolynomial
	{
	public:
        size_t polyCount = 0;
        StartingPolynomial<Degree>* polys = NULL;

        PPolynomial(void) = default;
        PPolynomial(const PPolynomial<Degree>& p) {
            set(p.polyCount);
            memcpy(polys, p.polys, sizeof(StartingPolynomial<Degree>) * p.polyCount);
        }
        ~PPolynomial(void) {
            if (polyCount) { free(polys); }
            polyCount = 0;
            polys = NULL;
        }


        /**     return size of polys    */
        int size(void) const {
            return int(sizeof(StartingPolynomial<Degree>) * polyCount);
        }

        /**     polyCount = size, polys is allocated    */
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

        /**     Note: this method will sort the elements in sps
          *     polys with the same start will be added together    */
        void set(StartingPolynomial<Degree>* sps, const int& count) {
            int i, c = 0;
            set(count);
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

        /**     realloc the memory to expand the polys pointer memory
          *     the old content will be remained    */
        void reset(const size_t& newSize) {
            polyCount = newSize;
            polys = (StartingPolynomial<Degree>*)realloc(polys, sizeof(StartingPolynomial<Degree>) * newSize);
        }


        /**     assume that StartPolynomial is sorted by set() function
          *     calculate f0(t) + f1(t) + f2(t) + ... + fn(t)
          *     StartPolynomial n+1's start >= t    */
        float operator()(const float& t) const {
            float v = 0;
            for (int i = 0; i<int(polyCount) && t>polys[i].start; i++) { v += polys[i].p(t); }
            return v;
        }


        /**     calculate the definite integral, integral start from the p[i].start, not min(tMin, tMax)
          *     let end = max(tMin, tMax)
          *     p[0].start / end [f0(x)dx] + p[1].start / end [f1(x)dx] + ... + p[n].start / end [fn(x)dx]
          *     p[n+1].start >= min(tMin, tMax)
          *     tMin can be bigger than tMax    */
        float integral(const float& tMin, const float& tMax) const {
            int m = 1;
            float start, end, s, v = 0;
            start = tMin;
            end = tMax;
            if (tMin > tMax) {
                m = -1;
                start = tMax;
                end = tMin;
            }
            for (int i = 0; i<int(polyCount) && polys[i].start < end; i++) {
                if (start < polys[i].start) { s = polys[i].start; }
                else { s = start; }
                v += polys[i].p.integral(s, end);
            }
            return v * m;
        }


        /**     integral(polys[0].start,polys[polyCount-1].start)   */
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

        static PPolynomial ConstantFunction(const float& radius = 0.5) {
            if (Degree < 0) {
                fprintf(stderr, "不能将阶数为 %d 的多项式设置为常数\n", Degree);
                exit(0);
            }
            PPolynomial q;
            q.set(2);

            q.polys[0].start = -radius;
            q.polys[1].start = radius;

            q.polys[0].p.coefficients[0] = 1.0;
            q.polys[1].p.coefficients[0] = -1.0;
            return q;
        }

        /**     use to generate approximation to Gaussian filter    */
        static PPolynomial GaussianApproximation(const float& width = 0.5);

        PPolynomial<Degree + 1> MovingAverage(const float& radius) {
            PPolynomial<Degree + 1> A;
            Polynomial<Degree + 1> p;
            StartingPolynomial<Degree + 1>* sps;
            sps = (StartingPolynomial<Degree + 1>*)malloc(sizeof(StartingPolynomial<Degree + 1>) * polyCount * 2);
            for (int i = 0; i<int(polyCount); i++) {
                sps[2 * i].start = polys[i].start - radius;
                sps[2 * i + 1].start = polys[i].start + radius;
                p = polys[i].p.integral() - polys[i].p.integral()(polys[i].start);
                sps[2 * i].p = p.shift(-radius);
                sps[2 * i + 1].p = p.shift(radius) * -1;
            }
            A.set(sps, int(polyCount * 2));
            free(sps);
            return A * 1.0 / (2 * radius);
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
    // 通用版本的 GaussianApproximation 函数实现
    template<int Degree>
    PPolynomial<Degree> PPolynomial<Degree>::GaussianApproximation(const float& width) {
        return PPolynomial<Degree - 1>::GaussianApproximation().MovingAverage(width);
    }

    // 特化版本的 GaussianApproximation 函数实现
    template<>
    inline PPolynomial<0> PPolynomial<0>::GaussianApproximation(const float& width) {
        return ConstantFunction(width);
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
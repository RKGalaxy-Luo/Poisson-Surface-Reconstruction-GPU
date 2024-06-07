#pragma once
#include <vector>
#include <device_launch_parameters.h>
#include <cuda_runtime_api.h>
#include <math/Factor.h>

namespace SparseSurfelFusion {

	template<int Degree>
	class Polynomial
	{
    public:
        float coefficients[Degree + 1] = { 0.0f }; // ����ʽ��ϵ��

        /**
         * \brief ���캯��,��ʼ������ϵ��Ϊ0.
         * 
         * \return 
         */
        __host__ __device__ Polynomial(void) = default;
        /**
         * \brief ���캯������P��ֵ��ǰ����ʽ������ʽ������С��Ϊ׼.
         *
         * \param P ����ʽ
         */
        template<int Degree2>
        Polynomial(const Polynomial<Degree2>& P) {
            for (int i = 0; i <= Degree && i <= Degree2; i++) {
                coefficients[i] = P.coefficients[i];
            }
        }

        /**
         * \brief ����(),����f(t)��ֵ.
         * 
         * \param t �Ա���
         * \return ����ֵ
         */
        __host__ __device__ float operator()(const float& t) const {
            float temp = 1;
            float v = 0;
            for (int i = 0; i <= Degree; i++) {
                v += temp * coefficients[i];
                temp *= t;
            }
            return v;
        }

        /**
         * \brief �����������[tmin, tmax]�ϵĶ�����, ���� tmin �� tmax Ҳ�ǿ��Ե�.
         * 
         * \param tMin ������߽�
         * \param tMax �����ұ߽�
         * \return ���������ֺ��ֵ
         */
        float integral(const float& tMin, const float& tMax) const {
            float v = 0;
            float t1, t2;
            t1 = tMin;
            t2 = tMax;
            for (int i = 0; i <= Degree; i++) {
                v += coefficients[i] * (t2 - t1) / (i + 1);
                if (t1 != -DBL_MAX && t1 != DBL_MAX) { t1 *= tMin; }
                if (t2 != -DBL_MAX && t2 != DBL_MAX) { t2 *= tMax; }
            }
            return v;
        }

        /**
         * \brief ���� "==" ,�ж϶���ʽ�Ƿ����.
         * 
         * \param p ��֮�ȽϵĶ���ʽ
         * \return ��ȷ���true, ���ȷ���false
         */
        bool operator == (const Polynomial& p) const {
            for (int i = 0; i <= Degree; i++) {
                if (coefficients[i] != p.coefficients[i]) {
                    return false;
                }
            }
            return true;
        }

        /**
         * \brief ���� "!=" ,�ж϶���ʽ�Ƿ����.
         * 
         * \param p ��֮�ȽϵĶ���ʽ
         * \return ���ȷ���true, ��ȷ���false
         */
        bool operator != (const Polynomial& p) const {
            for (int i = 0; i <= Degree; i++) {
                if (coefficients[i] == p.coefficients[i]) {
                    return false;
                }
            }
            return true;
        }

        /**
         * \brief �ж϶���ʽ�Ƿ�Ϊ0.
         * 
         * \return Ϊ0����true����Ϊ0����false
         */
        bool isZero(void) const {
            for (int i = 0; i <= Degree; i++) {
                if (coefficients[i] != 0) {
                    return false;
                }
            }
            return true;
        }

        /**
         * \brief ������ʽ����Ϊ0.
         * 
         */
        void setZero(void) {
            memset(coefficients, 0, sizeof(float) * (Degree + 1));
        }

        /**
         * \brief ���ظ�ֵ�� " = ", ��������ʽ��С�õ�degree��Ϊ��׼��degree����p��ֵ��ֵ����ǰ����ʽ.
         * 
         * \param p ����ʽp������ΪDegree2
         * \return ����ʽ
         */
        template<int Degree2>
        Polynomial& operator  = (const Polynomial<Degree2>& p) {
            int d = Degree < Degree2 ? Degree : Degree2;    // ѡ��degreeС��
            memset(coefficients, 0, sizeof(float) * (Degree + 1));
            memcpy(coefficients, p.coefficients, sizeof(float) * (d + 1));
            return *this;
        }
        /**
         * \brief �����Լӷ� " += ", ����ʽ�Լ�.
         * 
         * \param p ��Ҫ��ӵĶ���ʽ
         * \return �ԼӺ�Ľ��
         */
        Polynomial& operator += (const Polynomial& p) {
            for (int i = 0; i <= Degree; i++) { 
                coefficients[i] += p.coefficients[i]; 
            }
            return *this;
        }
        /**
         * \brief �����Լ��� " += ", ����ʽ�Լ�.
         * 
         * \param p ��Ҫ���Ķ���ʽ
         * \return �Լ���Ľ��
         */
        Polynomial& operator -= (const Polynomial& p) {
            for (int i = 0; i <= Degree; i++) {
                coefficients[i] -= p.coefficients[i]; 
            }
            return *this;
        }
        /**
         * \brief ����ʽȡ��.
         * 
         * \return ȡ����Ľ��
         */
        Polynomial  operator -  (void) const {
            Polynomial q = *this;
            for (int i = 0; i <= Degree; i++) { 
                q.coefficients[i] = -q.coefficients[i];
            }
            return q;
        }
        /**
         * \brief ����ʽ���.
         * 
         * \param p ��Ҫ��ӵĶ���ʽ
         * \return ��ӵĽ��
         */
        Polynomial  operator +  (const Polynomial& p) const {
            Polynomial q;
            for (int i = 0; i <= Degree; i++) {
                q.coefficients[i] = (coefficients[i] + p.coefficients[i]); 
            }
            return q;
        }
        /**
         * \brief ����ʽ���.
         * 
         * \param p ��Ҫ����Ķ���ʽ������
         * \return ����Ľ��
         */
        Polynomial  operator -  (const Polynomial& p) const {

        }

        /**
         * \brief ����ʽ�˷�.
         * 
         * \param p ��Ҫ��˵Ķ���ʽ
         */
        template<int Degree2>
        __host__ __device__ Polynomial<Degree + Degree2>  operator *  (const Polynomial<Degree2>& p) const {
            Polynomial<Degree + Degree2> q;
            for (int i = 0; i <= Degree; i++) { 
                for (int j = 0; j <= Degree2; j++) { 
                    q.coefficients[i + j] += coefficients[i] * p.coefficients[j];
                }
            }
            return q;
        }
        /**
         * \brief ���� " += " ����ʽ�������Լӣ�����ϵ�� + s.
         * 
         * \param s ����
         * \return ��������Ӻ�Ķ���ʽ
         */
        Polynomial& operator += (const float& s) {
            coefficients[0] += s;
            return *this;
        }
        /**
         * \brief ���� " -= " ����ʽ�������Լ�������ϵ�� - s.
         * 
         * \param s ����
         * \return �����������Ķ���ʽ
         */
        Polynomial& operator -= (const float& s) {
            coefficients[0] -= s;
            return *this;
        }
        /**
         * \brief ���� " *= ", ����ʽÿ����ϵ���Գ�s.
         * 
         * \param s ����
         * \return ����ϵ���Գ�s��Ķ���ʽ
         */
        Polynomial& operator *= (const float& s) {
            for (int i = 0; i <= Degree; i++) {
                coefficients[i] *= s; 
            }
            return *this;
        }
        /**
         * \brief ���� " /= ", ����ʽÿ����ϵ���Գ�s.
         *
         * \param s ����
         * \return ����ϵ���Գ�s��Ķ���ʽ
         */
        Polynomial& operator /= (const float& s) {
            for (int i = 0; i <= Degree; i++) {
                coefficients[i] /= s; 
            }
            return *this;
        }
        /**
         * \brief ���� " + ", ����ʽ������ + ����s.
         * 
         * \param s ����s
         * \return ������Ӻ�Ķ���ʽ
         */
        Polynomial  operator +  (const float& s) const {
            Polynomial<Degree> q = *this;
            q.coefficients[0] += s;
            return q;
        }
        /**
         * \brief ���� " - ", ����ʽ������ - ����s.
         *
         * \param s ����s
         * \return ���������Ķ���ʽ
         */
        Polynomial  operator -  (const float& s) const {
            Polynomial q = *this;
            q.coefficients[0] -= s;
            return q;
        }
        /**
         * \brief ���� " * ", ����ʽ����ϵ�� * ����s������ϵ����.
         *
         * \param s ����s
         * \return ����ʽ����ϵ�� �� ����s��Ķ���ʽ
         */
        Polynomial  operator *  (const float& s) const {
            Polynomial q;
            for (int i = 0; i <= Degree; i++) { 
                q.coefficients[i] = coefficients[i] * s; 
            }
            return q;
        }
        /**
         * \brief ���� " / ", ����ʽ����ϵ�� / ����s������ϵ����.
         *
         * \param s ����s
         * \return ����ʽ����ϵ�� �� ����s��Ķ���ʽ
         */
        Polynomial  operator /  (const float& s) const {
            Polynomial q(Degree);
            for (int i = 0; i <= Degree; i++) { 
                q.coefficients[i] = coefficients[i] / s;
            }
            return q;
        }

        /**
         * \brief �����±任��ԭʽ(100.0000 * x^0 + 100.0000 * x^1 + 100.0000 * x^2).
         *                  scale(10.0)  =>  100.0000 x^0 +10.0000 x^1 +1.0000 x^2  (ԭʼʽ��ֵ���ı�)
         * \param s ����ϵ��
         * \return ���ź��ֵ
         */
        __host__ __device__ Polynomial scale(const float& s) const {
            Polynomial q = *this;
            float s2 = 1.0;
            for (int i = 0; i <= Degree; i++) {
                q.coefficients[i] *= s2;
                s2 /= s;
            }
            return q;
        }
        /**
         * \brief �����±任��f(x) -> f(x-t)��, ����ʽx����ƽ��t.
         *        ԭʽ��f(x) = 1.0000 * x^0 + 1.0000 * x^1 + 1.0000 * x^2 + 1.0000 * x^3 ( t = 1ʱ )
          *       ����� 1
          *       һ���-1   +   1x
          *       ����� 1   -   2x   +   1x^2
          *       �����-1   +   3x   -   3x^2    +   1x^3
          *       �����f(x-1) = 2 * x^1 - 2 * x^2 + x^3
         * \param t ת���ĳ���
         * \return ����ʽ f(x - t)
         */
        __host__ __device__ Polynomial shift(const float& t) const {
            Polynomial<Degree> q;
            for (int i = 0; i <= Degree; i++) {
                float temp = 1;
                for (int j = i; j >= 0; j--) {
                    q.coefficients[j] += coefficients[i] * temp;
                    temp *= -t * j;
                    temp /= (i - j + 1);
                }
            }
            return q;
        }

        /**
         * \brief ������ʽ�ĵ�����f(x) -> f'(x)��.
         *        ԭʽ��f(x) = 1.0000 * x^0 + 1.0000 * x^1 + 1.0000 * x^2 + 1.0000 * x^3
         *        �����f'(x) = 1.0000 * x^0 + 2.0000 * x^1 +3.0000 * x^2
         * 
         * \return ��������ʽ
         */
        Polynomial<Degree - 1> derivative(void) const {
            Polynomial<Degree - 1> p;
            for (int i = 0; i < Degree; i++) {
                p.coefficients[i] = coefficients[i + 1] * (i + 1);
            }
            return p;
        }

        /**
         * \brief �������ʽ�Ļ���ԭ������f(x) -> F(x), ����C = 0��.
         *        ԭʽ��f(x) =  1.0000 * x^0 + 2.0000 * x^1 + 3.0000 * x^2 + 4.0000 * x^3
         *        �����F(x) = C * x^0 + 1.0000 * x^1 + 1.0000 * x^2 + 1.0000 * x^3 + 1.0000 * x^4 (C = 0)
         * 
         * \return ԭ��������ʽ
         */
        Polynomial<Degree + 1> integral(void) const {
            Polynomial<Degree + 1> p;
            p.coefficients[0] = 0;
            for (int i = 0; i <= Degree; i++) {
                p.coefficients[i + 1] = coefficients[i] / (i + 1);
            }
            return p;
        }

        /**
         * \brief �������ʽ.
         * 
         */
        void printnl(void) const {
            for (int j = 0; j <= Degree; j++) {
                printf("%6.4f �� x^%d ", coefficients[j], j);
                if (j < Degree && coefficients[j + 1] >= 0) {
                    printf(" + ");
                }
            }
            printf("\n");
        }

        /**
         * \brief ����ǰ�Ķ���ʽ *this ����һ������ʽ p ���ո����ı���������ӣ���������洢�ڵ�ǰ�Ķ���ʽ�У�
         *        ԭʽ��1.000 * x^0 + 1.000 * x^1 + 1.000 * x^2 + 1.000 * x^3  ���� a->addScaled(*a, 10)�������£�
         *        (1.000 * x^0 + 1.000 * x^1 + 1.000 * x^2 + 1.000 * x^3) + 10 * (1.000 * x^0 + 1.000 * x^1 + 1.000 * x^2 + 1.000 * x^3).
         *        ���Ϊ��11.000 * x^0 + 11.000 * x^1 + 11.000 * x^2 + 11.000 * x^3
         * \param p ����ʽp
         * \param scale ��ӱ���
         * \return ������Ӻ�Ľ��
         */
        Polynomial& addScaled(const Polynomial& p, const float& scale) {
            for (int i = 0; i <= Degree; i++) {
                coefficients[i] += p.coefficients[i] * scale;
            }
            return *this;
        }

        /**
         * \brief ��һ������ʽȡ������������洢����һ������ʽ�У������������ʽ in �ĸ�ֵ.
         * 
         * \param in �������ʽ
         * \param out �������ʽȡ���Ľ��
         */
        static void Negate(const Polynomial& in, Polynomial& out) {
            out = in;
            for (int i = 0; i <= Degree; i++) {
                out.coefficients[i] = -out.coefficients[i];
            }
        }

        /**
         * \brief ���� p1 - p2����������洢����һ������ʽq�С�Polynomial<Degree>::Subtract ֻ�ῼ��[x^0, x^1, ..., x^Degree]��Ķ���ʽp1 �� p2��.
         * 
         * \param p1 ����ʽp1������������׵Ķ���ʽ
         * \param p2 ����ʽp2������������׵Ķ���ʽ
         * \param q ����Ľ����������һ�� Degree �׵Ķ���ʽ
         */
        static void Subtract(const Polynomial& p1, const Polynomial& p2, Polynomial& q) {
            for (int i = 0; i <= Degree; i++) {
                q.coefficients[i] = p1.coefficients[i] - p2.coefficients[i];
            }
        }

        /**
         * \brief ��һ������ʽ���ո����ı����������ţ���������洢����һ������ʽq�С�Polynomial<Degree>::Scale ֻ�ῼ��[x^0, x^1, ..., x^Degree]��Ķ���ʽ p��.
         * 
         * \param p ����ʽp������������׵Ķ���ʽ
         * \param w ���ű���
         * \param q ���Ž����������һ��Degree�׵Ķ���ʽ
         */
        static void Scale(const Polynomial& p, const float& w, Polynomial& q) {
            for (int i = 0; i <= Degree; i++) {
                q.coefficients[i] = p.coefficients[i] * w;
            }
        }

        /**
         * \brief ����������ʽ��������ӣ���������洢����һ������ʽ�У�p1 * w1 + p2 * w2.
         * 
         * \param p1 ����ʽ1
         * \param w1 ����ʽ1��Ȩ��
         * \param p2 ����ʽ2
         * \param w2 ����ʽ2��Ȩ��
         * \param q ������洢��q��
         */
        static void AddScaled(const Polynomial& p1, const float& w1, const Polynomial& p2, const float& w2, Polynomial& q) {
            for (int i = 0; i <= Degree; i++) {
                q.coefficients[i] = p1.coefficients[i] * w1 + p2.coefficients[i] * w2;
            }
        }

        /**
         * \brief ����������ʽ��ӣ���������һ������ʽ�����һ��������Ȩ��ֵ��q = p1 + w2 * p2.
         * 
         * \param p1 ����ʽ1
         * \param p2 ����ʽ2
         * \param w2 �ڶ�������ʽȨ��
         * \param q ������洢��q��
         */
        static void AddScaled(const Polynomial& p1, const Polynomial& p2, const float& w2, Polynomial& q) {
            for (int i = 0; i <= Degree; i++) {
                q.coefficients[i] = p1.coefficients[i] + p2.coefficients[i] * w2;
            }
        }

        /**
         * \brief ����������ʽ��˺󰴱����ӵ���һ������ʽ�ϣ�q = w1 * p1 + p2.
         * 
         * \param p1 ��һ������ʽ p1
         * \param w1 p1��Ȩ��
         * \param p2 �ڶ�������ʽ p2 
         * \param q ������洢��q��
         */
        static void AddScaled(const Polynomial& p1, const float& w1, const Polynomial& p2, Polynomial& q) {
            for (int i = 0; i <= Degree; i++) {
                q.coefficients[i] = p1.coefficients[i] * w1 + p2.coefficients[i];
            }
        }

        /**
         * \brief ���������ʽ��
         *        coefficients[0] x^0 + coefficients[1] x^1 + coefficients[2] x^2 + coefficients[3] x^3 = c
         *        ���̵�ʵ����ᱻ������������ᱻ�ų�.
         * \param c ���̳���
         * \param roots ���̵ĸ�
         * \param EPS ��ʾ���ȵĲ���������ȷ����ľ�ȷ�ȡ����������֮��Ĳ�С�� EPS�����������ⱻ��Ϊ����ȵ�
         */
        void getSolutions(const float& c, std::vector<float>& roots, const float& EPS) const {
            float r[4][2];
            int rCount = 0;
            roots.clear();
            switch (Degree) {
            case 1:
                rCount = Factor::FactorSolver(coefficients[1], coefficients[0] - c, r, EPS);
                break;
            case 2:
                rCount = Factor::FactorSolver(coefficients[2], coefficients[1], coefficients[0] - c, r, EPS);
                break;
            case 3:
                rCount = Factor::FactorSolver(coefficients[3], coefficients[2], coefficients[1], coefficients[0] - c, r, EPS);
                break;
            case 4:
                rCount = Factor::FactorSolver(coefficients[4], coefficients[3], coefficients[2], coefficients[1], coefficients[0] - c, r, EPS);
                break;
            default:
                printf("��μ����Ϸ���û�������ʽ���޷����: %d\n", Degree);
            }
            for (int i = 0; i < rCount; i++) {
                if (fabs(r[i][1]) <= EPS) {
                    roots.push_back(r[i][0]);
                    printf("%d] %f\t%f\n", i, r[i][0], (*this)(r[i][0]) - c);
                }
            }
        }
	};
}



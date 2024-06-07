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
        float coefficients[Degree + 1] = { 0.0f }; // 多项式的系数

        /**
         * \brief 构造函数,初始化所有系数为0.
         * 
         * \return 
         */
        __host__ __device__ Polynomial(void) = default;
        /**
         * \brief 构造函数，用P赋值当前多项式，多项式项数最小的为准.
         *
         * \param P 多项式
         */
        template<int Degree2>
        Polynomial(const Polynomial<Degree2>& P) {
            for (int i = 0; i <= Degree && i <= Degree2; i++) {
                coefficients[i] = P.coefficients[i];
            }
        }

        /**
         * \brief 重载(),计算f(t)的值.
         * 
         * \param t 自变量
         * \return 函数值
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
         * \brief 计算给定区间[tmin, tmax]上的定积分, 其中 tmin ≥ tmax 也是可以的.
         * 
         * \param tMin 区间左边界
         * \param tMax 区间右边界
         * \return 函数定积分后的值
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
         * \brief 重载 "==" ,判断多项式是否相等.
         * 
         * \param p 与之比较的多项式
         * \return 相等返回true, 不等返回false
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
         * \brief 重载 "!=" ,判断多项式是否不相等.
         * 
         * \param p 与之比较的多项式
         * \return 不等返回true, 相等返回false
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
         * \brief 判断多项式是否为0.
         * 
         * \return 为0返回true，不为0返回false
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
         * \brief 将多项式设置为0.
         * 
         */
        void setZero(void) {
            memset(coefficients, 0, sizeof(float) * (Degree + 1));
        }

        /**
         * \brief 重载赋值符 " = ", 两个多项式最小得的degree作为基准的degree，将p的值赋值给当前多项式.
         * 
         * \param p 多项式p，阶数为Degree2
         * \return 多项式
         */
        template<int Degree2>
        Polynomial& operator  = (const Polynomial<Degree2>& p) {
            int d = Degree < Degree2 ? Degree : Degree2;    // 选择degree小的
            memset(coefficients, 0, sizeof(float) * (Degree + 1));
            memcpy(coefficients, p.coefficients, sizeof(float) * (d + 1));
            return *this;
        }
        /**
         * \brief 重载自加符 " += ", 多项式自加.
         * 
         * \param p 需要另加的多项式
         * \return 自加后的结果
         */
        Polynomial& operator += (const Polynomial& p) {
            for (int i = 0; i <= Degree; i++) { 
                coefficients[i] += p.coefficients[i]; 
            }
            return *this;
        }
        /**
         * \brief 重载自减符 " += ", 多项式自减.
         * 
         * \param p 需要减的多项式
         * \return 自减后的结果
         */
        Polynomial& operator -= (const Polynomial& p) {
            for (int i = 0; i <= Degree; i++) {
                coefficients[i] -= p.coefficients[i]; 
            }
            return *this;
        }
        /**
         * \brief 多项式取反.
         * 
         * \return 取反后的结果
         */
        Polynomial  operator -  (void) const {
            Polynomial q = *this;
            for (int i = 0; i <= Degree; i++) { 
                q.coefficients[i] = -q.coefficients[i];
            }
            return q;
        }
        /**
         * \brief 多项式相加.
         * 
         * \param p 需要相加的多项式
         * \return 相加的结果
         */
        Polynomial  operator +  (const Polynomial& p) const {
            Polynomial q;
            for (int i = 0; i <= Degree; i++) {
                q.coefficients[i] = (coefficients[i] + p.coefficients[i]); 
            }
            return q;
        }
        /**
         * \brief 多项式相减.
         * 
         * \param p 需要相减的多项式，减数
         * \return 相减的结果
         */
        Polynomial  operator -  (const Polynomial& p) const {

        }

        /**
         * \brief 多项式乘法.
         * 
         * \param p 需要相乘的多项式
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
         * \brief 重载 " += " 多项式常数项自加，常数系数 + s.
         * 
         * \param s 常数
         * \return 常数项相加后的多项式
         */
        Polynomial& operator += (const float& s) {
            coefficients[0] += s;
            return *this;
        }
        /**
         * \brief 重载 " -= " 多项式常数项自减，常数系数 - s.
         * 
         * \param s 常数
         * \return 常数项相减后的多项式
         */
        Polynomial& operator -= (const float& s) {
            coefficients[0] -= s;
            return *this;
        }
        /**
         * \brief 重载 " *= ", 多项式每个项系数自乘s.
         * 
         * \param s 常数
         * \return 各项系数自乘s后的多项式
         */
        Polynomial& operator *= (const float& s) {
            for (int i = 0; i <= Degree; i++) {
                coefficients[i] *= s; 
            }
            return *this;
        }
        /**
         * \brief 重载 " /= ", 多项式每个项系数自除s.
         *
         * \param s 常数
         * \return 各项系数自除s后的多项式
         */
        Polynomial& operator /= (const float& s) {
            for (int i = 0; i <= Degree; i++) {
                coefficients[i] /= s; 
            }
            return *this;
        }
        /**
         * \brief 重载 " + ", 多项式常数项 + 常数s.
         * 
         * \param s 常数s
         * \return 常数相加后的多项式
         */
        Polynomial  operator +  (const float& s) const {
            Polynomial<Degree> q = *this;
            q.coefficients[0] += s;
            return q;
        }
        /**
         * \brief 重载 " - ", 多项式常数项 - 常数s.
         *
         * \param s 常数s
         * \return 常数相减后的多项式
         */
        Polynomial  operator -  (const float& s) const {
            Polynomial q = *this;
            q.coefficients[0] -= s;
            return q;
        }
        /**
         * \brief 重载 " * ", 多项式各项系数 * 常数s【缩放系数】.
         *
         * \param s 常数s
         * \return 多项式各项系数 × 常数s后的多项式
         */
        Polynomial  operator *  (const float& s) const {
            Polynomial q;
            for (int i = 0; i <= Degree; i++) { 
                q.coefficients[i] = coefficients[i] * s; 
            }
            return q;
        }
        /**
         * \brief 重载 " / ", 多项式各项系数 / 常数s【缩放系数】.
         *
         * \param s 常数s
         * \return 多项式各项系数 ÷ 常数s后的多项式
         */
        Polynomial  operator /  (const float& s) const {
            Polynomial q(Degree);
            for (int i = 0; i <= Degree; i++) { 
                q.coefficients[i] = coefficients[i] / s;
            }
            return q;
        }

        /**
         * \brief 做如下变换：原式(100.0000 * x^0 + 100.0000 * x^1 + 100.0000 * x^2).
         *                  scale(10.0)  =>  100.0000 x^0 +10.0000 x^1 +1.0000 x^2  (原始式子值不改变)
         * \param s 缩放系数
         * \return 缩放后的值
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
         * \brief 做如下变换【f(x) -> f(x-t)】, 多项式x方向平移t.
         *        原式：f(x) = 1.0000 * x^0 + 1.0000 * x^1 + 1.0000 * x^2 + 1.0000 * x^3 ( t = 1时 )
          *       常数项： 1
          *       一次项：-1   +   1x
          *       二次项： 1   -   2x   +   1x^2
          *       三次项：-1   +   3x   -   3x^2    +   1x^3
          *       结果：f(x-1) = 2 * x^1 - 2 * x^2 + x^3
         * \param t 转换的常数
         * \return 多项式 f(x - t)
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
         * \brief 求解多项式的导数【f(x) -> f'(x)】.
         *        原式：f(x) = 1.0000 * x^0 + 1.0000 * x^1 + 1.0000 * x^2 + 1.0000 * x^3
         *        结果：f'(x) = 1.0000 * x^0 + 2.0000 * x^1 +3.0000 * x^2
         * 
         * \return 导数多项式
         */
        Polynomial<Degree - 1> derivative(void) const {
            Polynomial<Degree - 1> p;
            for (int i = 0; i < Degree; i++) {
                p.coefficients[i] = coefficients[i + 1] * (i + 1);
            }
            return p;
        }

        /**
         * \brief 计算多项式的积分原函数【f(x) -> F(x), 其中C = 0】.
         *        原式：f(x) =  1.0000 * x^0 + 2.0000 * x^1 + 3.0000 * x^2 + 4.0000 * x^3
         *        结果：F(x) = C * x^0 + 1.0000 * x^1 + 1.0000 * x^2 + 1.0000 * x^3 + 1.0000 * x^4 (C = 0)
         * 
         * \return 原函数多项式
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
         * \brief 输出多项式.
         * 
         */
        void printnl(void) const {
            for (int j = 0; j <= Degree; j++) {
                printf("%6.4f · x^%d ", coefficients[j], j);
                if (j < Degree && coefficients[j + 1] >= 0) {
                    printf(" + ");
                }
            }
            printf("\n");
        }

        /**
         * \brief 将当前的多项式 *this 与另一个多项式 p 按照给定的比例进行相加，并将结果存储在当前的多项式中：
         *        原式：1.000 * x^0 + 1.000 * x^1 + 1.000 * x^2 + 1.000 * x^3  调用 a->addScaled(*a, 10)计算如下：
         *        (1.000 * x^0 + 1.000 * x^1 + 1.000 * x^2 + 1.000 * x^3) + 10 * (1.000 * x^0 + 1.000 * x^1 + 1.000 * x^2 + 1.000 * x^3).
         *        结果为：11.000 * x^0 + 11.000 * x^1 + 11.000 * x^2 + 11.000 * x^3
         * \param p 多项式p
         * \param scale 相加比例
         * \return 比例相加后的结果
         */
        Polynomial& addScaled(const Polynomial& p, const float& scale) {
            for (int i = 0; i <= Degree; i++) {
                coefficients[i] += p.coefficients[i] * scale;
            }
            return *this;
        }

        /**
         * \brief 将一个多项式取负，并将结果存储在另一个多项式中：计算给定多项式 in 的负值.
         * 
         * \param in 输入多项式
         * \param out 输出多项式取负的结果
         */
        static void Negate(const Polynomial& in, Polynomial& out) {
            out = in;
            for (int i = 0; i <= Degree; i++) {
                out.coefficients[i] = -out.coefficients[i];
            }
        }

        /**
         * \brief 计算 p1 - p2，并将结果存储在另一个多项式q中【Polynomial<Degree>::Subtract 只会考虑[x^0, x^1, ..., x^Degree]项的多项式p1 和 p2】.
         * 
         * \param p1 多项式p1，可以是任意阶的多项式
         * \param p2 多项式p2，可以是任意阶的多项式
         * \param q 计算的结果，必须是一个 Degree 阶的多项式
         */
        static void Subtract(const Polynomial& p1, const Polynomial& p2, Polynomial& q) {
            for (int i = 0; i <= Degree; i++) {
                q.coefficients[i] = p1.coefficients[i] - p2.coefficients[i];
            }
        }

        /**
         * \brief 将一个多项式按照给定的比例进行缩放，并将结果存储在另一个多项式q中【Polynomial<Degree>::Scale 只会考虑[x^0, x^1, ..., x^Degree]项的多项式 p】.
         * 
         * \param p 多项式p，可以是任意阶的多项式
         * \param w 缩放比例
         * \param q 缩放结果，必须是一个Degree阶的多项式
         */
        static void Scale(const Polynomial& p, const float& w, Polynomial& q) {
            for (int i = 0; i <= Degree; i++) {
                q.coefficients[i] = p.coefficients[i] * w;
            }
        }

        /**
         * \brief 将两个多项式按比例相加，并将结果存储在另一个多项式中：p1 * w1 + p2 * w2.
         * 
         * \param p1 多项式1
         * \param w1 多项式1的权重
         * \param p2 多项式2
         * \param w2 多项式2的权重
         * \param q 将结果存储到q中
         */
        static void AddScaled(const Polynomial& p1, const float& w1, const Polynomial& p2, const float& w2, Polynomial& q) {
            for (int i = 0; i <= Degree; i++) {
                q.coefficients[i] = p1.coefficients[i] * w1 + p2.coefficients[i] * w2;
            }
        }

        /**
         * \brief 将两个多项式相加，并且其中一个多项式会乘以一个给定的权重值：q = p1 + w2 * p2.
         * 
         * \param p1 多项式1
         * \param p2 多项式2
         * \param w2 第二个多项式权重
         * \param q 将结果存储到q中
         */
        static void AddScaled(const Polynomial& p1, const Polynomial& p2, const float& w2, Polynomial& q) {
            for (int i = 0; i <= Degree; i++) {
                q.coefficients[i] = p1.coefficients[i] + p2.coefficients[i] * w2;
            }
        }

        /**
         * \brief 将两个多项式相乘后按比例加到另一个多项式上：q = w1 * p1 + p2.
         * 
         * \param p1 第一个多项式 p1
         * \param w1 p1的权重
         * \param p2 第二个多项式 p2 
         * \param q 将结果存储到q中
         */
        static void AddScaled(const Polynomial& p1, const float& w1, const Polynomial& p2, Polynomial& q) {
            for (int i = 0; i <= Degree; i++) {
                q.coefficients[i] = p1.coefficients[i] * w1 + p2.coefficients[i];
            }
        }

        /**
         * \brief 求解下述等式：
         *        coefficients[0] x^0 + coefficients[1] x^1 + coefficients[2] x^2 + coefficients[3] x^3 = c
         *        方程的实数解会被保留，虚数解会被排除.
         * \param c 方程常数
         * \param roots 方程的根
         * \param EPS 表示精度的参数，用于确定解的精确度。如果两个解之间的差小于 EPS，则这两个解被认为是相等的
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
                printf("五次及以上方程没有求根公式，无法求解: %d\n", Degree);
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



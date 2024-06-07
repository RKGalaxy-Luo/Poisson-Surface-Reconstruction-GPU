/*****************************************************************//**
 * \file   Factor.h
 * \brief  方程求解
 * 
 * \author LUOJIAXUAN
 * \date   May 21st 2024
 *********************************************************************/
#pragma once
#include <cmath>
namespace SparseSurfelFusion{

	class Factor {
	public:
		struct FactorConstants {
			static const double SQRT_3;	// 根号三
			static const double PI;		// pi
		};
	public:
		/**
		  * \brief 一次方程 ( a0 * x^0 + a1 * x^1 = 0 ) , 获得根 x = ( roots[0][0] + roots[0][1] * i )
		  *
		  * \param a1 一次项系数
		  * \param a0 常数项
		  * \param roots 求解的根
		  * \param 两个根距离小于ESP，就判定是一个根
		  * \return 返回方程根的数量
		  */
		int FactorSolver(double a1, double a0, double roots[1][2], const double& EPS);
		/**
		  * \brief 二次方程 ( a0 * x^0 + a1 * x^1 + a2 * x^2 = 0 ),
		  *        获得根: x1 = ( roots[0][0] + roots[0][1] * i )
		  *               x2 = ( roots[1][0] + roots[1][1] * i )
		  * \param a2 二次项系数
		  * \param a1 一次项系数
		  * \param a0 常数项
		  * \param roots 求解的根
		  * \param 两个根距离小于ESP，就判定是一个根
		  * \return 返回方程根的数量
		  */
		int FactorSolver(double a2, double a1, double a0, double roots[2][2], const double& EPS);

		/**
		 * \brief 三次方程 ( a0 * x^0 + a1 * x^1 + a2 * x^2 + a3 * x^3 = 0 ).
		  *        获得根: x1 = ( roots[0][0] + roots[0][1] * i )
		  *               x2 = ( roots[1][0] + roots[1][1] * i )
		  *               x3 = ( roots[2][0] + roots[2][1] * i )
		  *
		  * \param a3 三次项系数
		  * \param a2 二次项系数
		  * \param a1 一次项系数
		  * \param a0 常数项
		  * \param roots 求解的根
		  * \param 两个根距离小于ESP，就判定是一个根
		  * \return 返回方程根的数量
		 */
		int FactorSolver(double a3, double a2, double a1, double a0, double roots[3][2], const double& EPS);

		/**
		 * \brief 四次方程 ( a0 * x^0 + a1 * x^1 + a2 * x^2 + a3 * x^3 + a4 * x^4 = 0 ).
		  *        获得根: x1 = ( roots[0][0] + roots[0][1] * i )
		  *               x2 = ( roots[1][0] + roots[1][1] * i )
		  *               x3 = ( roots[2][0] + roots[2][1] * i )
		  *               x4 = ( roots[3][0] + roots[3][1] * i )
		  *
		  * \param a4 四次项系数
		  * \param a3 三次项系数
		  * \param a2 二次项系数
		  * \param a1 一次项系数
		  * \param a0 常数项
		  * \param roots 求解的根
		  * \param 两个根距离小于ESP，就判定是一个根
		  * \return 返回方程根的数量
		 */
		int FactorSolver(double a4, double a3, double a2, double a1, double a0, double roots[4][2], const double& EPS);

		/**
			获得 dim 次方程的解
			假设  方程解为 x1, x2 = [1, 2]
			3 x1 + 7 x2 = 17
			8 x1 + 2 x2 = 12
			[    3   7   ][  x1   ]   =   [   17  ]
			[    8   2   ][  x2   ]       [   12  ]
			系数矩阵 [ dim * dim ]     =   { 3, 7, 8, 2 }
			常数矩阵 [ dim ]           =   { 17, 12 }
			=>
			方程解矩阵   [ dim ]       =   { 1, 2 }
			如果有某个唯一解, 返回true
			如果没有解或有无数个解, 返回false
		  */
		bool Solve(const double* eqns, const double* values, double* solutions, const int& dim);

		/**
		 * \brief 求点(x,y)的角度(弧度 = 度 * PI / 180).
		 */
		double ArcTan2(const double& y, const double& x);

		/**
		 * \brief 求(in[0], in[1])的角度(弧度 = 度 * PI / 180).
		 */
		double Angle(const double in[2]);

		/**
		  * \brief out被覆盖, 将点 (in[0],in[1]) 的向量的角度减半, 新点 (out[0],out[1]) 与原点 (0,0) 的距离
		  *        将是原点 (0,0) 与原始点 (in[0],in[1]) 之间距离的平方根
		  */
		void Sqrt(const double in[2], double out[2]);

		/**
		 * \brief out被覆盖, (out[0], out[1]) = (in1[0] + in2[0], in1[1] + in2[1]).
		 */
		void Add(const double in1[2], const double in2[2], double out[2]);

		/**
		 * \brief out被覆盖, (out[0], out[1]) = (in1[0] - in2[0], in1[1] - in2[1]).
		 */
		void Subtract(const double in1[2], const double in2[2], double out[2]);

		/**
		 * \brief out被覆盖.
		 */
		void Multiply(const double in1[2], const double in2[2], double out[2]);

		/**
		 * \brief out被覆盖,.
		 */
		void Divide(const double in1[2], const double in2[2], double out[2]);


	};

}

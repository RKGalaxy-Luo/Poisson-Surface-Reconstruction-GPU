/*****************************************************************//**
 * \file   Factor.h
 * \brief  �������
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
			static const double SQRT_3;	// ������
			static const double PI;		// pi
		};
	public:
		/**
		  * \brief һ�η��� ( a0 * x^0 + a1 * x^1 = 0 ) , ��ø� x = ( roots[0][0] + roots[0][1] * i )
		  *
		  * \param a1 һ����ϵ��
		  * \param a0 ������
		  * \param roots ���ĸ�
		  * \param ����������С��ESP�����ж���һ����
		  * \return ���ط��̸�������
		  */
		int FactorSolver(double a1, double a0, double roots[1][2], const double& EPS);
		/**
		  * \brief ���η��� ( a0 * x^0 + a1 * x^1 + a2 * x^2 = 0 ),
		  *        ��ø�: x1 = ( roots[0][0] + roots[0][1] * i )
		  *               x2 = ( roots[1][0] + roots[1][1] * i )
		  * \param a2 ������ϵ��
		  * \param a1 һ����ϵ��
		  * \param a0 ������
		  * \param roots ���ĸ�
		  * \param ����������С��ESP�����ж���һ����
		  * \return ���ط��̸�������
		  */
		int FactorSolver(double a2, double a1, double a0, double roots[2][2], const double& EPS);

		/**
		 * \brief ���η��� ( a0 * x^0 + a1 * x^1 + a2 * x^2 + a3 * x^3 = 0 ).
		  *        ��ø�: x1 = ( roots[0][0] + roots[0][1] * i )
		  *               x2 = ( roots[1][0] + roots[1][1] * i )
		  *               x3 = ( roots[2][0] + roots[2][1] * i )
		  *
		  * \param a3 ������ϵ��
		  * \param a2 ������ϵ��
		  * \param a1 һ����ϵ��
		  * \param a0 ������
		  * \param roots ���ĸ�
		  * \param ����������С��ESP�����ж���һ����
		  * \return ���ط��̸�������
		 */
		int FactorSolver(double a3, double a2, double a1, double a0, double roots[3][2], const double& EPS);

		/**
		 * \brief �Ĵη��� ( a0 * x^0 + a1 * x^1 + a2 * x^2 + a3 * x^3 + a4 * x^4 = 0 ).
		  *        ��ø�: x1 = ( roots[0][0] + roots[0][1] * i )
		  *               x2 = ( roots[1][0] + roots[1][1] * i )
		  *               x3 = ( roots[2][0] + roots[2][1] * i )
		  *               x4 = ( roots[3][0] + roots[3][1] * i )
		  *
		  * \param a4 �Ĵ���ϵ��
		  * \param a3 ������ϵ��
		  * \param a2 ������ϵ��
		  * \param a1 һ����ϵ��
		  * \param a0 ������
		  * \param roots ���ĸ�
		  * \param ����������С��ESP�����ж���һ����
		  * \return ���ط��̸�������
		 */
		int FactorSolver(double a4, double a3, double a2, double a1, double a0, double roots[4][2], const double& EPS);

		/**
			��� dim �η��̵Ľ�
			����  ���̽�Ϊ x1, x2 = [1, 2]
			3 x1 + 7 x2 = 17
			8 x1 + 2 x2 = 12
			[    3   7   ][  x1   ]   =   [   17  ]
			[    8   2   ][  x2   ]       [   12  ]
			ϵ������ [ dim * dim ]     =   { 3, 7, 8, 2 }
			�������� [ dim ]           =   { 17, 12 }
			=>
			���̽����   [ dim ]       =   { 1, 2 }
			�����ĳ��Ψһ��, ����true
			���û�н������������, ����false
		  */
		bool Solve(const double* eqns, const double* values, double* solutions, const int& dim);

		/**
		 * \brief ���(x,y)�ĽǶ�(���� = �� * PI / 180).
		 */
		double ArcTan2(const double& y, const double& x);

		/**
		 * \brief ��(in[0], in[1])�ĽǶ�(���� = �� * PI / 180).
		 */
		double Angle(const double in[2]);

		/**
		  * \brief out������, ���� (in[0],in[1]) �������ĽǶȼ���, �µ� (out[0],out[1]) ��ԭ�� (0,0) �ľ���
		  *        ����ԭ�� (0,0) ��ԭʼ�� (in[0],in[1]) ֮������ƽ����
		  */
		void Sqrt(const double in[2], double out[2]);

		/**
		 * \brief out������, (out[0], out[1]) = (in1[0] + in2[0], in1[1] + in2[1]).
		 */
		void Add(const double in1[2], const double in2[2], double out[2]);

		/**
		 * \brief out������, (out[0], out[1]) = (in1[0] - in2[0], in1[1] - in2[1]).
		 */
		void Subtract(const double in1[2], const double in2[2], double out[2]);

		/**
		 * \brief out������.
		 */
		void Multiply(const double in1[2], const double in2[2], double out[2]);

		/**
		 * \brief out������,.
		 */
		void Divide(const double in1[2], const double in2[2], double out[2]);


	};

}

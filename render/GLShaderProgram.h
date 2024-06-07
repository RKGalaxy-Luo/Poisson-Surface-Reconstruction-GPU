/*****************************************************************//**
 * \file   GLShaderProgram.h
 * \brief  OpenGL渲染器类，获取渲染器文件
 * 
 * \author LUO
 * \date   February 23rd 2024
 *********************************************************************/
#pragma once
#include <base/GlobalConfigs.h>
#include <base/CommonTypes.h>
#include <base/Logging.h>
#include <glad/glad.h>
#include <glm/glm.hpp>//OpenGL矩阵运算库
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <fstream>
#include <sstream>
#include <Eigen/Dense>

namespace SparseSurfelFusion {
	class GLShaderProgram {
	private:
		GLuint vertexShader;
		GLuint fragmentShader;
		GLuint geometryShader;
		GLuint programID;
	public:
		//构造函数和析构函数
		/**
		 * \brief 构造函数，赋值初值.
		 * 
		 */
		explicit GLShaderProgram();
		~GLShaderProgram();
		/* 从源代码编译着色程序的方法
		 */
		/**
		 * \brief 根据顶点着色器和片段着色器路径，编译shader.
		 * 
		 * \param vertexPath 顶点着色器文件路径
		 * \param fragmentPath 片段着色器文件路径
		 */
		void Compile(const std::string& vertexPath, const std::string& fragmentPath);

		/**
		 * \brief 根据着色器文件路径，读取并编译着色器文件.
		 * 
		 * \param vertexShaderCode 顶点着色器源码
		 * \param fragmentShaderCode 片段着色器源码
		 * \param geometryShaderCode 几何着色器源码
		 */
		void Compile(const char* vertexShaderCode, const char* fragmentShaderCode, const char* geometryShaderCode = nullptr);

	private:
		/**
		 * \brief 检查着色器文件读取，以及着色器编译，是否存在错误.
		 * 
		 * \param shader 着色器标识符
		 * \param type 检查类型："ShaderProgram" 是着色器程序编译错误，其余均是着色器文件读取错误
		 */
		static void checkShaderCompilerError(GLuint shader, const std::string& type);

		/* The methods to use and setup uniform values
		 */
	public:
		/**
		 * \brief 激活着色器程序，以便使用uniform值.
		 * 
		 */
		void BindProgram() const { glUseProgram(programID); }
		/**
		 * \brief 解除着色器程序.
		 * 
		 */
		void UnbindProgram() const { glUseProgram(0); }
		/**
		 * \brief 获得当前着色器标识符.
		 * 
		 * \return 当前着色器标识符
		 */
		GLuint getProgramID() const { return programID; }

		/**
		 * \brief 将Matrix值传入shader进行计算渲染.
		 * 
		 * \param UniformName 参数在shader源码中的名字
		 * \param value 传入的Matrix类型的值
		 */
		void SetUniformMatrix(const std::string& UniformName, const Eigen::Matrix4f& value) const;

		/**
		 * \brief 将Vector4f值传入shader进行计算渲染.
		 *
		 * \param UniformName 参数在shader源码中的名字
		 * \param value 传入的Vector4f类型的值
		 */
		void SetUniformVector(const std::string& UniformName, const  Eigen::Vector4f& vec) const;

		/**
		 * \brief 将float4值传入shader进行计算渲染.
		 *
		 * \param UniformName 参数在shader源码中的名字
		 * \param value 传入的float4类型的值
		 */
		void SetUniformVector(const std::string& UniformName, const float4& vec) const;

		/**
		 * \brief 将glm::vec4值传入shader进行计算渲染.
		 *
		 * \param UniformName 参数在shader源码中的名字
		 * \param value 传入的glm::vec4类型的值
		 */
		void SetUniformVector(const std::string& UniformName, const glm::vec4& value) const;

		/**
		 * \brief 传入四维向量到shader中进行计算.
		 * 
		 * \param UniformName 参数在shader源码中的名字
		 * \param x	传入四维向量的x分量
		 * \param y	传入四维向量的y分量
		 * \param z	传入四维向量的z分量
		 * \param w	传入四维向量的w分量
		 */
		void SetUniformVector(const std::string& UniformName, float x, float y, float z, float w) const;

		/**
		 * \brief 将float3值传入shader进行计算渲染.
		 *
		 * \param UniformName 参数在shader源码中的名字
		 * \param value 传入的float3类型的值
		 */
		void SetUniformVector(const std::string& UniformName, const float3& vec) const;

		/**
		 * \brief 将glm::vec3值传入shader进行计算渲染.
		 *
		 * \param UniformName 参数在shader源码中的名字
		 * \param value 传入的glm::vec3类型的值
		 */
		void SetUniformVector(const std::string& UniformName, const glm::vec3& value) const;

		/**
		 * \brief 传入三维向量到shader中进行计算.
		 * 
		 * \param UniformName 参数在shader源码中的名字
		 * \param x	传入三维向量的x分量
		 * \param y	传入三维向量的y分量
		 * \param z	传入三维向量的z分量
		 */
		void SetUniformVector(const std::string& UniformName, float x, float y, float z) const;

		/**
		 * \brief 将float2值传入shader进行计算渲染.
		 *
		 * \param UniformName 参数在shader源码中的名字
		 * \param value 传入的float2类型的值
		 */
		void SetUniformVector(const std::string& UniformName, const float2& vec) const;

		/**
		 * \brief 将glm::vec2值传入shader进行计算渲染.
		 *
		 * \param UniformName 参数在shader源码中的名字
		 * \param value 传入的glm::vec2类型的值
		 */
		void SetUniformVector(const std::string& UniformName, const glm::vec2& value) const;

		/**
		 * \brief 传入二维向量到shader中进行计算.
		 * 
		 * \param UniformName 参数在shader源码中的名字
		 * \param x	传入二维向量的x分量
		 * \param y	传入二维向量的y分量
		 */
		void SetUniformVector(const std::string& UniformName, float x, float y) const;

		/**
		 * \brief 将float值传入shader进行计算渲染.
		 *
		 * \param UniformName 参数在shader源码中的名字
		 * \param value 传入的float类型的值
		 */
		void SetUniformFloat(const std::string& UniformName, const float value) const;
		/**
		 * \brief 将glm::mat4值传入shader进行计算渲染.
		 * 
		 * \param UniformName 参数在shader源码中的名字
		 * \param value 传入的glm::mat4类型的值
		 */
		void setUniformMat4(const std::string& UniformName, glm::mat4 value) const;


	};
}



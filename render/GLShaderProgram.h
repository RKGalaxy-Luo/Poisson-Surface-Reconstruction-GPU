/*****************************************************************//**
 * \file   GLShaderProgram.h
 * \brief  OpenGL��Ⱦ���࣬��ȡ��Ⱦ���ļ�
 * 
 * \author LUO
 * \date   February 23rd 2024
 *********************************************************************/
#pragma once
#include <base/GlobalConfigs.h>
#include <base/CommonTypes.h>
#include <base/Logging.h>
#include <glad/glad.h>
#include <glm/glm.hpp>//OpenGL���������
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
		//���캯������������
		/**
		 * \brief ���캯������ֵ��ֵ.
		 * 
		 */
		explicit GLShaderProgram();
		~GLShaderProgram();
		/* ��Դ���������ɫ����ķ���
		 */
		/**
		 * \brief ���ݶ�����ɫ����Ƭ����ɫ��·��������shader.
		 * 
		 * \param vertexPath ������ɫ���ļ�·��
		 * \param fragmentPath Ƭ����ɫ���ļ�·��
		 */
		void Compile(const std::string& vertexPath, const std::string& fragmentPath);

		/**
		 * \brief ������ɫ���ļ�·������ȡ��������ɫ���ļ�.
		 * 
		 * \param vertexShaderCode ������ɫ��Դ��
		 * \param fragmentShaderCode Ƭ����ɫ��Դ��
		 * \param geometryShaderCode ������ɫ��Դ��
		 */
		void Compile(const char* vertexShaderCode, const char* fragmentShaderCode, const char* geometryShaderCode = nullptr);

	private:
		/**
		 * \brief �����ɫ���ļ���ȡ���Լ���ɫ�����룬�Ƿ���ڴ���.
		 * 
		 * \param shader ��ɫ����ʶ��
		 * \param type ������ͣ�"ShaderProgram" ����ɫ���������������������ɫ���ļ���ȡ����
		 */
		static void checkShaderCompilerError(GLuint shader, const std::string& type);

		/* The methods to use and setup uniform values
		 */
	public:
		/**
		 * \brief ������ɫ�������Ա�ʹ��uniformֵ.
		 * 
		 */
		void BindProgram() const { glUseProgram(programID); }
		/**
		 * \brief �����ɫ������.
		 * 
		 */
		void UnbindProgram() const { glUseProgram(0); }
		/**
		 * \brief ��õ�ǰ��ɫ����ʶ��.
		 * 
		 * \return ��ǰ��ɫ����ʶ��
		 */
		GLuint getProgramID() const { return programID; }

		/**
		 * \brief ��Matrixֵ����shader���м�����Ⱦ.
		 * 
		 * \param UniformName ������shaderԴ���е�����
		 * \param value �����Matrix���͵�ֵ
		 */
		void SetUniformMatrix(const std::string& UniformName, const Eigen::Matrix4f& value) const;

		/**
		 * \brief ��Vector4fֵ����shader���м�����Ⱦ.
		 *
		 * \param UniformName ������shaderԴ���е�����
		 * \param value �����Vector4f���͵�ֵ
		 */
		void SetUniformVector(const std::string& UniformName, const  Eigen::Vector4f& vec) const;

		/**
		 * \brief ��float4ֵ����shader���м�����Ⱦ.
		 *
		 * \param UniformName ������shaderԴ���е�����
		 * \param value �����float4���͵�ֵ
		 */
		void SetUniformVector(const std::string& UniformName, const float4& vec) const;

		/**
		 * \brief ��glm::vec4ֵ����shader���м�����Ⱦ.
		 *
		 * \param UniformName ������shaderԴ���е�����
		 * \param value �����glm::vec4���͵�ֵ
		 */
		void SetUniformVector(const std::string& UniformName, const glm::vec4& value) const;

		/**
		 * \brief ������ά������shader�н��м���.
		 * 
		 * \param UniformName ������shaderԴ���е�����
		 * \param x	������ά������x����
		 * \param y	������ά������y����
		 * \param z	������ά������z����
		 * \param w	������ά������w����
		 */
		void SetUniformVector(const std::string& UniformName, float x, float y, float z, float w) const;

		/**
		 * \brief ��float3ֵ����shader���м�����Ⱦ.
		 *
		 * \param UniformName ������shaderԴ���е�����
		 * \param value �����float3���͵�ֵ
		 */
		void SetUniformVector(const std::string& UniformName, const float3& vec) const;

		/**
		 * \brief ��glm::vec3ֵ����shader���м�����Ⱦ.
		 *
		 * \param UniformName ������shaderԴ���е�����
		 * \param value �����glm::vec3���͵�ֵ
		 */
		void SetUniformVector(const std::string& UniformName, const glm::vec3& value) const;

		/**
		 * \brief ������ά������shader�н��м���.
		 * 
		 * \param UniformName ������shaderԴ���е�����
		 * \param x	������ά������x����
		 * \param y	������ά������y����
		 * \param z	������ά������z����
		 */
		void SetUniformVector(const std::string& UniformName, float x, float y, float z) const;

		/**
		 * \brief ��float2ֵ����shader���м�����Ⱦ.
		 *
		 * \param UniformName ������shaderԴ���е�����
		 * \param value �����float2���͵�ֵ
		 */
		void SetUniformVector(const std::string& UniformName, const float2& vec) const;

		/**
		 * \brief ��glm::vec2ֵ����shader���м�����Ⱦ.
		 *
		 * \param UniformName ������shaderԴ���е�����
		 * \param value �����glm::vec2���͵�ֵ
		 */
		void SetUniformVector(const std::string& UniformName, const glm::vec2& value) const;

		/**
		 * \brief �����ά������shader�н��м���.
		 * 
		 * \param UniformName ������shaderԴ���е�����
		 * \param x	�����ά������x����
		 * \param y	�����ά������y����
		 */
		void SetUniformVector(const std::string& UniformName, float x, float y) const;

		/**
		 * \brief ��floatֵ����shader���м�����Ⱦ.
		 *
		 * \param UniformName ������shaderԴ���е�����
		 * \param value �����float���͵�ֵ
		 */
		void SetUniformFloat(const std::string& UniformName, const float value) const;
		/**
		 * \brief ��glm::mat4ֵ����shader���м�����Ⱦ.
		 * 
		 * \param UniformName ������shaderԴ���е�����
		 * \param value �����glm::mat4���͵�ֵ
		 */
		void setUniformMat4(const std::string& UniformName, glm::mat4 value) const;


	};
}



/*****************************************************************//**
 * \file   GLShaderProgram.h
 * \brief  OpenGL渲染器类，获取渲染器文件
 *
 * \author LUO
 * \date   February 23rd 2024
 *********************************************************************/
#include "GLShaderProgram.h"

SparseSurfelFusion::GLShaderProgram::GLShaderProgram() : vertexShader(0) , fragmentShader(0) , geometryShader(0) , programID(0)
{
}

SparseSurfelFusion::GLShaderProgram::~GLShaderProgram()
{
	glDeleteProgram(programID);
}

void SparseSurfelFusion::GLShaderProgram::Compile(const std::string& vertexPath, const std::string& fragmentPath)
{
	std::string vertex_code;
	std::string fragment_code;
	std::ifstream vertexShaderFile;
	std::ifstream fragmentShaderFile;
	vertexShaderFile.exceptions(std::ifstream::failbit | std::ifstream::badbit);
	fragmentShaderFile.exceptions(std::ifstream::failbit | std::ifstream::badbit);
	try {
		vertexShaderFile.open(vertexPath);
		fragmentShaderFile.open(fragmentPath);
		std::stringstream vertex_stream, fragment_stream;
		vertex_stream << vertexShaderFile.rdbuf();
		fragment_stream << fragmentShaderFile.rdbuf();
		vertexShaderFile.close();
		fragmentShaderFile.close();
		vertex_code = vertex_stream.str();
		fragment_code = fragment_stream.str();
	}
	catch (std::ifstream::failure e) {
		LOGGING(FATAL) << "错误：着色器文件没有正确加载！";
	}
	vertex_code.push_back('\0');
	fragment_code.push_back('\0');
	Compile(vertex_code.c_str(), fragment_code.c_str());
}

void SparseSurfelFusion::GLShaderProgram::Compile(const char* vertexShaderCode, const char* fragmentShaderCode, const char* geometryShaderCode)
{
	// 创建一个着色器对象，将其注册为顶点着色器  -->  GL_VERTEX_SHADER
	vertexShader = glCreateShader(GL_VERTEX_SHADER);
	// 参数1：着色器对象
	// 参数2：指定了传递的源码字符串数量，这里只有一个
	// 参数3：顶点着色器真正的源码
	// 参数4：先设置为NULL
	glShaderSource(vertexShader, 1, &vertexShaderCode, NULL);
	// 编译着色器
	glCompileShader(vertexShader);
	checkShaderCompilerError(vertexShader, "VertexShader");

	//对于片段着色器
	fragmentShader = glCreateShader(GL_FRAGMENT_SHADER);
	glShaderSource(fragmentShader, 1, &fragmentShaderCode, NULL);
	glCompileShader(fragmentShader);
	checkShaderCompilerError(fragmentShader, "FragmentShader");

	//对于几何着色器
	if (geometryShaderCode != nullptr) {
		geometryShader = glCreateShader(GL_GEOMETRY_SHADER);
		glShaderSource(geometryShader, 1, &geometryShaderCode, NULL);
		glCompileShader(geometryShader);
		checkShaderCompilerError(geometryShader, "GeometryShader");
	}

	//链接到着色器程序
	programID = glCreateProgram();
	glAttachShader(programID, vertexShader);
	glAttachShader(programID, fragmentShader);
	if (geometryShader != 0) glAttachShader(programID, geometryShader);
	glLinkProgram(programID);
	checkShaderCompilerError(programID, "ShaderProgram");

	//因为他们已经和这个项目联系在一起了
	glDeleteShader(vertexShader);
	glDeleteShader(fragmentShader);
	if (geometryShader != 0) glDeleteShader(geometryShader);
}

void SparseSurfelFusion::GLShaderProgram::checkShaderCompilerError(GLuint shader, const std::string& type)
{
	int success;
	char infoLog[1024];
	if (type != "ShaderProgram")
	{
		glGetShaderiv(shader, GL_COMPILE_STATUS, &success);
		if (!success)
		{
			glGetShaderInfoLog(shader, 1024, NULL, infoLog);
			LOGGING(FATAL) << "Error 渲染器文件读取错误： " << type << "\n" << infoLog << "\n -- --------------------------------------------------- -- " << std::endl;
		}
	}
	else
	{
		glGetProgramiv(shader, GL_LINK_STATUS, &success);
		if (!success)
		{
			glGetProgramInfoLog(shader, 1024, NULL, infoLog);
			LOGGING(FATAL) << "Error 着色器程序编译错误: " << type << "\n" << infoLog << "\n -- --------------------------------------------------- -- " << std::endl;
		}
	}
}

void SparseSurfelFusion::GLShaderProgram::SetUniformMatrix(const std::string& UniformName, const Eigen::Matrix4f& value) const
{
	GLint uniformLocation = glGetUniformLocation(getProgramID(), UniformName.c_str());	// 获得参数的位置
	// 参数1：uniform的位置值
	// 参数2：告诉OpenGL我们将要发送多少个Uniform参数
	// 参数3：询问我们是否希望对我们的矩阵进行转置(Transpose)，也就是说交换我们矩阵的行和列
	// 参数4：真正的矩阵数据
	glUniformMatrix4fv(uniformLocation, 1, GL_FALSE, value.data());
}

void SparseSurfelFusion::GLShaderProgram::SetUniformVector(const std::string& UniformName, const Eigen::Vector4f& vec) const
{
	GLint uniformLocation = glGetUniformLocation(getProgramID(), UniformName.c_str());	// 获得参数的位置
	// 参数1：uniform的位置值
	// 参数2：向量vec的x分量
	// 参数3：向量vec的y分量
	// 参数4：向量vec的z分量
	// 参数5：向量vec的w分量
	glUniform4f(uniformLocation, vec[0], vec[1], vec[2], vec[3]);
}

void SparseSurfelFusion::GLShaderProgram::SetUniformVector(const std::string& UniformName, const float4& vec) const
{
	GLint uniformLocation = glGetUniformLocation(getProgramID(), UniformName.c_str());	// 获得参数的位置
	// 参数1：uniform的位置值
	// 参数2：向量vec的x分量
	// 参数3：向量vec的y分量
	// 参数4：向量vec的z分量
	// 参数5：向量vec的w分量
	glUniform4f(uniformLocation, vec.x, vec.y, vec.z, vec.w);
}

void SparseSurfelFusion::GLShaderProgram::SetUniformVector(const std::string& UniformName, const glm::vec4& value) const
{
	glUniform4fv(glGetUniformLocation(getProgramID(), UniformName.c_str()), 1, &value[0]);
}

void SparseSurfelFusion::GLShaderProgram::SetUniformVector(const std::string& UniformName, float x, float y, float z, float w) const
{
	glUniform4f(glGetUniformLocation(getProgramID(), UniformName.c_str()), x, y, z, w);
}

void SparseSurfelFusion::GLShaderProgram::SetUniformVector(const std::string& UniformName, const float3& vec) const
{
	GLint uniformLocation = glGetUniformLocation(getProgramID(), UniformName.c_str());	// 获得参数的位置
	// 参数1：uniform的位置值
	// 参数2：向量vec的x分量
	// 参数3：向量vec的y分量
	// 参数4：向量vec的z分量
	glUniform3f(uniformLocation, vec.x, vec.y, vec.z);
}

void SparseSurfelFusion::GLShaderProgram::SetUniformVector(const std::string& UniformName, const glm::vec3& value) const
{
	glUniform3fv(glGetUniformLocation(getProgramID(), UniformName.c_str()), 1, &value[0]);
}

void SparseSurfelFusion::GLShaderProgram::SetUniformVector(const std::string& UniformName, float x, float y, float z) const
{
	glUniform3f(glGetUniformLocation(getProgramID(), UniformName.c_str()), x, y, z);
}

void SparseSurfelFusion::GLShaderProgram::SetUniformVector(const std::string& UniformName, const float2& vec) const
{
	GLint uniformLocation = glGetUniformLocation(getProgramID(), UniformName.c_str());	// 获得参数的位置
	// 参数1：uniform的位置值
	// 参数2：向量vec的x分量
	// 参数3：向量vec的y分量
	glUniform2f(uniformLocation, vec.x, vec.y);
}

void SparseSurfelFusion::GLShaderProgram::SetUniformVector(const std::string& UniformName, const glm::vec2& value) const
{
	glUniform2fv(glGetUniformLocation(getProgramID(), UniformName.c_str()), 1, &value[0]);
}

void SparseSurfelFusion::GLShaderProgram::SetUniformVector(const std::string& UniformName, float x, float y) const
{
	glUniform2f(glGetUniformLocation(getProgramID(), UniformName.c_str()), x, y);
}

void SparseSurfelFusion::GLShaderProgram::SetUniformFloat(const std::string& UniformName, const float value) const
{
	GLint uniformLocation = glGetUniformLocation(getProgramID(), UniformName.c_str());	// 获得参数的位置
	// 参数1：uniform的位置值
	// 参数2：向量vec的x分量
	glUniform1f(uniformLocation, value);
}

void SparseSurfelFusion::GLShaderProgram::setUniformMat4(const std::string& UniformName, glm::mat4 value) const
{
	//第一个参数是uniform的位置值
	//第二个参数告诉OpenGL我们将要发送多少个矩阵
	//第三个参数询问我们是否希望对我们的矩阵进行转置(Transpose)，也就是说交换我们矩阵的行和列
	//第四个参数是真正的矩阵数据，但是GLM并不是把它们的矩阵储存为OpenGL所希望接受的那种
	//因此我们要先用GLM的自带的函数value_ptr来变换这些数据
	GLint uniformLocation = glGetUniformLocation(getProgramID(), UniformName.c_str());
	glUniformMatrix4fv(uniformLocation, 1, GL_FALSE, glm::value_ptr(value));
}

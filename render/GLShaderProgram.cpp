/*****************************************************************//**
 * \file   GLShaderProgram.h
 * \brief  OpenGL��Ⱦ���࣬��ȡ��Ⱦ���ļ�
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
		LOGGING(FATAL) << "������ɫ���ļ�û����ȷ���أ�";
	}
	vertex_code.push_back('\0');
	fragment_code.push_back('\0');
	Compile(vertex_code.c_str(), fragment_code.c_str());
}

void SparseSurfelFusion::GLShaderProgram::Compile(const char* vertexShaderCode, const char* fragmentShaderCode, const char* geometryShaderCode)
{
	// ����һ����ɫ�����󣬽���ע��Ϊ������ɫ��  -->  GL_VERTEX_SHADER
	vertexShader = glCreateShader(GL_VERTEX_SHADER);
	// ����1����ɫ������
	// ����2��ָ���˴��ݵ�Դ���ַ�������������ֻ��һ��
	// ����3��������ɫ��������Դ��
	// ����4��������ΪNULL
	glShaderSource(vertexShader, 1, &vertexShaderCode, NULL);
	// ������ɫ��
	glCompileShader(vertexShader);
	checkShaderCompilerError(vertexShader, "VertexShader");

	//����Ƭ����ɫ��
	fragmentShader = glCreateShader(GL_FRAGMENT_SHADER);
	glShaderSource(fragmentShader, 1, &fragmentShaderCode, NULL);
	glCompileShader(fragmentShader);
	checkShaderCompilerError(fragmentShader, "FragmentShader");

	//���ڼ�����ɫ��
	if (geometryShaderCode != nullptr) {
		geometryShader = glCreateShader(GL_GEOMETRY_SHADER);
		glShaderSource(geometryShader, 1, &geometryShaderCode, NULL);
		glCompileShader(geometryShader);
		checkShaderCompilerError(geometryShader, "GeometryShader");
	}

	//���ӵ���ɫ������
	programID = glCreateProgram();
	glAttachShader(programID, vertexShader);
	glAttachShader(programID, fragmentShader);
	if (geometryShader != 0) glAttachShader(programID, geometryShader);
	glLinkProgram(programID);
	checkShaderCompilerError(programID, "ShaderProgram");

	//��Ϊ�����Ѿ��������Ŀ��ϵ��һ����
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
			LOGGING(FATAL) << "Error ��Ⱦ���ļ���ȡ���� " << type << "\n" << infoLog << "\n -- --------------------------------------------------- -- " << std::endl;
		}
	}
	else
	{
		glGetProgramiv(shader, GL_LINK_STATUS, &success);
		if (!success)
		{
			glGetProgramInfoLog(shader, 1024, NULL, infoLog);
			LOGGING(FATAL) << "Error ��ɫ������������: " << type << "\n" << infoLog << "\n -- --------------------------------------------------- -- " << std::endl;
		}
	}
}

void SparseSurfelFusion::GLShaderProgram::SetUniformMatrix(const std::string& UniformName, const Eigen::Matrix4f& value) const
{
	GLint uniformLocation = glGetUniformLocation(getProgramID(), UniformName.c_str());	// ��ò�����λ��
	// ����1��uniform��λ��ֵ
	// ����2������OpenGL���ǽ�Ҫ���Ͷ��ٸ�Uniform����
	// ����3��ѯ�������Ƿ�ϣ�������ǵľ������ת��(Transpose)��Ҳ����˵�������Ǿ�����к���
	// ����4�������ľ�������
	glUniformMatrix4fv(uniformLocation, 1, GL_FALSE, value.data());
}

void SparseSurfelFusion::GLShaderProgram::SetUniformVector(const std::string& UniformName, const Eigen::Vector4f& vec) const
{
	GLint uniformLocation = glGetUniformLocation(getProgramID(), UniformName.c_str());	// ��ò�����λ��
	// ����1��uniform��λ��ֵ
	// ����2������vec��x����
	// ����3������vec��y����
	// ����4������vec��z����
	// ����5������vec��w����
	glUniform4f(uniformLocation, vec[0], vec[1], vec[2], vec[3]);
}

void SparseSurfelFusion::GLShaderProgram::SetUniformVector(const std::string& UniformName, const float4& vec) const
{
	GLint uniformLocation = glGetUniformLocation(getProgramID(), UniformName.c_str());	// ��ò�����λ��
	// ����1��uniform��λ��ֵ
	// ����2������vec��x����
	// ����3������vec��y����
	// ����4������vec��z����
	// ����5������vec��w����
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
	GLint uniformLocation = glGetUniformLocation(getProgramID(), UniformName.c_str());	// ��ò�����λ��
	// ����1��uniform��λ��ֵ
	// ����2������vec��x����
	// ����3������vec��y����
	// ����4������vec��z����
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
	GLint uniformLocation = glGetUniformLocation(getProgramID(), UniformName.c_str());	// ��ò�����λ��
	// ����1��uniform��λ��ֵ
	// ����2������vec��x����
	// ����3������vec��y����
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
	GLint uniformLocation = glGetUniformLocation(getProgramID(), UniformName.c_str());	// ��ò�����λ��
	// ����1��uniform��λ��ֵ
	// ����2������vec��x����
	glUniform1f(uniformLocation, value);
}

void SparseSurfelFusion::GLShaderProgram::setUniformMat4(const std::string& UniformName, glm::mat4 value) const
{
	//��һ��������uniform��λ��ֵ
	//�ڶ�����������OpenGL���ǽ�Ҫ���Ͷ��ٸ�����
	//����������ѯ�������Ƿ�ϣ�������ǵľ������ת��(Transpose)��Ҳ����˵�������Ǿ�����к���
	//���ĸ������������ľ������ݣ�����GLM�����ǰ����ǵľ��󴢴�ΪOpenGL��ϣ�����ܵ�����
	//�������Ҫ����GLM���Դ��ĺ���value_ptr���任��Щ����
	GLint uniformLocation = glGetUniformLocation(getProgramID(), UniformName.c_str());
	glUniformMatrix4fv(uniformLocation, 1, GL_FALSE, glm::value_ptr(value));
}

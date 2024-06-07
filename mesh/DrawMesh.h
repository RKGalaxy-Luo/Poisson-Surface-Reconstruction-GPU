/*****************************************************************//**
 * \file   DrawMesh.h
 * \brief  OpenGL������Ⱦ����
 * 
 * \author LUOJIAXUAN
 * \date   June 5th 2024
 *********************************************************************/
#pragma once
#include <glad/glad.h>			// <GLFW/glfw3.h>���Զ������ܶ��ϰ汾�����<glad/glad.h>��ǰ�棬��ô��ʹ��glad��Ӧ���°汾��OpenGL����Ҳ��Ϊʲô<glad/glad.h>������<GLFW/glfw3.h>֮ǰ
//#define GLFW_INCLUDE_NONE		// ��ʾ�Ľ��� <GLFW/glfw3.h> �Զ��������������Ĺ��ܣ�ʹ���������֮��Ͳ����ٴӿ��������а�������<GLFW/glfw3.h>Ҳ������<glad/glad.h>֮ǰ
#include <GLFW/glfw3.h>

#include <glm/glm.hpp>//OpenGL���������
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include "Geometry.h"
#include <chrono>
#include <render/GLShaderProgram.h>
#include <base/DeviceReadWrite/DeviceBufferArray.h>


static glm::vec3 box[68] = {
	// x��								x����ɫ
	{ 0.0f,   0.0f,   0.0f },			{1.0f,   0.0f,   0.0f},
	{ 1.0f,   0.0f,   0.0f },			{1.0f,   0.0f,   0.0f},
	{ 1.0f,   0.0f,   0.0f },			{1.0f,   0.0f,   0.0f},
	{ 0.8f,   0.1f,   0.0f },			{1.0f,   0.0f,   0.0f},
	{ 1.0f,   0.0f,   0.0f },			{1.0f,   0.0f,   0.0f},
	{ 0.8f,  -0.1f,   0.0f },		    {1.0f,   0.0f,   0.0f},
	{ 1.2f,   0.1f,   0.0f },		    {1.0f,   0.0f,   0.0f},
	{ 1.3f,  -0.1f,   0.0f },		    {1.0f,   0.0f,   0.0f},
	{ 1.2f,  -0.1f,   0.0f },		    {1.0f,   0.0f,   0.0f},
	{ 1.3f,   0.1f,   0.0f },		    {1.0f,   0.0f,   0.0f},

	{ 0.0f,   0.0f,   0.0f },			{0.0f,   1.0f,   0.0f},
	{ 0.0f,   1.0f,   0.0f },			{0.0f,   1.0f,   0.0f},
	{ 0.0f,   1.0f,   0.0f },			{0.0f,   1.0f,   0.0f},
	{ 0.1f,   0.8f,   0.0f },			{0.0f,   1.0f,   0.0f},
	{ 0.0f,   1.0f,   0.0f },			{0.0f,   1.0f,   0.0f},
	{ -0.1,   0.8f,   0.0f },			{0.0f,   1.0f,   0.0f},
	{ 0.05f,  1.3f,   0.0f },			{0.0f,   1.0f,   0.0f},
	{ 0.0f,   1.2f,   0.0f },			{0.0f,   1.0f,   0.0f},
	{ -0.05f, 1.3f,   0.0f },			{0.0f,   1.0f,   0.0f},
	{ 0.0f,   1.2f,   0.0f },			{0.0f,   1.0f,   0.0f},
	{ 0.0f,   1.2f,   0.0f },			{0.0f,   1.0f,   0.0f},
	{ 0.0f,   1.1f,   0.0f },			{0.0f,   1.0f,   0.0f},

	{ 0.0f,   0.0f,   0.0f },			{0.0f,   0.0f,   1.0f},
	{ 0.0f,   0.0f,   1.0f },			{0.0f,   0.0f,   1.0f},
	{ 0.0f,   0.0f,   1.0f },			{0.0f,   0.0f,   1.0f},
	{ 0.1f,   0.0f,   0.8f },			{0.0f,   0.0f,   1.0f},
	{ 0.0f,   0.0f,   1.0f },			{0.0f,   0.0f,   1.0f},
	{ -0.1f,  0.0f,   0.8f },			{0.0f,   0.0f,   1.0f},
	{ -0.05f, 0.0f,   1.2f },			{0.0f,   0.0f,   1.0f},
	{ 0.05f,  0.0f,   1.2f },			{0.0f,   0.0f,   1.0f},
	{ 0.05f,  0.0f,   1.2f },			{0.0f,   0.0f,   1.0f},
	{ -0.05f, 0.0f,   1.1f },			{0.0f,   0.0f,   1.0f},
	{ -0.05f, 0.0f,   1.1f },			{0.0f,   0.0f,   1.0f},
	{ 0.05f,  0.0f,   1.1f },			{0.0f,   0.0f,   1.0f}
};

namespace SparseSurfelFusion {
	namespace device {

		/**
		 * \brief ������һ��.
		 */
		__device__ float3 VectorNormalize(const float3& normal);

		/**
		 * \brief �������.
		 */
		__device__ float3 CrossProduct(const float3& Vector_OA, const float3& Vector_OB);

		/**
		 * \brief ���������ߺ˺���.
		 */
		__global__ void CalculateMeshNormalsKernel(const Point3D<float>* verticesArray, const TriangleIndex* indicesArray, const unsigned int meshCount, Point3D<float>* normalsArray);

		/**
		 * \brief ���㶥���ڽӶ��ٸ�������.
		 */
		__global__ void CountConnectedTriangleNumKernel(const TriangleIndex* indicesArray, const unsigned int meshCount, unsigned int* ConnectedTriangleNum);

		/**
		 * \brief �����ڽ������η�������.
		 */
		__global__ void VerticesNormalsSumKernel(const Point3D<float>* meshNormals, const TriangleIndex* indicesArray, const unsigned int meshCount, Point3D<float>* VerticesNormalsSum);

		/**
		 * \brief ͨ��ƽ���ڽ�����Mesh�ķ��ߣ����㵱ǰ����ķ���.
		 */
		__global__ void CalculateVerticesAverageNormals(const unsigned int* ConnectedTriangleNum, const Point3D<float>* VerticesNormalsSum, const unsigned int verticesCount, Point3D<float>* VerticesAverageNormals);
	}
	class DrawMesh
	{
	public:
		DrawMesh();
		~DrawMesh() = default;
		
		using Ptr = std::shared_ptr<DrawMesh>;

		/**
		 * \brief �����ؽ���������ķ���.
		 * 
		 * \param mesh ��������
		 */
		void CalculateMeshNormals(CoredVectorMeshData& mesh, cudaStream_t stream = 0);

		/**
		 * \brief ������Ⱦ������.
		 * 
		 * \param mesh ��������
		 */
		void DrawRenderedMesh(CoredVectorMeshData& mesh);

	private:

		std::vector<Point3D<float>> VerticesNormals;

		GLFWwindow* window;					// ����ָ��
		GLShaderProgram meshShader;			// ������Ⱦ
		GLShaderProgram coordinateShader;	// ����ϵ��Ⱦ

		// ���Ƶ���
		GLuint GeometryVAO;					// �������ɵ������VAO
		GLuint GeometryVBO;					// �������ɵ������VBO
		GLuint GeometryIBO;					// �������ɵ������EBO/IBO
		// ������Ⱦ��������ϵ
		GLuint coordinateSystemVAO;			// ����ϵVAO
		GLuint coordinateSystemVBO;			// ����ϵ�������VBO


		/**
		 * \brief ��ʼ������������ϵ.
		 *
		 */
		void initialCoordinateSystem();


	};

}


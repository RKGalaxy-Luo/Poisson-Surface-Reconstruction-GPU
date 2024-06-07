/*****************************************************************//**
 * \file   DrawMesh.h
 * \brief  OpenGL绘制渲染网格
 * 
 * \author LUOJIAXUAN
 * \date   June 5th 2024
 *********************************************************************/
#pragma once
#include <glad/glad.h>			// <GLFW/glfw3.h>会自动包含很多老版本，如果<glad/glad.h>在前面，那么就使用glad对应最新版本的OpenGL，这也是为什么<glad/glad.h>必须在<GLFW/glfw3.h>之前
//#define GLFW_INCLUDE_NONE		// 显示的禁用 <GLFW/glfw3.h> 自动包含开发环境的功能，使用这个功能之后就不会再从开发环境中包含，即<GLFW/glfw3.h>也可以在<glad/glad.h>之前
#include <GLFW/glfw3.h>

#include <glm/glm.hpp>//OpenGL矩阵运算库
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include "Geometry.h"
#include <chrono>
#include <render/GLShaderProgram.h>
#include <base/DeviceReadWrite/DeviceBufferArray.h>


static glm::vec3 box[68] = {
	// x轴								x轴颜色
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
		 * \brief 向量归一化.
		 */
		__device__ float3 VectorNormalize(const float3& normal);

		/**
		 * \brief 向量叉乘.
		 */
		__device__ float3 CrossProduct(const float3& Vector_OA, const float3& Vector_OB);

		/**
		 * \brief 计算网格法线核函数.
		 */
		__global__ void CalculateMeshNormalsKernel(const Point3D<float>* verticesArray, const TriangleIndex* indicesArray, const unsigned int meshCount, Point3D<float>* normalsArray);

		/**
		 * \brief 计算顶点邻接多少个三角形.
		 */
		__global__ void CountConnectedTriangleNumKernel(const TriangleIndex* indicesArray, const unsigned int meshCount, unsigned int* ConnectedTriangleNum);

		/**
		 * \brief 计算邻接三角形法向量和.
		 */
		__global__ void VerticesNormalsSumKernel(const Point3D<float>* meshNormals, const TriangleIndex* indicesArray, const unsigned int meshCount, Point3D<float>* VerticesNormalsSum);

		/**
		 * \brief 通过平均邻接三角Mesh的法线，计算当前顶点的法线.
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
		 * \brief 计算重建三角网格的法线.
		 * 
		 * \param mesh 传入网格
		 */
		void CalculateMeshNormals(CoredVectorMeshData& mesh, cudaStream_t stream = 0);

		/**
		 * \brief 绘制渲染的网格.
		 * 
		 * \param mesh 传入网格
		 */
		void DrawRenderedMesh(CoredVectorMeshData& mesh);

	private:

		std::vector<Point3D<float>> VerticesNormals;

		GLFWwindow* window;					// 窗口指针
		GLShaderProgram meshShader;			// 网格渲染
		GLShaderProgram coordinateShader;	// 坐标系渲染

		// 绘制点云
		GLuint GeometryVAO;					// 点云生成的网格的VAO
		GLuint GeometryVBO;					// 点云生成的网格的VBO
		GLuint GeometryIBO;					// 点云生成的网格的EBO/IBO
		// 绘制渲染窗口坐标系
		GLuint coordinateSystemVAO;			// 坐标系VAO
		GLuint coordinateSystemVBO;			// 坐标系轴点网格VBO


		/**
		 * \brief 初始化并加载坐标系.
		 *
		 */
		void initialCoordinateSystem();


	};

}


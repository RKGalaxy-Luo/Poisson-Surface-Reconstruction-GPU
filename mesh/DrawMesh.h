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

#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <cuda_gl_interop.h>	// CUDA与OpenGL显存共享
#include <tuple>				// tuple是一个固定大小的不同类型值的集合，是泛化的std::pair

#include "Geometry.h"
#include <chrono>
#include <render/GLShaderProgram.h>
#include <base/DeviceReadWrite/DeviceBufferArray.h>
#include <base/Constants.h>

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
		~DrawMesh();
		
		using Ptr = std::shared_ptr<DrawMesh>;

		/**
		 * \brief 计算重建三角网格的法线.
		 * 
		 * \param meshVertices 网格重建后的顶点
		 * \param meshTriangleIndices 网格重建后的三角形索引
		 * \param stream cuda流
		 */
		void CalculateMeshNormals(DeviceArrayView<Point3D<float>> meshVertices, DeviceArrayView<TriangleIndex> meshTriangleIndices, cudaStream_t stream = 0);

		/**
		 * \brief 绘制渲染的网格.
		 * 
		 * \param stream cuda流
		 */
		void DrawRenderedMesh(cudaStream_t stream);

	private:

		DeviceBufferArray<Point3D<float>> VerticesAverageNormals;	// 归一化的顶点平均法向量
		DeviceBufferArray<Point3D<float>> MeshVertices;				// 网格顶点
		DeviceBufferArray<TriangleIndex> MeshTriangleIndices;		// 网格三角面元索引

		const unsigned int WindowWidth = 1920 * 0.9;
		const unsigned int WindowHeight = 1080 * 0.9;

		GLFWwindow* window;					// 窗口指针
		GLShaderProgram meshShader;			// 网格渲染
		GLShaderProgram coordinateShader;	// 坐标系渲染

		// 绘制点云
		GLuint GeometryVAO;					// 点云生成的网格的VAO
		GLuint GeometryVBO;					// 点云生成的网格的VBO
		GLuint GeometryIBO;					// 点云生成的网格的EBO/IBO

		cudaGraphicsResource_t cudaVBOResources;// 注册缓冲区对象到CUDA
		cudaGraphicsResource_t cudaIBOResources;// 注册IBO对象到CUDA

		// 绘制渲染窗口坐标系
		GLuint coordinateSystemVAO;			// 坐标系VAO
		GLuint coordinateSystemVBO;			// 坐标系轴点网格VBO

		unsigned int TranglesCount = 0;		// 传入实时顶点的数量
		unsigned int VerticesCount = 0;		// 传入点的数量

		// 创建变换
		glm::mat4 view = glm::mat4(1.0f);		// 确保初始化矩阵是单位矩阵
		glm::mat4 projection = glm::mat4(1.0f);	// 投影矩阵，选择是透视还是正射
		glm::mat4 model = glm::mat4(1.0f);		// 计算每个对象的模型矩阵，并在绘制之前将其传递给着色器

		/**
		 * \brief 初始化并加载坐标系.
		 *
		 */
		void initialCoordinateSystem();

		/**
		 * \brief 注册cuda资源.
		 *
		 */
		void registerCudaResources();

		/**
		 * \brief 将数据资源映射到cuda.
		 *
		 * \param MeshVertices 网格顶底
		 * \param MeshTriangleIndices 网格三角面元索引
		 * \param stream cuda流
		 */
		void mapToCuda(DeviceArrayView<Point3D<float>> MeshVertices, DeviceArrayView<TriangleIndex> MeshTriangleIndices, cudaStream_t stream = 0);

		/**
		 * \brief 将需要渲染的点映射到cuda资源上.
		 *
		 */
		void unmapFromCuda(cudaStream_t stream = 0);

		/**
		 * \brief 清空窗口.
		 *
		 */
		void clearWindow();

		/**
		 * \brief 绘制网格.
		 *
		 * \param view 传入视角矩阵
		 * \param projection 传入投影矩阵
		 * \param model 传入模型矩阵
		 */
		void drawMesh(glm::mat4& view, glm::mat4& projection, glm::mat4& model);

		/**
		 * \brief 绘制坐标系.
		 *
		 * \param view 传入视角矩阵
		 * \param projection 传入投影矩阵
		 * \param model 传入模型矩阵
		 */
		void drawCoordinateSystem(glm::mat4& view, glm::mat4& projection, glm::mat4& model);

		/**
		 * \param 双缓冲并捕捉事件.
		 *
		 */
		void swapBufferAndCatchEvent();
	};

}


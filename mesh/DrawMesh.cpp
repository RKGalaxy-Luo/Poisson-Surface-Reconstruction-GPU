/*****************************************************************//**
 * \file   DrawMesh.cpp
 * \brief  OpenGL绘制渲染网格
 * 
 * \author LUOJIAXUAN
 * \date   June 5th 2024
 *********************************************************************/
#include "DrawMesh.h"

void framebufferSizeCallback(GLFWwindow* window, int width, int height)
{
	glViewport(0, 0, width, height);															// 设置视口
}

SparseSurfelFusion::DrawMesh::DrawMesh()
{
	int glfwSate = glfwInit();
	if (glfwSate == GLFW_FALSE)
	{
		std::cout << "GLFW 初始化失败!" << std::endl;
		exit(EXIT_FAILURE);
	}

	// opengl版本为4.6
	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 6);
	// 设置 OpenGL 配置文件为核心配置文件
	glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

	// 默认的framebuffer属性
	glfwWindowHint(GLFW_VISIBLE, GL_TRUE);		// 窗口可见
	glfwWindowHint(GLFW_SAMPLES, 1);
	glfwWindowHint(GLFW_STEREO, GL_FALSE);
	glfwWindowHint(GLFW_DOUBLEBUFFER, GL_TRUE);
	glfwWindowHint(GLFW_RESIZABLE, GL_FALSE);	// 窗口大小不可调整

	window = glfwCreateWindow(WINDOW_WIDTH, WINDOW_HEIGHT, "MeshRender", NULL, NULL);

	if (window == NULL) {
		LOGGING(FATAL) << "未正确创建GLFW窗口！";
	}
	else std::cout << "窗口 MeshRender 创建完成！" << std::endl;

	glfwMakeContextCurrent(window);

	// 初始化 GLAD
	if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress)) {
		LOGGING(FATAL) << "GLAD 初始化失败！";
		glfwDestroyWindow(window);
		glfwTerminate();
		exit(EXIT_FAILURE);
	}


	// 开启深度测试, 禁用“面剔除”功能
	glEnable(GL_DEPTH_TEST);				// 启用深度测试后，OpenGL会在绘制像素之前，根据它们的深度值进行比较，并只绘制深度测试通过的像素，从而产生正确的渲染效果
	glDepthFunc(GL_LESS);					// 用于深度测试，它决定哪些片段（像素）应该被显示，哪些应该被丢弃，基于它们的深度值
	//glPolygonMode(GL_FRONT, GL_LINE);		// 填充面
	//glEnable(GL_CULL_FACE);					// 意味着OpenGL将渲染所有的三角形面，而不管它们的顶点顺序，不管其是否被遮挡
	//glEnable(GL_PROGRAM_POINT_SIZE);		// 调用 glEnable(GL_PROGRAM_POINT_SIZE) 函数会启用程序控制的点大小功能，允许您在着色器程序中使用内置变量 gl_PointSize 来控制点的大小

	const std::string vertexShaderPath = SHADER_PATH_PREFIX + std::string("MeshShader.vert");
	const std::string fragmentShaderPath = SHADER_PATH_PREFIX + std::string("MeshShader.frag");

	meshShader.Compile(vertexShaderPath, fragmentShaderPath);
	initialCoordinateSystem();
}

void SparseSurfelFusion::DrawMesh::DrawRenderedMesh(CoredVectorMeshData& mesh)
{
	const int TranglesCount = mesh.triangleCount();		// 传入实时顶点的数量
	const int verticesNum = mesh.InCorePointsCount();	// 传入点的数量
	//printf("MeshCount = %d\n", TranglesCount);
	std::vector<GLfloat> vertices;
	std::vector<GLfloat> normals;
	std::vector<GLuint> elementIndex;

	if (!(mesh.GetVertexArray(vertices) && mesh.GetTriangleIndices(elementIndex))) {
		LOGGING(FATAL) << "渲染数据为空";
	}

	normals.resize(verticesNum * 3);
	memcpy(normals.data(), VerticesNormals.data(), sizeof(float) * verticesNum * 3);

	GLuint* elemIdxPtr = elementIndex.data();
	GLfloat* verticesPtr = vertices.data();
	GLfloat* normalsPtr = normals.data();
	glGenVertexArrays(1, &GeometryVAO);		// 生成VAO
	glGenBuffers(1, &GeometryVBO);			// 生成VBO
	glGenBuffers(1, &GeometryIBO);			// 创建1个IBO，并将标识符存储在IBO变量中

	glBindVertexArray(GeometryVAO);						// 绑定VAO
	glBindBuffer(GL_ARRAY_BUFFER, GeometryVBO);			// 绑定VBO
	GLsizei bufferSize = sizeof(GLfloat) * verticesNum * 6;				// x,y,z,nx,ny,nz = 6个GLfloat

	glBufferData(GL_ARRAY_BUFFER, bufferSize, NULL, GL_DYNAMIC_DRAW);	// 动态绘制，目前只是先开辟个大小
	// 参数1：指定要更新的缓冲区对象的目标
	// 参数2：指定要更新的数据在缓冲区中的偏移量（以字节为单位）。偏移量表示从缓冲区的起始位置开始的偏移量。
	// 参数3：指定要更新的数据的大小（以字节为单位）
	// 参数4：指向包含要更新的数据的指针
	glBufferSubData(GL_ARRAY_BUFFER, 0, sizeof(GLfloat) * verticesNum * 3, verticesPtr);								// 分批加载属性数组
	glBufferSubData(GL_ARRAY_BUFFER, sizeof(GLfloat) * verticesNum * 3, sizeof(GLfloat) * verticesNum * 3, normalsPtr);	// 分批加载属性数组

	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, GeometryIBO);														// 将EBO绑定到GL_ELEMENT_ARRAY_BUFFER目标
	glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(GLuint) * TranglesCount * 3, elemIdxPtr, GL_DYNAMIC_DRAW);	// 将索引数据从CPU传输到GPU，绘制顶点还需要索引数组？
	// 位置
	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(GLfloat), (void*)(0 * sizeof(GLfloat) * verticesNum));	// 设置VAO解释器
	glEnableVertexAttribArray(0);	// layout (location = 0)
	// 法线
	glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(GLfloat), (void*)(3 * sizeof(GLfloat) * verticesNum));	// 设置VAO解释器
	glEnableVertexAttribArray(1);	// layout (location = 1)

	glBindBuffer(GL_ARRAY_BUFFER, 0);			// 解绑VBO
	glBindVertexArray(0);						// 解绑VAO
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);	// 解绑IBO
	// 在双缓冲渲染中，通常会使用两个 VAO，交替地进行渲染和更新【GLFW自带双缓存渲染】

	// 调用了glClearColor来设置清空屏幕所用的颜色
	glClearColor(0.0f, 0.0f, 0.0f, 1.0f); //RGBA
	// 通过调用glClear函数来清空屏幕的颜色缓冲，它接受一个缓冲位(Buffer Bit)来指定要清空的缓冲，可能的缓冲位有GL_COLOR_BUFFER_BIT
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);// 现在同时清除深度缓冲区!(不清除深度画不出来立体图像)

	// 激活着色器
	meshShader.BindProgram(); //renderer构造时已经编译

	meshShader.SetUniformVector("objectColor", 0.7f, 0.7f, 0.7f);	// 紫色
	meshShader.SetUniformVector("lightColor", 1.0f, 1.0f, 1.0f);
	meshShader.SetUniformVector("lightPos", -1.2f, -1.0f, -2.0f);

	// 创建变换
	glm::mat4 view = glm::mat4(1.0f);		// 确保初始化矩阵是单位矩阵
	glm::mat4 projection = glm::mat4(1.0f);	// 投影矩阵，选择是透视还是正射
	glm::mat4 model = glm::mat4(1.0f);		// 计算每个对象的模型矩阵，并在绘制之前将其传递给着色器
	//设置透视矩阵
	projection = glm::perspective(glm::radians(30.0f), (float)WINDOW_WIDTH / (float)WINDOW_HEIGHT, 0.1f, 100.0f);
	meshShader.setUniformMat4(std::string("projection"), projection); // 注意:目前我们每帧设置投影矩阵，但由于投影矩阵很少改变，所以最好在主循环之外设置它一次。
	float radius = 3.0f;//摄像头绕的半径
	float camX = static_cast<float>(sin(glfwGetTime() * 0.5f) * radius);
	float camZ = static_cast<float>(cos(glfwGetTime() * 0.5f) * radius);
	view = glm::lookAt(glm::vec3(camX, 3.0f, camZ), glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3(0.0f, 1.0f, 0.0f));
	meshShader.setUniformMat4(std::string("view"), view);
	meshShader.SetUniformVector("viewPos", glm::vec3(camX, 3.0f, camZ));
	model = glm::translate(model, glm::vec3(0.0f, 0.0f, 0.0f));
	model = glm::rotate(model, glm::radians(0.0f), glm::vec3(0.0f, 1.0f, 0.0f));//绕Up向量(0,1,0)旋转
	meshShader.setUniformMat4(std::string("model"), model);

	glBindVertexArray(GeometryVAO); // 绑定VAO后绘制

	glDrawElements(GL_TRIANGLES, TranglesCount * 3, GL_UNSIGNED_INT, 0);

	// 清除绑定
	glBindVertexArray(0);
	meshShader.UnbindProgram();

	// 绘制坐标系
	coordinateShader.BindProgram();	// 绑定坐标轴的shader
	coordinateShader.setUniformMat4(std::string("projection"), projection);
	coordinateShader.setUniformMat4(std::string("view"), view);
	coordinateShader.setUniformMat4(std::string("model"), model);
	glBindVertexArray(coordinateSystemVAO); // 绑定VAO后绘制

	glLineWidth(3.0f);
	glDrawArrays(GL_LINES, 0, 34);	// box有54个元素，绘制线段

	// 清除绑定
	glBindVertexArray(0);
	coordinateShader.UnbindProgram();

	// 函数会交换颜色缓冲（它是一个储存着GLFW窗口每一个像素颜色值的大缓冲），它在这一迭代中被用来绘制，并且将会作为输出显示在屏幕上
	glfwSwapBuffers(window);
	glfwPollEvents();

	// 释放资源
	glDeleteVertexArrays(1, &GeometryVAO);
	glDeleteBuffers(1, &GeometryVBO);
	glDeleteBuffers(1, &GeometryIBO);
}



void SparseSurfelFusion::DrawMesh::initialCoordinateSystem()
{
	std::vector<float> pvalues;			// 点坐标	

	// Live域中的点，查看中间过程的顶点着色器
	const std::string coordinate_vert_path = SHADER_PATH_PREFIX + std::string("CoordinateSystemShader.vert");
	// Live域中的点，查看中间过程的片段着色器
	const std::string coordinate_frag_path = SHADER_PATH_PREFIX + std::string("CoordinateSystemShader.frag");
	coordinateShader.Compile(coordinate_vert_path, coordinate_frag_path);
	glGenVertexArrays(1, &coordinateSystemVAO);	// 生成VAO
	glGenBuffers(1, &coordinateSystemVBO);		// 生成VBO
	const unsigned int Num = sizeof(box) / sizeof(box[0]);
	for (int i = 0; i < Num; i++) {
		pvalues.push_back(box[i][0]);
		pvalues.push_back(box[i][1]);
		pvalues.push_back(box[i][2]);
	}
	//std::cout << "Num = " << Num << "     pvalues.size() = " << pvalues.size() << std::endl;
	glBindVertexArray(coordinateSystemVAO);
	glBindBuffer(GL_ARRAY_BUFFER, coordinateSystemVBO);
	GLsizei bufferSize = sizeof(GLfloat) * pvalues.size();		// float数据的数量
	glBufferData(GL_ARRAY_BUFFER, bufferSize, pvalues.data(), GL_DYNAMIC_DRAW);	// 动态绘制，目前只是先开辟个大小

	// 位置
	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(GLfloat), (void*)(0 * sizeof(GLfloat)));	// 设置VAO解释器
	glEnableVertexAttribArray(0);	// layout (location = 0)
	// 颜色
	glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(GLfloat), (void*)(3 * sizeof(GLfloat)));		// 设置VAO解释器
	glEnableVertexAttribArray(1);	// layout (location = 1)

	glBindBuffer(GL_ARRAY_BUFFER, 0);			// 解绑VBO
	glBindVertexArray(0);						// 解绑VAO
}



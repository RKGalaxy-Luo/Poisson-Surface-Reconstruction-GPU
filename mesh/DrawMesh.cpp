/*****************************************************************//**
 * \file   DrawMesh.cpp
 * \brief  OpenGL������Ⱦ����
 * 
 * \author LUOJIAXUAN
 * \date   June 5th 2024
 *********************************************************************/
#include "DrawMesh.h"

void framebufferSizeCallback(GLFWwindow* window, int width, int height)
{
	glViewport(0, 0, width, height);															// �����ӿ�
}

SparseSurfelFusion::DrawMesh::DrawMesh()
{
	int glfwSate = glfwInit();
	if (glfwSate == GLFW_FALSE)
	{
		std::cout << "GLFW ��ʼ��ʧ��!" << std::endl;
		exit(EXIT_FAILURE);
	}

	// opengl�汾Ϊ4.6
	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 6);
	// ���� OpenGL �����ļ�Ϊ���������ļ�
	glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

	// Ĭ�ϵ�framebuffer����
	glfwWindowHint(GLFW_VISIBLE, GL_TRUE);		// ���ڿɼ�
	glfwWindowHint(GLFW_SAMPLES, 1);
	glfwWindowHint(GLFW_STEREO, GL_FALSE);
	glfwWindowHint(GLFW_DOUBLEBUFFER, GL_TRUE);
	glfwWindowHint(GLFW_RESIZABLE, GL_FALSE);	// ���ڴ�С���ɵ���

	window = glfwCreateWindow(WINDOW_WIDTH, WINDOW_HEIGHT, "MeshRender", NULL, NULL);

	if (window == NULL) {
		LOGGING(FATAL) << "δ��ȷ����GLFW���ڣ�";
	}
	else std::cout << "���� MeshRender ������ɣ�" << std::endl;

	glfwMakeContextCurrent(window);

	// ��ʼ�� GLAD
	if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress)) {
		LOGGING(FATAL) << "GLAD ��ʼ��ʧ�ܣ�";
		glfwDestroyWindow(window);
		glfwTerminate();
		exit(EXIT_FAILURE);
	}


	// ������Ȳ���, ���á����޳�������
	glEnable(GL_DEPTH_TEST);				// ������Ȳ��Ժ�OpenGL���ڻ�������֮ǰ���������ǵ����ֵ���бȽϣ���ֻ������Ȳ���ͨ�������أ��Ӷ�������ȷ����ȾЧ��
	glDepthFunc(GL_LESS);					// ������Ȳ��ԣ���������ЩƬ�Σ����أ�Ӧ�ñ���ʾ����ЩӦ�ñ��������������ǵ����ֵ
	//glPolygonMode(GL_FRONT, GL_LINE);		// �����
	//glEnable(GL_CULL_FACE);					// ��ζ��OpenGL����Ⱦ���е��������棬���������ǵĶ���˳�򣬲������Ƿ��ڵ�
	//glEnable(GL_PROGRAM_POINT_SIZE);		// ���� glEnable(GL_PROGRAM_POINT_SIZE) ���������ó�����Ƶĵ��С���ܣ�����������ɫ��������ʹ�����ñ��� gl_PointSize �����Ƶ�Ĵ�С

	const std::string vertexShaderPath = SHADER_PATH_PREFIX + std::string("MeshShader.vert");
	const std::string fragmentShaderPath = SHADER_PATH_PREFIX + std::string("MeshShader.frag");

	meshShader.Compile(vertexShaderPath, fragmentShaderPath);
	initialCoordinateSystem();
}

void SparseSurfelFusion::DrawMesh::DrawRenderedMesh(CoredVectorMeshData& mesh)
{
	const int TranglesCount = mesh.triangleCount();		// ����ʵʱ���������
	const int verticesNum = mesh.InCorePointsCount();	// ����������
	//printf("MeshCount = %d\n", TranglesCount);
	std::vector<GLfloat> vertices;
	std::vector<GLfloat> normals;
	std::vector<GLuint> elementIndex;

	if (!(mesh.GetVertexArray(vertices) && mesh.GetTriangleIndices(elementIndex))) {
		LOGGING(FATAL) << "��Ⱦ����Ϊ��";
	}

	normals.resize(verticesNum * 3);
	memcpy(normals.data(), VerticesNormals.data(), sizeof(float) * verticesNum * 3);

	GLuint* elemIdxPtr = elementIndex.data();
	GLfloat* verticesPtr = vertices.data();
	GLfloat* normalsPtr = normals.data();
	glGenVertexArrays(1, &GeometryVAO);		// ����VAO
	glGenBuffers(1, &GeometryVBO);			// ����VBO
	glGenBuffers(1, &GeometryIBO);			// ����1��IBO��������ʶ���洢��IBO������

	glBindVertexArray(GeometryVAO);						// ��VAO
	glBindBuffer(GL_ARRAY_BUFFER, GeometryVBO);			// ��VBO
	GLsizei bufferSize = sizeof(GLfloat) * verticesNum * 6;				// x,y,z,nx,ny,nz = 6��GLfloat

	glBufferData(GL_ARRAY_BUFFER, bufferSize, NULL, GL_DYNAMIC_DRAW);	// ��̬���ƣ�Ŀǰֻ���ȿ��ٸ���С
	// ����1��ָ��Ҫ���µĻ����������Ŀ��
	// ����2��ָ��Ҫ���µ������ڻ������е�ƫ���������ֽ�Ϊ��λ����ƫ������ʾ�ӻ���������ʼλ�ÿ�ʼ��ƫ������
	// ����3��ָ��Ҫ���µ����ݵĴ�С�����ֽ�Ϊ��λ��
	// ����4��ָ�����Ҫ���µ����ݵ�ָ��
	glBufferSubData(GL_ARRAY_BUFFER, 0, sizeof(GLfloat) * verticesNum * 3, verticesPtr);								// ����������������
	glBufferSubData(GL_ARRAY_BUFFER, sizeof(GLfloat) * verticesNum * 3, sizeof(GLfloat) * verticesNum * 3, normalsPtr);	// ����������������

	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, GeometryIBO);														// ��EBO�󶨵�GL_ELEMENT_ARRAY_BUFFERĿ��
	glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(GLuint) * TranglesCount * 3, elemIdxPtr, GL_DYNAMIC_DRAW);	// ���������ݴ�CPU���䵽GPU�����ƶ��㻹��Ҫ�������飿
	// λ��
	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(GLfloat), (void*)(0 * sizeof(GLfloat) * verticesNum));	// ����VAO������
	glEnableVertexAttribArray(0);	// layout (location = 0)
	// ����
	glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(GLfloat), (void*)(3 * sizeof(GLfloat) * verticesNum));	// ����VAO������
	glEnableVertexAttribArray(1);	// layout (location = 1)

	glBindBuffer(GL_ARRAY_BUFFER, 0);			// ���VBO
	glBindVertexArray(0);						// ���VAO
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);	// ���IBO
	// ��˫������Ⱦ�У�ͨ����ʹ������ VAO������ؽ�����Ⱦ�͸��¡�GLFW�Դ�˫������Ⱦ��

	// ������glClearColor�����������Ļ���õ���ɫ
	glClearColor(0.0f, 0.0f, 0.0f, 1.0f); //RGBA
	// ͨ������glClear�����������Ļ����ɫ���壬������һ������λ(Buffer Bit)��ָ��Ҫ��յĻ��壬���ܵĻ���λ��GL_COLOR_BUFFER_BIT
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);// ����ͬʱ�����Ȼ�����!(�������Ȼ�����������ͼ��)

	// ������ɫ��
	meshShader.BindProgram(); //renderer����ʱ�Ѿ�����

	meshShader.SetUniformVector("objectColor", 0.7f, 0.7f, 0.7f);	// ��ɫ
	meshShader.SetUniformVector("lightColor", 1.0f, 1.0f, 1.0f);
	meshShader.SetUniformVector("lightPos", -1.2f, -1.0f, -2.0f);

	// �����任
	glm::mat4 view = glm::mat4(1.0f);		// ȷ����ʼ�������ǵ�λ����
	glm::mat4 projection = glm::mat4(1.0f);	// ͶӰ����ѡ����͸�ӻ�������
	glm::mat4 model = glm::mat4(1.0f);		// ����ÿ�������ģ�;��󣬲��ڻ���֮ǰ���䴫�ݸ���ɫ��
	//����͸�Ӿ���
	projection = glm::perspective(glm::radians(30.0f), (float)WINDOW_WIDTH / (float)WINDOW_HEIGHT, 0.1f, 100.0f);
	meshShader.setUniformMat4(std::string("projection"), projection); // ע��:Ŀǰ����ÿ֡����ͶӰ���󣬵�����ͶӰ������ٸı䣬�����������ѭ��֮��������һ�Ρ�
	float radius = 3.0f;//����ͷ�Ƶİ뾶
	float camX = static_cast<float>(sin(glfwGetTime() * 0.5f) * radius);
	float camZ = static_cast<float>(cos(glfwGetTime() * 0.5f) * radius);
	view = glm::lookAt(glm::vec3(camX, 3.0f, camZ), glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3(0.0f, 1.0f, 0.0f));
	meshShader.setUniformMat4(std::string("view"), view);
	meshShader.SetUniformVector("viewPos", glm::vec3(camX, 3.0f, camZ));
	model = glm::translate(model, glm::vec3(0.0f, 0.0f, 0.0f));
	model = glm::rotate(model, glm::radians(0.0f), glm::vec3(0.0f, 1.0f, 0.0f));//��Up����(0,1,0)��ת
	meshShader.setUniformMat4(std::string("model"), model);

	glBindVertexArray(GeometryVAO); // ��VAO�����

	glDrawElements(GL_TRIANGLES, TranglesCount * 3, GL_UNSIGNED_INT, 0);

	// �����
	glBindVertexArray(0);
	meshShader.UnbindProgram();

	// ��������ϵ
	coordinateShader.BindProgram();	// ���������shader
	coordinateShader.setUniformMat4(std::string("projection"), projection);
	coordinateShader.setUniformMat4(std::string("view"), view);
	coordinateShader.setUniformMat4(std::string("model"), model);
	glBindVertexArray(coordinateSystemVAO); // ��VAO�����

	glLineWidth(3.0f);
	glDrawArrays(GL_LINES, 0, 34);	// box��54��Ԫ�أ������߶�

	// �����
	glBindVertexArray(0);
	coordinateShader.UnbindProgram();

	// �����ύ����ɫ���壨����һ��������GLFW����ÿһ��������ɫֵ�Ĵ󻺳壩��������һ�����б��������ƣ����ҽ�����Ϊ�����ʾ����Ļ��
	glfwSwapBuffers(window);
	glfwPollEvents();

	// �ͷ���Դ
	glDeleteVertexArrays(1, &GeometryVAO);
	glDeleteBuffers(1, &GeometryVBO);
	glDeleteBuffers(1, &GeometryIBO);
}



void SparseSurfelFusion::DrawMesh::initialCoordinateSystem()
{
	std::vector<float> pvalues;			// ������	

	// Live���еĵ㣬�鿴�м���̵Ķ�����ɫ��
	const std::string coordinate_vert_path = SHADER_PATH_PREFIX + std::string("CoordinateSystemShader.vert");
	// Live���еĵ㣬�鿴�м���̵�Ƭ����ɫ��
	const std::string coordinate_frag_path = SHADER_PATH_PREFIX + std::string("CoordinateSystemShader.frag");
	coordinateShader.Compile(coordinate_vert_path, coordinate_frag_path);
	glGenVertexArrays(1, &coordinateSystemVAO);	// ����VAO
	glGenBuffers(1, &coordinateSystemVBO);		// ����VBO
	const unsigned int Num = sizeof(box) / sizeof(box[0]);
	for (int i = 0; i < Num; i++) {
		pvalues.push_back(box[i][0]);
		pvalues.push_back(box[i][1]);
		pvalues.push_back(box[i][2]);
	}
	//std::cout << "Num = " << Num << "     pvalues.size() = " << pvalues.size() << std::endl;
	glBindVertexArray(coordinateSystemVAO);
	glBindBuffer(GL_ARRAY_BUFFER, coordinateSystemVBO);
	GLsizei bufferSize = sizeof(GLfloat) * pvalues.size();		// float���ݵ�����
	glBufferData(GL_ARRAY_BUFFER, bufferSize, pvalues.data(), GL_DYNAMIC_DRAW);	// ��̬���ƣ�Ŀǰֻ���ȿ��ٸ���С

	// λ��
	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(GLfloat), (void*)(0 * sizeof(GLfloat)));	// ����VAO������
	glEnableVertexAttribArray(0);	// layout (location = 0)
	// ��ɫ
	glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(GLfloat), (void*)(3 * sizeof(GLfloat)));		// ����VAO������
	glEnableVertexAttribArray(1);	// layout (location = 1)

	glBindBuffer(GL_ARRAY_BUFFER, 0);			// ���VBO
	glBindVertexArray(0);						// ���VAO
}



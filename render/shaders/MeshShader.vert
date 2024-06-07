#version 460 core

layout (location = 0) in vec3 pos;      // 位置变量的属性位置值为 0 
layout (location = 1) in vec3 normal;   // 法向量变量的属性位置值为 1

uniform mat4 model;         // 模型矩阵
uniform mat4 view;          // 视角矩阵
uniform mat4 projection;    // 投影矩阵

out vec3 Normal;            // 法线
out vec3 FragPos;           // 在世界空间中的顶点位置

void main()
{
    FragPos = vec3(model * vec4(pos, 1.0));
    Normal = normal;
    gl_Position = projection * view * vec4(FragPos, 1.0);        // 顶点的坐标(前面已经有左乘的FragPos, 此处无需计算)
}
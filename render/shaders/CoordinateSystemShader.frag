#version 460 core

in vec3 color;

out vec4 FragColor; // 输出颜色

void main()
{
    FragColor = vec4(color.x, color.y, color.z, 1.0f); // 片段着色器设置顶点颜色
} 

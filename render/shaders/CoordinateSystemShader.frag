#version 460 core

in vec3 color;

out vec4 FragColor; // �����ɫ

void main()
{
    FragColor = vec4(color.x, color.y, color.z, 1.0f); // Ƭ����ɫ�����ö�����ɫ
} 

#version 460

layout (location = 0) in vec3 coordinatePos;    // ��������ϵ�����ؼ������ά����
layout (location = 1) in vec3 coordinateColor;  // ��������ϵ�����ؼ������ɫ����

out vec3 color;

uniform mat4 model;         // ģ�;���
uniform mat4 view;          // �ӽǾ���
uniform mat4 projection;    // ͶӰ����

void main()
{
    color = coordinateColor;
    highp vec4 normalizedVertexPos = projection * view * model * vec4(coordinatePos.xyz, 1.0);        // ����ת�����������ϵ�ڣ�����һ����[-1,1]��Χ�ڣ��Ա�OpenGL��ʾ
    gl_Position = normalizedVertexPos;      // ʹ������ת����
}
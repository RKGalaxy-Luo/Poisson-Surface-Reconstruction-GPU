#version 460 core

in vec3 Normal;             // ����(�Ѿ���һ��)
in vec3 FragPos;            // ������ռ��еĶ���λ��
in vec3 Color;              // ������ɫ

out vec4 FragColor;         // ������ɫ

uniform vec3 lightPos;      // ��Դλ��
uniform vec3 viewPos;       // �ӽ�λ��
uniform vec3 lightColor;    // �����ɫ

void main()
{
    // ������
    float ambientStrength = 0.1;                                    // ������������
    vec3 ambient = ambientStrength * lightColor;                    // ������ɫ
  	
    // ������ 
    vec3 norm = Normal;                                             // ���������ٹ�һ��
    vec3 lightDir = normalize(lightPos - FragPos);                  // ���Դ�䵽��ǰ�����������һ��
    float diff = max(dot(norm, lightDir), 0.0);                     // ������ģ�ͼ��㷴��ֵ
    vec3 diffuse = diff * lightColor;                               // ����ǰ������ֵ���Թ�����ɫ(ʲô��ɫ�Ĺ�������ȡʲô��ɫ�ķ���)
    
    // ���淴��
    float specularStrength = 0.5;                                   // ����ǿ��
    vec3 viewDir = normalize(viewPos - FragPos);                    // �ӽ��붥��λ�õ�������һ��
    vec3 halfwayDir = normalize(lightDir + viewDir);                // ��Blinn-Phongģ�͡�
    float spec = pow(max(dot(viewDir, halfwayDir), 0.0), 32);       // 32�Ƿ���ȣ�����ķ����Խ�ߣ�����������Խǿ��ɢ���Խ�٣��߹��ͻ�ԽС
    // vec3 reflectDir = reflect(-lightDir, norm);                  // ��Phongģ�͡�-lightDir�Ƿ����·, reflectDir��������
    // float spec = pow(max(dot(viewDir, reflectDir), 0.0), 32);    // 32�Ƿ���ȣ�����ķ����Խ�ߣ�����������Խǿ��ɢ���Խ�٣��߹��ͻ�ԽС
    vec3 specular = specularStrength * spec * lightColor;           // ���㾵�淴�䲢���Թ���
        
    vec3 result = (ambient + diffuse + specular) * Color;           // (������ + ������ + ���淴��) * ������ɫ
    FragColor = vec4(result, 1.0);
} 


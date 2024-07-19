#version 460 core

in vec3 Normal;             // 法线(已经归一化)
in vec3 FragPos;            // 在世界空间中的顶点位置
in vec3 Color;              // 顶点颜色

out vec4 FragColor;         // 顶点颜色

uniform vec3 lightPos;      // 光源位置
uniform vec3 viewPos;       // 视角位置
uniform vec3 lightColor;    // 光的颜色

void main()
{
    // 环境光
    float ambientStrength = 0.1;                                    // 常量环境因子
    vec3 ambient = ambientStrength * lightColor;                    // 光照颜色
  	
    // 漫反射 
    vec3 norm = Normal;                                             // 法线无需再归一化
    vec3 lightDir = normalize(lightPos - FragPos);                  // 点光源射到当前顶点的向量归一化
    float diff = max(dot(norm, lightDir), 0.0);                     // 漫反射模型计算反射值
    vec3 diffuse = diff * lightColor;                               // 将当前慢反射值乘以光照颜色(什么颜色的光照则提取什么颜色的分量)
    
    // 镜面反射
    float specularStrength = 0.5;                                   // 镜面强度
    vec3 viewDir = normalize(viewPos - FragPos);                    // 视角与顶点位置的向量归一化
    vec3 halfwayDir = normalize(lightDir + viewDir);                // 【Blinn-Phong模型】
    float spec = pow(max(dot(viewDir, halfwayDir), 0.0), 32);       // 32是反光度，物体的反光度越高，反射光的能力越强，散射得越少，高光点就会越小
    // vec3 reflectDir = reflect(-lightDir, norm);                  // 【Phong模型】-lightDir是反向光路, reflectDir反射向量
    // float spec = pow(max(dot(viewDir, reflectDir), 0.0), 32);    // 32是反光度，物体的反光度越高，反射光的能力越强，散射得越少，高光点就会越小
    vec3 specular = specularStrength * spec * lightColor;           // 计算镜面反射并乘以光照
        
    vec3 result = (ambient + diffuse + specular) * Color;           // (环境光 + 漫反射 + 镜面反射) * 物体颜色
    FragColor = vec4(result, 1.0);
} 


# Poisson Reconstruction Base on GPU 泊松曲面重建
**Reconstruct surface using point clouds and it's a GPU-accelerated implementation**

本项目基于论文[《Poisson Surface Reconstruction》](https://hhoppe.com/poissonrecon.pdf)和[《Data-Parallel Octrees for Surface Reconstruction》](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=5473223)的原理进行实现，同时参考开源项目[PoissonRecon_GPU](https://github.com/DavidXu-JJ/PoissonRecon_GPU)

**算法将使用CUDA实现对泊松曲面重建算法的加速**

​	①用CUDA多流并行进行优化，同时将Thrust库替换成高性能运算cub库。

​	②同时加入了内存管理机制，优化了内存的开辟与释放，将部分可预见内存预先开辟，并将不可预见并且内存消耗巨大的显存设置为临时内存，即时释放。

​	③加入了OpenGL渲染管线，即可以实现将点云实现实时三角网格化并渲染。

​	④将计算得到的网格直接映射到在CUDA中注册的OpenGL资源中，减少了渲染过程中：Device --> Host --> Device数据传输时间。

**实验环境**

​	Window11，VS2022，CUDA12.0（CUDA11.6及以上应该均可），PCL1.12.1，VTK9.3，glad4.6.0（OpenGL3.3.0及以上均可），glm0.9.9.7

**实验细节**

​	使用RTX 2050 Laptop 4GB处理Bunny数据集并将Octree深度设置为7，最大点云数量阈值设置为200k，最大网格三角面元数量设置为600k，GPU显存运行峰值为3.3GB（包括OpenGL渲染管线着色器消耗），循环运行500次（其中包括OpenGL渲染管线运行的时间）。

**实验结论**

​	【无颜色】循环运行500次平均时间为272.911ms。

​	【有颜色】循环运行500次平均时间为300.717ms。

**实验效果**

![](OutputResult/Result.gif)

![](OutputResult/ColoredResult.gif)

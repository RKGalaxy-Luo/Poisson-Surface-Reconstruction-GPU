#include <base/ThreadPool.h>
#include <conio.h>
#include <chrono>
#include <mesh/PoissonReconstruction.h>

using namespace SparseSurfelFusion;

int main(int argc, char** argv) {

	std::shared_ptr<ThreadPool> pool = std::make_shared<ThreadPool>(MAX_THREADS);
	PoissonReconstruction PoissonRecon;

	//PoissonRecon.readPLYFile(PlyDataPath);

	PoissonRecon.readPCDFile(PcdDataPath);
	size_t Frame = 0;
	double averageRuntime = 0;
	while (Frame < 500) {
		if (_kbhit()) { // 如果有按键按下，则_kbhit()函数返回真
			int ch = _getch();// 使用_getch()函数获取按下的键值
			if (ch == ESC_KEY) {
				break;
			}
		}
		printf("··········································  第 %lld 帧  ··········································  \n", Frame);
		auto start = std::chrono::high_resolution_clock::now();						// 记录开始时间点
		CoredVectorMeshData mesh;	// 网格
		PoissonRecon.SolvePoissionReconstructionMesh(PoissonRecon.getDenseSurfel(), mesh);
		PoissonRecon.DrawRebuildMesh(mesh);
		mesh.clearAllContainer();
		auto end = std::chrono::high_resolution_clock::now();							// 记录结束时间点
		std::chrono::duration<double, std::milli> duration = end - start;				// 计算执行时间（以ms为单位）
		std::cout << "算法运行整体时间: " << duration.count() << " ms" << std::endl;		// 输出
		std::cout << std::endl;
		std::cout << "-----------------------------------------------------" << std::endl;	// 输出
		std::cout << std::endl;
		averageRuntime += duration.count();
		Frame++;
	}
	std::cout << "算法运行平均时间: " << averageRuntime * 1.0f / Frame << " ms" << std::endl;		// 输出
	exit(-1);
}
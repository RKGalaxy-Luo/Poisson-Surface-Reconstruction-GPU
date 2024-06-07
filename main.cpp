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
		if (_kbhit()) { // ����а������£���_kbhit()����������
			int ch = _getch();// ʹ��_getch()������ȡ���µļ�ֵ
			if (ch == ESC_KEY) {
				break;
			}
		}
		printf("������������������������������������������������������������������������������������  �� %lld ֡  ������������������������������������������������������������������������������������  \n", Frame);
		auto start = std::chrono::high_resolution_clock::now();						// ��¼��ʼʱ���
		CoredVectorMeshData mesh;	// ����
		PoissonRecon.SolvePoissionReconstructionMesh(PoissonRecon.getDenseSurfel(), mesh);
		PoissonRecon.DrawRebuildMesh(mesh);
		mesh.clearAllContainer();
		auto end = std::chrono::high_resolution_clock::now();							// ��¼����ʱ���
		std::chrono::duration<double, std::milli> duration = end - start;				// ����ִ��ʱ�䣨��msΪ��λ��
		std::cout << "�㷨��������ʱ��: " << duration.count() << " ms" << std::endl;		// ���
		std::cout << std::endl;
		std::cout << "-----------------------------------------------------" << std::endl;	// ���
		std::cout << std::endl;
		averageRuntime += duration.count();
		Frame++;
	}
	std::cout << "�㷨����ƽ��ʱ��: " << averageRuntime * 1.0f / Frame << " ms" << std::endl;		// ���
	exit(-1);
}
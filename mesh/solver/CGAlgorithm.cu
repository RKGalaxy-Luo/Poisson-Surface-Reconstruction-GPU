/*****************************************************************//**
 * \file   CGAlgorithm.cu
 * \brief  梯度下降法cuda实现
 * 
 * \author LUOJIAXUAN
 * \date   May 28th 2024
 *********************************************************************/
#include "CGAlgorithm.cuh"

extern "C" __global__ void gpuConjugateGradient(int* I, int* J, float* val, float* x, float* Ax, float* p, float* r, double* dot_result, int nnz, int N, float tol)
{
    cg::thread_block cta = cg::this_thread_block();
    cg::grid_group grid = cg::this_grid();

    int max_iter = 10000;

    float alpha = 1.0;
    float alpham1 = -1.0;
    float r0 = 0.0, r1, b, a, na;

    SparseSurfelFusion::device::gpuSpMV(I, J, val, nnz, N, alpha, x, Ax, cta, grid);

    cg::sync(grid);

    SparseSurfelFusion::device::gpuSaxpy(Ax, r, alpham1, N, grid);

    cg::sync(grid);

    SparseSurfelFusion::device::gpuDotProduct(r, r, dot_result, N, cta, grid);

    cg::sync(grid);

    r1 = *dot_result;

    int k = 1;
    while (r1 > tol * tol && k <= max_iter) {
        if (k > 1) {
            b = r1 / r0;
            SparseSurfelFusion::device::gpuScaleVectorAndSaxpy(r, p, alpha, b, N, grid);
        }
        else {
            SparseSurfelFusion::device::gpuCopyVector(r, p, N, grid);
        }

        cg::sync(grid);

        SparseSurfelFusion::device::gpuSpMV(I, J, val, nnz, N, alpha, p, Ax, cta, grid);

        if (threadIdx.x == 0 && blockIdx.x == 0) *dot_result = 0.0;

        cg::sync(grid);

        SparseSurfelFusion::device::gpuDotProduct(p, Ax, dot_result, N, cta, grid);

        cg::sync(grid);

        a = r1 / *dot_result;

        SparseSurfelFusion::device::gpuSaxpy(p, x, a, N, grid);


        na = -a;
        SparseSurfelFusion::device::gpuSaxpy(Ax, r, na, N, grid);

        r0 = r1;

        cg::sync(grid);
        if (threadIdx.x == 0 && blockIdx.x == 0) *dot_result = 0.0;

        cg::sync(grid);

        SparseSurfelFusion::device::gpuDotProduct(r, r, dot_result, N, cta, grid);

        cg::sync(grid);

        r1 = *dot_result;

        k++;
    }
}

__device__ void SparseSurfelFusion::device::gpuSpMV(int* I, int* J, float* val, int nnz, int num_rows, float alpha, float* inputVecX, float* outputVecY, cg::thread_block& cta, const cg::grid_group& grid)
{
    for (int i = grid.thread_rank(); i < num_rows; i += grid.size()) {
        int row_elem = I[i];
        int next_row_elem = I[i + 1];
        int num_elems_this_row = next_row_elem - row_elem;

        float output = 0.0;
        for (int j = 0; j < num_elems_this_row; j++) {
            // I or J or val arrays - can be put in shared memory
            // as the access is random and reused in next calls of gpuSpMV function.
            output += alpha * val[row_elem + j] * inputVecX[J[row_elem + j]];
        }
        outputVecY[i] = output;
    }
}

__device__ void SparseSurfelFusion::device::gpuSaxpy(float* x, float* y, float a, int size, const cg::grid_group& grid)
{
    for (int i = grid.thread_rank(); i < size; i += grid.size()) {
        y[i] = a * x[i] + y[i];
    }
}

__device__ void SparseSurfelFusion::device::gpuDotProduct(float* vecA, float* vecB, double* result, int size, const cg::thread_block& cta, const cg::grid_group& grid)
{
    extern __shared__ double tmp[];

    double temp_sum = 0.0;
    for (int i = grid.thread_rank(); i < size; i += grid.size()) {
        temp_sum += static_cast<double>(vecA[i] * vecB[i]);
    }

    cg::thread_block_tile<32> tile32 = cg::tiled_partition<32>(cta);

    temp_sum = cg::reduce(tile32, temp_sum, cg::plus<double>());

    if (tile32.thread_rank() == 0) {
        tmp[tile32.meta_group_rank()] = temp_sum;
    }

    cg::sync(cta);

    if (tile32.meta_group_rank() == 0) {
        temp_sum = tile32.thread_rank() < tile32.meta_group_size() ? tmp[tile32.thread_rank()] : 0.0;
        temp_sum = cg::reduce(tile32, temp_sum, cg::plus<double>());

        if (tile32.thread_rank() == 0) {
            atomicAdd(result, temp_sum);
        }
    }
}

__device__ void SparseSurfelFusion::device::gpuCopyVector(float* srcA, float* destB, int size, const cg::grid_group& grid)
{
    for (int i = grid.thread_rank(); i < size; i += grid.size()) {
        destB[i] = srcA[i];
    }
}

__device__ void SparseSurfelFusion::device::gpuScaleVectorAndSaxpy(const float* x, float* y, float a, float scale, int size, const cg::grid_group& grid)
{
    for (int i = grid.thread_rank(); i < size; i += grid.size()) {
        y[i] = a * x[i] + scale * y[i];
    }
}




__global__ void SparseSurfelFusion::device::gpuGetTestSummary(int* I, int* J, float* val, float* x, float* rhs, int* err, int num)
{
    const unsigned int idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (idx >= num)	return;	// 0层邻居已经初始化  
    float rsum = 0.0f;

    for (int i = I[idx]; i < I[idx + 1]; i++) {
        rsum += val[i] * x[J[i]];
    }
    int diff = fabsf(rsum - rhs[idx]) * 1e10;
    atomicMax(err, diff);
}

void SparseSurfelFusion::genTridiag(int* I, int* J, float* val, int N, int nz)
{
    I[0] = 0, J[0] = 0, J[1] = 1;
    val[0] = static_cast<float>(rand()) / RAND_MAX + 10.0f;
    val[1] = static_cast<float>(rand()) / RAND_MAX;
    int start;

    for (int i = 1; i < N; i++) {
        if (i > 1) {
            I[i] = I[i - 1] + 3;
        }
        else {
            I[1] = 2;
        }

        start = (i - 1) * 3 + 2;
        J[start] = i - 1;
        J[start + 1] = i;

        if (i < N - 1) {
            J[start + 2] = i + 1;
        }

        val[start] = val[start - 1];
        val[start + 1] = static_cast<float>(rand()) / RAND_MAX + 10.0f;

        if (i < N - 1) {
            val[start + 2] = static_cast<float>(rand()) / RAND_MAX;
        }
    }

    I[N] = nz;
}

void SparseSurfelFusion::cpuSpMV(int* I, int* J, float* val, int nnz, int num_rows, float alpha, float* inputVecX, float* outputVecY)
{
    for (int i = 0; i < num_rows; i++) {
        int num_elems_this_row = I[i + 1] - I[i];

        float output = 0.0;
        for (int j = 0; j < num_elems_this_row; j++) {
            output += alpha * val[I[i] + j] * inputVecX[J[I[i] + j]];
        }
        outputVecY[i] = output;
    }
    return;
}

double SparseSurfelFusion::dotProduct(float* vecA, float* vecB, int size)
{
    double result = 0.0;
    for (int i = 0; i < size; i++) {
        result = result + (vecA[i] * vecB[i]);
    }
    return result;
}

void SparseSurfelFusion::scaleVector(float* vec, float alpha, int size)
{
    for (int i = 0; i < size; i++) {
        vec[i] = alpha * vec[i];
    }
}

void SparseSurfelFusion::saxpy(float* x, float* y, float a, int size)
{
    for (int i = 0; i < size; i++) {
        y[i] = a * x[i] + y[i];
    }
}

void SparseSurfelFusion::cpuConjugateGrad(int* I, int* J, float* val, float* x, float* Ax, float* p, float* r, int nnz, int N, float tol)
{
    int max_iter = 10000;

    float alpha = 1.0;
    float alpham1 = -1.0;
    float r0 = 0.0, b, a, na;

    cpuSpMV(I, J, val, nnz, N, alpha, x, Ax);
    saxpy(Ax, r, alpham1, N);

    float r1 = dotProduct(r, r, N);

    int k = 1;

    while (r1 > tol * tol && k <= max_iter) {
        if (k > 1) {
            b = r1 / r0;
            scaleVector(p, b, N);

            saxpy(r, p, alpha, N);
        }
        else {
            for (int i = 0; i < N; i++) p[i] = r[i];
        }

        cpuSpMV(I, J, val, nnz, N, alpha, p, Ax);

        float dot = dotProduct(p, Ax, N);
        a = r1 / dot;

        saxpy(p, x, a, N);
        na = -a;
        saxpy(Ax, r, na, N);

        r0 = r1;
        r1 = dotProduct(r, r, N);

        printf("\nCPU code iteration = %3d, residual = %e\n", k, sqrt(r1));
        k++;
    }
}

bool SparseSurfelFusion::areAlmostEqual(float a, float b, float maxRelDiff)
{
    float diff = fabsf(a - b);
    float abs_a = fabsf(a);
    float abs_b = fabsf(b);
    float largest = abs_a > abs_b ? abs_a : abs_b;

    if (diff <= largest * maxRelDiff) {
        return true;
    }
    else {
        printf("最大真实相差(diff) = %.8e\n", maxRelDiff);
        printf("diff %.8e > largest * maxRelDiff %.8e 因此 %.8e and %.8e 不相同 \n", diff, largest * maxRelDiff, a, b);
        return false;
    }
}

void SparseSurfelFusion::solverCG_DeviceToDevice(const int& N, const int& nz, int* I, int* J, float* val, float* rhs, float* x, cudaStream_t stream)
{
    const float tol = 1e-5f;
    double r1 = 0;      // 记录残差
    float* r = NULL;    // 运行 CG 算法的临时内存
    float* p = NULL;    // 运行 CG 算法的临时内存
    float* Ax = NULL;   // 运行 CG 算法的临时内存
    //cudaEvent_t start, stop;

    //printf("开始 [%s]...\n", "共轭梯度多块计算 (Conjugate Gradient MultiBlock CG)");

    // This will pick the best possible CUDA capable device
    cudaDeviceProp deviceProp;
    int devID = 0;
    CHECKCUDA(cudaGetDeviceProperties(&deviceProp, devID));

    if (!deviceProp.managedMemory) {
        // This sample requires being run on a device that supports Unified Memory
        fprintf(stderr, "此设备不支持统一内存 \n");
        exit(EXIT_WAIVED);
    }

    // This sample requires being run on a device that supports Cooperative Kernel Launch
    if (!deviceProp.cooperativeLaunch) {
        printf( "\n选择的 GPU (%d) 不支持协作核函数启动 (Cooperative Kernel Launch), 放弃运行\n", devID);
        exit(EXIT_WAIVED);
    }

    // Statistics about the GPU device
    //printf("> GPU 设备有 %d 个流处理器, 流处理器计算能力 %d.%d \n\n", deviceProp.multiProcessorCount, deviceProp.major, deviceProp.minor);

    double* dot_result = NULL;
    CHECKCUDA(cudaMallocAsync(reinterpret_cast<void**>(&dot_result), sizeof(double), stream));
    CHECKCUDA(cudaMemsetAsync(dot_result, 0.0, sizeof(double), stream));

    // 运行 CG 算法的临时内存
    CHECKCUDA(cudaMallocAsync(reinterpret_cast<void**>(&r) , N * sizeof(float), stream));
    CHECKCUDA(cudaMallocAsync(reinterpret_cast<void**>(&p) , N * sizeof(float), stream));
    CHECKCUDA(cudaMallocAsync(reinterpret_cast<void**>(&Ax), N * sizeof(float), stream));

#if ENABLE_CPU_DEBUG_CODE
    float* Ax_cpu = reinterpret_cast<float*>(malloc(sizeof(float) * N));
    float* r_cpu = reinterpret_cast<float*>(malloc(sizeof(float) * N));
    float* p_cpu = reinterpret_cast<float*>(malloc(sizeof(float) * N));
    float* x_cpu = reinterpret_cast<float*>(malloc(sizeof(float) * N));

    for (int i = 0; i < N; i++) {
        r_cpu[i] = 1.0;
        Ax_cpu[i] = x_cpu[i] = 0.0;
    }

#endif

    CHECKCUDA(cudaMemcpyAsync(r, rhs, sizeof(float) * N, cudaMemcpyDeviceToDevice, stream));
    CHECKCUDA(cudaMemsetAsync(x, 0.0, sizeof(float) * N, stream));

    void* kernelArgs[] = {
            (void*)&I , (void*)&J, (void*)&val, (void*)&x,
            (void*)&Ax, (void*)&p, (void*)&r  , (void*)&dot_result,
            (void*)&nz, (void*)&N, (void*)&tol,
    };

    int sMemSize = sizeof(double) * ((THREADS_PER_BLOCK / 32) + 1);
    int numBlocksPerSm = 0;
    int numThreads = THREADS_PER_BLOCK;

    CHECKCUDA(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&numBlocksPerSm, gpuConjugateGradient, numThreads, sMemSize));

    int numSms = deviceProp.multiProcessorCount;
    dim3 dimGrid(numSms * numBlocksPerSm, 1, 1);
    dim3 dimBlock(THREADS_PER_BLOCK, 1, 1);
    CHECKCUDA(cudaLaunchCooperativeKernel((void*)gpuConjugateGradient, dimGrid, dimBlock, kernelArgs, sMemSize, stream));

    CHECKCUDA(cudaMemcpyAsync(&r1, dot_result, sizeof(double), cudaMemcpyDeviceToHost, stream));

    //r1 = *dot_result;
#ifdef CHECK_MESH_BUILD_TIME_COST
    printf("残差 = %e  ", sqrt(r1));    // 科学计数法输出
#endif // CHECK_MESH_BUILD_TIME_COST

#if ENABLE_CPU_DEBUG_CODE
    cpuConjugateGrad(I, J, val, x_cpu, Ax_cpu, p_cpu, r_cpu, nz, N, tol);
#endif

    int* err = NULL;
    CHECKCUDA(cudaMallocAsync(reinterpret_cast<void**>(&err), sizeof(int), stream));
    CHECKCUDA(cudaMemsetAsync(err, 0, sizeof(int), stream));

    int err_Int_Host;
    dim3 block(128);
    dim3 grid(pcl::gpu::divUp(N, block.x));
    device::gpuGetTestSummary << <grid, block, 0, stream >> > (I, J, val, x, rhs, err, N);
    CHECKCUDA(cudaMemcpyAsync(&err_Int_Host, err, sizeof(int), cudaMemcpyDeviceToHost, stream));
    CHECKCUDA(cudaStreamSynchronize(stream));
    float errHost = err_Int_Host / 1e10;

    CHECKCUDA(cudaFreeAsync(r, stream));
    CHECKCUDA(cudaFreeAsync(p, stream));
    CHECKCUDA(cudaFreeAsync(Ax, stream));
    CHECKCUDA(cudaFreeAsync(dot_result, stream));
    CHECKCUDA(cudaFreeAsync(err, stream));

#if ENABLE_CPU_DEBUG_CODE
    free(Ax_cpu);
    free(r_cpu);
    free(p_cpu);
    free(x_cpu);
#endif

#ifdef CHECK_MESH_BUILD_TIME_COST
    printf("误差量 = %e \n", errHost);
#endif // CHECK_MESH_BUILD_TIME_COST
}

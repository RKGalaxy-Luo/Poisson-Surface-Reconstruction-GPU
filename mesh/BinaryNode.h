/*****************************************************************//**
 * \file   BinaryNode.h
 * \brief  配适B样条基函数计算的节点
 * 
 * \author LUOJIAXUAN
 * \date   May 18th 2024
 *********************************************************************/
#pragma once
#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>

namespace SparseSurfelFusion {
	template<class Real>
	class BinaryNode {
    public:
        /**     2^depth                                         */
        __host__ __device__ static inline int CenterCount(int depth) { return 1 << depth; }
        /**     2^0 + 2^1 + 2^maxDepth = 2^(maxDepth+1) - 1     */
        __host__ __device__ static inline int CumulativeCenterCount(int maxDepth) { return (1 << (maxDepth + 1)) - 1; }
        __host__ __device__ static inline int Index(int depth, int offSet) { return (1 << depth) + offSet - 1; }
        __host__ __device__ static inline int CornerIndex(int maxDepth, int depth, int offSet, int forwardCorner)
        {
            return (offSet + forwardCorner) << (maxDepth - depth);
        }
        __host__ __device__ static inline Real CornerIndexPosition(int index, int maxDepth)
        {
            return Real(index) / (1 << maxDepth);
        }
        __host__ __device__ static inline Real Width(int depth)
        {
            return Real(1.0 / (1 << depth));
        }
        __host__ __device__ static inline void CenterAndWidth(int depth, int offset, Real& center, Real& width)
        {
            width = Real(1.0 / (1 << depth));
            center = Real((0.5 + offset) * width);
        }
        __host__ __device__ static inline void CenterAndWidth(int idx, Real& center, Real& width)
        {
            int depth, offset;
            DepthAndOffset(idx, depth, offset);
            CenterAndWidth(depth, offset, center, width);
        }
        __host__ __device__ static inline void DepthAndOffset(int idx, int& depth, int& offset)
        {
            int i = idx + 1;
            depth = -1;
            while (i) {
                i >>= 1;
                depth++;
            }
            offset = (idx + 1) - (1 << depth);
        }
	};
}

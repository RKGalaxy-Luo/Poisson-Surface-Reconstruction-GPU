/*****************************************************************//**
 * \file   OctNode.cuh
 * \brief  �㷨�����ݽṹ
 * 
 * \author LUOJIAXUAN
 * \date   May 2nd 2024
 *********************************************************************/
#pragma once
#include "Geometry.h"

namespace SparseSurfelFusion{
    /**
     * \brief ��¼�˲����ڵ���������ͣ���С 276 Bytes.
     */
    class OctNode {
    public:
        int key;            // �ڵ�ļ�key
        int pidx;           // �ڵ���SortedArray�еĵ�һ��Ԫ�ص�index
        int pnum;           // �뵱ǰ�ڵ�key��ͬ�ĳ��ܵ������
        int parent;         // 1�����ڵ�
        int children[8];    // 8�����ӽڵ�
        int neighs[27];     // 27���ھӽڵ�
        // record the start in maxDepth NodeArray the first node at maxDepth is index 0
        int didx;           // ��ǰ�ڵ����������ΪD���ӽڵ��һ���㣬idx��С�ĵ�
        int dnum;           // ��ǰ�ڵ����������ΪD���ӽڵ��������������Ч�����Ч��

        int vertices[8];    // ����¼��idxƫ��һλ?����¼�ڵ�Ķ�����Ϣ�������Ϣ����VertexArray�����У�����ֻ�Ǽ�¼�Žڵ���VertexArray�е�index

        // (real idx) + 1,
        // idx start from (0 + 1)
        int edges[12];

        // (real idx) + 1
        // idx start from (0 + 1)
        int faces[6];

        int hasTriangle;
        int hasIntersection;
    };

    /**
     * \brief OctNode�ļ򻯰棬ȥ����face�����Ǳ���ཻ���.
     */
    class EasyOctNode {
    public:
        int key;
        int parent;
        int children[8];
        int neighs[27];

        // (real idx) + 1,
        // idx start from (0 + 1)
        // encode the vertices idx?
        int vertices[8];

        // (real idx) + 1,
        // idx start from (0 + 1)
        int edges[12];

        __device__ EasyOctNode& operator = (const OctNode& n) {
            key = n.key;
            parent = n.parent;
#pragma unroll  // ��ѭ������չ�������Ż�������ͨ����ѭ�����еĴ��븴�ƶ��������ѭ���������Ӷ���߳����ִ���ٶ�
            for (int i = 0; i < 8; ++i) {
                children[i] = n.children[i];
                vertices[i] = n.vertices[i];
            }
#pragma unroll
            for (int i = 0; i < 27; ++i) {
                neighs[i] = n.neighs[i];
            }
#pragma unroll
            for (int i = 0; i < 12; ++i) {
                edges[i] = n.edges[i];
            }
        }
    };

    /**
     * \brief ����vertex�������ͣ���С 56 Bytes.
     */
    class VertexNode {
    public:
        Point3D<float> pos = Point3D<float>(0.0f, 0.0f, 0.0f);  // ��ǰVertex��λ��
        int ownerNodeIdx = 0;                                   // ���Vertex����NodeArray����һ���ڵ�(index)��ͬһ���ڵ����ӵ�ж��vertex�� ��һ��vertexֻ����һ���ڵ�index
        int vertexKind = 0;                                     // �����Ӧ��λ��index�������˳���ǣ���������Ϊ�ο���x��ǰ����y�����ң�z���µ��ϡ�
        int depth = 0;                                          // ���Vertex��Ӧ�Ľڵ����
        int nodes[8] = { 0, 0, 0, 0, 0, 0, 0, 0 };              // ��¼��ǰ�����Owner�ھ��У������vertex���ڵĽڵ�index(������ڵ����ڣ���������ھӽڵ�����Ĳ�����1һ���ڵ���)
    };

    class EdgeNode {
    public:
        // int orientation;
        // int off[2];
        int edgeKind = 0;
        int ownerNodeIdx = 0;
        int nodes[4] = { 0, 0, 0, 0 };
    };

    class FaceNode {
    public:
        int faceKind = -1;
        int ownerNodeIdx = -1;
        int hasParentFace = -1;
        int nodes[2] = { -1, -1 };
    };
}


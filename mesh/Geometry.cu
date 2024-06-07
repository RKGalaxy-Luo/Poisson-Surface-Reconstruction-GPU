/*****************************************************************//**
 * \file   Geometry.cu
 * \brief  算法基础数据结构及方法实现
 * 
 * \author LUOJIAXUAN
 * \date   May 2nd 2024
 *********************************************************************/
#include <mesh/Geometry.h>

////////////////////
// Basic Operator //
////////////////////
template<class T>
__host__ __device__ double SparseSurfelFusion::SquareLength(const Point3D<T>& p) {
	return p.coords[0] * p.coords[0] + p.coords[1] * p.coords[1] + p.coords[2] * p.coords[2]; 
}

template<class T>
double SparseSurfelFusion::Length(const Point3D<T>& p) {
	return sqrt(SquareLength(p)); 
}

template<class T>
__host__ __device__ double SparseSurfelFusion::SquareDistance(const Point3D<T>& p1, const Point3D<T>& p2) {
	return (p1.coords[0] - p2.coords[0]) * (p1.coords[0] - p2.coords[0]) + (p1.coords[1] - p2.coords[1]) * (p1.coords[1] - p2.coords[1]) + (p1.coords[2] - p2.coords[2]) * (p1.coords[2] - p2.coords[2]);
}

template<class T>
double SparseSurfelFusion::Distance(const Point3D<T>& p1, const Point3D<T>& p2) {
    return sqrt(SquareDistance(p1, p2)); 
}

template <class T>
void SparseSurfelFusion::CrossProduct(const Point3D<T>& p1, const Point3D<T>& p2, Point3D<T>& p) {
	p.coords[0] = p1.coords[1] * p2.coords[2] - p1.coords[2] * p2.coords[1];
	p.coords[1] = -p1.coords[0] * p2.coords[2] + p1.coords[2] * p2.coords[0];
	p.coords[2] = p1.coords[0] * p2.coords[1] - p1.coords[1] * p2.coords[0];
}

template<class T>
void SparseSurfelFusion::EdgeCollapse(const T& edgeRatio, std::vector<TriangleIndex>& triangles, std::vector< Point3D<T> >& positions, std::vector< Point3D<T> >* normals) {
    int i, j, * remapTable, * pointCount, idx[3];
    Point3D<T> p[3], q[2], c;
    double d[3], a;
    double Ratio = 12.0 / sqrt(3.0);	// (Sum of Squares Length / Area) for and equilateral triangle

    remapTable = new int[positions.size()];
    pointCount = new int[positions.size()];
    for (i = 0; i<int(positions.size()); i++) {
        remapTable[i] = i;
        pointCount[i] = 1;
    }
    for (i = int(triangles.size() - 1); i >= 0; i--) {
        for (j = 0; j < 3; j++) {
            idx[j] = triangles[i].idx[j];
            while (remapTable[idx[j]] < idx[j]) { idx[j] = remapTable[idx[j]]; }
        }
        if (idx[0] == idx[1] || idx[0] == idx[2] || idx[1] == idx[2]) {
            triangles[i] = triangles[triangles.size() - 1];
            triangles.pop_back();
            continue;
        }
        for (j = 0; j < 3; j++) {
            p[j].coords[0] = positions[idx[j]].coords[0] / pointCount[idx[j]];
            p[j].coords[1] = positions[idx[j]].coords[1] / pointCount[idx[j]];
            p[j].coords[2] = positions[idx[j]].coords[2] / pointCount[idx[j]];
        }
        for (j = 0; j < 3; j++) {
            q[0].coords[j] = p[1].coords[j] - p[0].coords[j];
            q[1].coords[j] = p[2].coords[j] - p[0].coords[j];
            d[j] = SquareDistance(p[j], p[(j + 1) % 3]);
        }
        CrossProduct(q[0], q[1], c);
        a = Length(c) / 2;

        if ((d[0] + d[1] + d[2]) * edgeRatio > a * Ratio) {
            // Find the smallest edge
            j = 0;
            if (d[1] < d[j]) { j = 1; }
            if (d[2] < d[j]) { j = 2; }

            int idx1, idx2;
            if (idx[j] < idx[(j + 1) % 3]) {
                idx1 = idx[j];
                idx2 = idx[(j + 1) % 3];
            }
            else {
                idx2 = idx[j];
                idx1 = idx[(j + 1) % 3];
            }
            positions[idx1].coords[0] += positions[idx2].coords[0];
            positions[idx1].coords[1] += positions[idx2].coords[1];
            positions[idx1].coords[2] += positions[idx2].coords[2];
            if (normals) {
                (*normals)[idx1].coords[0] += (*normals)[idx2].coords[0];
                (*normals)[idx1].coords[1] += (*normals)[idx2].coords[1];
                (*normals)[idx1].coords[2] += (*normals)[idx2].coords[2];
            }
            pointCount[idx1] += pointCount[idx2];
            remapTable[idx2] = idx1;
            triangles[i] = triangles[triangles.size() - 1];
            triangles.pop_back();
        }
    }
    int pCount = 0;
    for (i = 0; i<int(positions.size()); i++) {
        for (j = 0; j < 3; j++) { positions[i].coords[j] /= pointCount[i]; }
        if (normals) {
            T l = T(Length((*normals)[i]));
            for (j = 0; j < 3; j++) { (*normals)[i].coords[j] /= l; }
        }
        if (remapTable[i] == i) { // If vertex i is being used
            positions[pCount] = positions[i];
            if (normals) { (*normals)[pCount] = (*normals)[i]; }
            pointCount[i] = pCount;
            pCount++;
        }
    }
    positions.resize(pCount);
    for (i = int(triangles.size() - 1); i >= 0; i--) {
        for (j = 0; j < 3; j++) {
            idx[j] = triangles[i].idx[j];
            while (remapTable[idx[j]] < idx[j]) { idx[j] = remapTable[idx[j]]; }
            triangles[i].idx[j] = pointCount[idx[j]];
        }
        if (idx[0] == idx[1] || idx[0] == idx[2] || idx[1] == idx[2]) {
            triangles[i] = triangles[triangles.size() - 1];
            triangles.pop_back();
        }
    }

    delete[] pointCount;
    delete[] remapTable;
}
template<class T>
void SparseSurfelFusion::TriangleCollapse(const T& edgeRatio, std::vector<TriangleIndex>& triangles, std::vector< Point3D<T> >& positions, std::vector< Point3D<T> >* normals) {
    int i, j, * remapTable, * pointCount, idx[3];
    Point3D<T> p[3], q[2], c;
    double d[3], a;
    double Ratio = 12.0 / sqrt(3.0);	// (Sum of Squares Length / Area) for and equilateral triangle

    remapTable = new int[positions.size()];
    pointCount = new int[positions.size()];
    for (i = 0; i<int(positions.size()); i++) {
        remapTable[i] = i;
        pointCount[i] = 1;
    }
    for (i = int(triangles.size() - 1); i >= 0; i--) {
        for (j = 0; j < 3; j++) {
            idx[j] = triangles[i].idx[j];
            while (remapTable[idx[j]] < idx[j]) { idx[j] = remapTable[idx[j]]; }
        }
        if (idx[0] == idx[1] || idx[0] == idx[2] || idx[1] == idx[2]) {
            triangles[i] = triangles[triangles.size() - 1];
            triangles.pop_back();
            continue;
        }
        for (j = 0; j < 3; j++) {
            p[j].coords[0] = positions[idx[j]].coords[0] / pointCount[idx[j]];
            p[j].coords[1] = positions[idx[j]].coords[1] / pointCount[idx[j]];
            p[j].coords[2] = positions[idx[j]].coords[2] / pointCount[idx[j]];
        }
        for (j = 0; j < 3; j++) {
            q[0].coords[j] = p[1].coords[j] - p[0].coords[j];
            q[1].coords[j] = p[2].coords[j] - p[0].coords[j];
            d[j] = SquareDistance(p[j], p[(j + 1) % 3]);
        }
        CrossProduct(q[0], q[1], c);
        a = Length(c) / 2;

        if ((d[0] + d[1] + d[2]) * edgeRatio > a * Ratio) {
            // Find the smallest edge
            j = 0;
            if (d[1] < d[j]) { j = 1; }
            if (d[2] < d[j]) { j = 2; }

            int idx1, idx2, idx3;
            if (idx[0] < idx[1]) {
                if (idx[0] < idx[2]) {
                    idx1 = idx[0];
                    idx2 = idx[2];
                    idx3 = idx[1];
                }
                else {
                    idx1 = idx[2];
                    idx2 = idx[0];
                    idx3 = idx[1];
                }
            }
            else {
                if (idx[1] < idx[2]) {
                    idx1 = idx[1];
                    idx2 = idx[2];
                    idx3 = idx[0];
                }
                else {
                    idx1 = idx[2];
                    idx2 = idx[1];
                    idx3 = idx[0];
                }
            }
            positions[idx1].coords[0] += positions[idx2].coords[0] + positions[idx3].coords[0];
            positions[idx1].coords[1] += positions[idx2].coords[1] + positions[idx3].coords[1];
            positions[idx1].coords[2] += positions[idx2].coords[2] + positions[idx3].coords[2];
            if (normals) {
                (*normals)[idx1].coords[0] += (*normals)[idx2].coords[0] + (*normals)[idx3].coords[0];
                (*normals)[idx1].coords[1] += (*normals)[idx2].coords[1] + (*normals)[idx3].coords[1];
                (*normals)[idx1].coords[2] += (*normals)[idx2].coords[2] + (*normals)[idx3].coords[2];
            }
            pointCount[idx1] += pointCount[idx2] + pointCount[idx3];
            remapTable[idx2] = idx1;
            remapTable[idx3] = idx1;
            triangles[i] = triangles[triangles.size() - 1];
            triangles.pop_back();
        }
    }
    int pCount = 0;
    for (i = 0; i<int(positions.size()); i++) {
        for (j = 0; j < 3; j++) { positions[i].coords[j] /= pointCount[i]; }
        if (normals) {
            T l = T(Length((*normals)[i]));
            for (j = 0; j < 3; j++) { (*normals)[i].coords[j] /= l; }
        }
        if (remapTable[i] == i) { // If vertex i is being used
            positions[pCount] = positions[i];
            if (normals) { (*normals)[pCount] = (*normals)[i]; }
            pointCount[i] = pCount;
            pCount++;
        }
    }
    positions.resize(pCount);
    for (i = int(triangles.size() - 1); i >= 0; i--) {
        for (j = 0; j < 3; j++) {
            idx[j] = triangles[i].idx[j];
            while (remapTable[idx[j]] < idx[j]) { idx[j] = remapTable[idx[j]]; }
            triangles[i].idx[j] = pointCount[idx[j]];
        }
        if (idx[0] == idx[1] || idx[0] == idx[2] || idx[1] == idx[2]) {
            triangles[i] = triangles[triangles.size() - 1];
            triangles.pop_back();
        }
    }
    delete[] pointCount;
    delete[] remapTable;
}

//////////////
// MeshData //
//////////////

///////////////////
// CoredMeshData //
///////////////////
const int SparseSurfelFusion::CoredMeshData::IN_CORE_FLAG[] = { 1, 2, 4 };

/////////////////////////
// CoredVectorMeshData //
/////////////////////////
SparseSurfelFusion::CoredVectorMeshData::CoredVectorMeshData(void) {
	oocPointIndex = triangleIndex = 0; 
}

void SparseSurfelFusion::CoredVectorMeshData::resetIterator(void) {
	oocPointIndex = triangleIndex = 0; 
}

int SparseSurfelFusion::CoredVectorMeshData::addOutOfCorePoint(const Point3D<float>& p) {
	oocPoints.push_back(p);
	return int(oocPoints.size()) - 1;
}

int SparseSurfelFusion::CoredVectorMeshData::addTriangle(const TriangleIndex& t, const int& coreFlag) {
	TriangleIndex tt;
	if (coreFlag & CoredMeshData::IN_CORE_FLAG[0]) { tt.idx[0] = t.idx[0]; }
	else { tt.idx[0] = -t.idx[0] - 1; }
	if (coreFlag & CoredMeshData::IN_CORE_FLAG[1]) { tt.idx[1] = t.idx[1]; }
	else { tt.idx[1] = -t.idx[1] - 1; }
	if (coreFlag & CoredMeshData::IN_CORE_FLAG[2]) { tt.idx[2] = t.idx[2]; }
	else { tt.idx[2] = -t.idx[2] - 1; }
	triangles.push_back(tt);
	return int(triangles.size()) - 1;
}

int SparseSurfelFusion::CoredVectorMeshData::nextOutOfCorePoint(Point3D<float>& p) {
    if (oocPointIndex < int(oocPoints.size())) {
		p = oocPoints[oocPointIndex++];
		return 1;
	}
	else { return 0; }
}

int SparseSurfelFusion::CoredVectorMeshData::nextTriangle(TriangleIndex& t, int& inCoreFlag) {
	inCoreFlag = 0;
	if (triangleIndex<int(triangles.size())) {
		t = triangles[triangleIndex++];
		if (t.idx[0] < 0) { t.idx[0] = -t.idx[0] - 1; }
		else { inCoreFlag |= CoredMeshData::IN_CORE_FLAG[0]; }
		if (t.idx[1] < 0) { t.idx[1] = -t.idx[1] - 1; }
		else { inCoreFlag |= CoredMeshData::IN_CORE_FLAG[1]; }
		if (t.idx[2] < 0) { t.idx[2] = -t.idx[2] - 1; }
		else { inCoreFlag |= CoredMeshData::IN_CORE_FLAG[2]; }
		return 1;
	}
	else { return 0; }
}

int SparseSurfelFusion::CoredVectorMeshData::outOfCorePointCount(void) {
	return int(oocPoints.size()); 
}

int SparseSurfelFusion::CoredVectorMeshData::InCorePointsCount(void)
{
    return int(inCorePoints.size());
}

int SparseSurfelFusion::CoredVectorMeshData::triangleCount(void) {
	return int(triangles.size()); 
}

bool SparseSurfelFusion::CoredVectorMeshData::GetTriangleIndices(std::vector<unsigned int>& triangleIndices)
{
    if (triangles.size() > 0) {
        triangleIndices.resize(triangles.size() * 3);
        memcpy(triangleIndices.data(), triangles.data(), sizeof(unsigned int) * triangles.size() * 3);
        return true;
    }
    else return false;
}

bool SparseSurfelFusion::CoredVectorMeshData::GetTriangleIndices(std::vector<TriangleIndex>& triangleIndices)
{
    if (triangles.size() > 0) {
        triangleIndices.resize(triangles.size());
        memcpy(triangleIndices.data(), triangles.data(), sizeof(TriangleIndex) * triangles.size());
        return true;
    }
    else return false;
}

bool SparseSurfelFusion::CoredVectorMeshData::GetVertexArray(std::vector<float>& vertexArray)
{
    if (inCorePoints.size() > 0) {
        vertexArray.resize(inCorePoints.size() * 3);
        memcpy(vertexArray.data(), inCorePoints.data(), sizeof(float) * inCorePoints.size() * 3);
        return true;
    }
    else return false;
}

bool SparseSurfelFusion::CoredVectorMeshData::GetVertexArray(std::vector<Point3D<float>>& vertexArray)
{
    if (inCorePoints.size() > 0) {
        vertexArray.resize(inCorePoints.size());
        memcpy(vertexArray.data(), inCorePoints.data(), sizeof(Point3D<float>) * inCorePoints.size());
        return true;
    }
    else return false;
}

void SparseSurfelFusion::CoredVectorMeshData::clearAllContainer()
{
    inCorePoints.clear();
    triangles.clear();
}


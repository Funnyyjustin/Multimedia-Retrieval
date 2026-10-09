#pragma once
#include "object.h"

class Mesh;

const int minVertCount = 5000;
const int maxVertCount = 10000;

void resampleMesh(Mesh& input, int minVert = minVertCount, int maxVert = maxVertCount);
void upsampleMeshSimple(Mesh& input);
void upsampleMeshLoop(Mesh& input);
void downsampleMesh(Mesh& input, int targetVertices = -1);
float computeArea(const Primitive& prim);
float computeAverageArea(const Mesh& mesh);

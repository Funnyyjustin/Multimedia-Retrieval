#pragma once
#include "object.h"

class Mesh;

void resampleMesh(Mesh& input);
void upsampleMeshSimple(Mesh& input);
void upsampleMeshLoop(Mesh& input);
void downsampleMesh(Mesh& input);
float computeArea(const Primitive& prim);
float computeAverageArea(const Mesh& mesh);
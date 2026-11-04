#pragma once
#include <vector>
#include "object.h"

std::vector<double> featureExtract(Mesh& mesh);
void featureExtraction(std::vector<Mesh>& meshes, std::vector<std::vector<double>>& elem_desc);

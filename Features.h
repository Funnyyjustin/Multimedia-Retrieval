#pragma once
#include <vector>
#include "object.h"

std::vector<double> featureExtract(Mesh& mesh);
std::vector<std::vector<double>> featureExtraction(std::vector<Mesh>& meshes);

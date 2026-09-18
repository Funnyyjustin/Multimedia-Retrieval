#pragma once

#include <filesystem>
#include "vec3.h"
#include <vector>
#include "primitive.h"


/// <summary>
/// Object is list of primitives (triangles and quads) with label
/// </summary>
class Mesh 
{
public:
	Mesh(const std::vector<Primitive>& prims) : primitives(prims)
	{

	}

	void setLabel(const std::string& label) 
	{
		this->label = label;
	}

private:
	std::vector<Primitive> primitives;
	std::string label;
};
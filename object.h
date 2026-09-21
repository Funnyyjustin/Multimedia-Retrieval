#pragma once

#include <filesystem>
#include "vec3.h"
#include <vector>
#include "primitive.h"
#include <string>


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

	int size() 
	{
		return primitives.size();
	}

	void show()
	{
		for (int i = 0; i < primitives.size(); i++) primitives[i].show();
	}

private:
	std::vector<Primitive> primitives;
	std::string label;
};
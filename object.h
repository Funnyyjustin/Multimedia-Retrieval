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
	std::vector<float> trivertsgl;
	Point3 minPoint;
	Point3 maxPoint;
	bool trisOnly;
	bool quadsOnly;

	Mesh(const std::vector<Primitive>& prims, std::vector<float> verts) : primitives(prims), trivertsgl(verts)
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
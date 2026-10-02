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
	std::vector<Primitive> primitives;

	Mesh(const std::vector<Primitive>& prims, const std::vector<float>& verts, const int& numVerts) : primitives(prims), trivertsgl(verts), numVerts(numVerts)
	{
		
	}

	void updateGLBuffers() 
	{
		trivertsgl.clear();
		trivertsgl.reserve(primitives.size() * 9);
		for (const auto& prim : primitives)
		{
			for (const auto& vert : prim.vertices)
			{
				trivertsgl.push_back((float)vert.x());
				trivertsgl.push_back((float)vert.y());
				trivertsgl.push_back((float)vert.z());
			}
		}
		numVerts = trivertsgl.size() / 3;
	}
	
	void setLabel(const std::string& label) 
	{
		this->label = label;
	}

	std::string getLabel()
	{
		return label;
	}

	int primsSize() 
	{
		return primitives.size();
	}

	int vertsSize()
	{
		return numVerts;
	}

	void show()
	{
		for (int i = 0; i < primitives.size(); i++) primitives[i].show();
	}

private:
	std::string label;
	int numVerts;
};
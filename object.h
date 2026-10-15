#pragma once

#include <filesystem>
#include "vec3.h"
#include <vector>
#include "primitive.h"
#include <string>

struct Face
{
	std::vector<int> v;
	Point3 centroid;
	double area;
};

/// <summary>
/// 3D Mesh represented as an indexed face set with vertices and faces
/// </summary>
class Mesh 
{
public:
	std::vector<Point3> vertices;
	std::vector<Face> faces;
	std::vector<float> trivertsgl;
	
	Point3 minPoint;
	Point3 maxPoint;
	Point3 barycenter;
	bool trisOnly;
	bool quadsOnly;

	double area;
	double volume;

	Vec3 ev1, ev2, ev3; // Eigenvectors

	Mesh() = default;
	
	Mesh(const std::vector<Point3>& verts, const std::vector<Face>& fcs): vertices(verts), faces(fcs)
	{
		updateGLBuffers();
	}

	void updateGLBuffers() 
	{
		trivertsgl.clear();
		trivertsgl.reserve(faces.size() * 9); // Estimate for triangle vertices

		for (const auto& face : faces)
		{
			if (face.v.size() == 3)
			{
				for (int idx : face.v)
				{
					trivertsgl.push_back((float)vertices[idx].x());
					trivertsgl.push_back((float)vertices[idx].y());
					trivertsgl.push_back((float)vertices[idx].z());
				}
			}
			else if (face.v.size() == 4)
			{
				// Convert quad (0, 1, 2, 3) to 2 triangles for OpenGL: (0, 1, 2) and (0, 2, 3)
				int quadIdx[6] = { face.v[0], face.v[1], face.v[2], face.v[0], face.v[2], face.v[3] };
				for (int idx : quadIdx)
				{
					trivertsgl.push_back((float)vertices[idx].x());
					trivertsgl.push_back((float)vertices[idx].y());
					trivertsgl.push_back((float)vertices[idx].z());
				}
			}
		}
	}
	
	int primsSize() const { return (int)faces.size(); }
	int vertsSize() const { return (int)vertices.size(); }

	void setLabel(const std::string& l) { label = l; }
	std::string getLabel() const { return label; }

private:
	std::string label;
};

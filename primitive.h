#pragma once

#include "vec3.h"
#include <string>
#include <vector>
#include <iostream>

class Primitive
{
public:
	Point3 position;
	std::vector<Vec3> normals;
	std::vector<Point3> vertices;

	Primitive(const std::vector<Point3>& vertices, const std::vector<Vec3>& normals = {}) : vertices(vertices), normals(normals)
	{
		
	}

	void show()
	{
		for (int i = 0; i < vertices.size(); i++)
		{
			std::cout << vertices[i].show() << std::endl;
		}
		std::cout << std::endl;
	}
};

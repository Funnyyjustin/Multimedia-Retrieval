#pragma once

#include "vec3.h"
#include <string>

class Primitive
{
public:
	Point3 position;
	Vec3 normal;
	std::vector<Point3> vertices;

	Primitive(const std::vector<Point3> vertices) : vertices(vertices)
	{
		auto n = cross(vertices[0] - vertices[1], vertices[0] - vertices[2]);
		normal = unit_vector(n);
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


/// <summary>
/// Triangle is three points and normal calculated from the three points
/// </summary>
/*
class Triangle : public Primitive
{
public:
	Triangle(const Point3& a, const Point3& b, const Point3& c) : a(a), b(b), c(c)
	{
		auto n = cross(a - b, a - c);
		normal = unit_vector(n);
	}

	std::string show() override
	{
		return 
			"     " + a.show() + "\n" +
			"     " + b.show() + "\n" +
			"     " + c.show() + "\n";
	}

private:
	Point3 a, b, c;
};


/// <summary>
/// Quad is four points and normal calculated from three of those points (idk if that's correct)
/// </summary>
class Quad : public Primitive
{
public:
	Quad(const Point3& a, const Point3& b, const Point3& c, const Point3& d) : a(a), b(b), c(c), d(d)
	{
		auto n = cross(a - b, a - c);
		normal = unit_vector(n);
	}

	std::string show() override
	{
		return 
			"     " + a.show() + "\n" +
			"     " + b.show() + "\n" +
			"     " + c.show() + "\n" +
			"     " + d.show() + "\n";
	}

private:
	Point3 a, b, c, d;
};

*/
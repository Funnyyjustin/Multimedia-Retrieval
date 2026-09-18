#pragma once

#include "vec3.h"

class Primitive
{
public:

	Point3 position;
	Vec3 normal;
};


/// <summary>
/// Triangle is three points and normal calculated from the three points
/// </summary>
class Triangle : public Primitive
{
public:
	Triangle(const Point3& a, const Point3& b, const Point3& c) : a(a), b(b), c(c)
	{
		auto n = cross(a - b, a - c);
		normal = unit_vector(n);
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

private:
	Point3 a, b, c, d;
};
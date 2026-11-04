#pragma once

#include "object.h"
#include <filesystem>
#include <iostream>
#include <fstream>
#include <sstream>
#include <type_traits>

using string = std::string;

class Parser 
{
	// Use links given by Telea with code examples of parser 
	// OFF formatting: https://shape.cs.princeton.edu/benchmark/documentation/off_format.html
public:
	Mesh parseOFF(const std::string& path) 
	{
		std::ifstream file(path);

		//std::vector<Primitive> prims;
		//std::vector<float> trivertsgl;
		std::vector<Point3> vertices;
		//std::vector<Point3> normals;
		std::vector<Face> faces;

		float minX = INFINITY; float maxX = -INFINITY;
		float minY = INFINITY; float maxY = -INFINITY;
		float minZ = INFINITY; float maxZ = -INFINITY;

		bool onlyTri = true;
		bool onlyQuad = true;

		if (file.is_open())
		{
			while (file)
			{
				string line = " ";
				std::getline(file, line);

				if (line.substr(0, 2) == "v ")
				{
					std::istringstream l(line);
					string word;

					// Skip the first word: "v"
					string skip;
					l >> skip;

					l >> word;
					float a = std::stof(word);
					if (a > maxX) maxX = a;
					if (a < minX) minX = a;
					
					l >> word;
					float b = std::stof(word);
					if (b > maxY) maxY = b;
					if (b < minY) minY = b;

					l >> word;
					float c = std::stof(word);
					if (c > maxZ) maxZ = c;
					if (c < minZ) minZ = c;

					vertices.push_back(Point3(a, b, c));
				}
				else if (line.substr(0, 2) == "vn")
				{
					continue;
					/*
					std::istringstream l(line);
					string word;
					
					// Skip the first word: "vn"
					string skip;
					l >> skip;

					l >> word;
					float a = std::stof(word);

					l >> word;
					float b = std::stof(word);

					l >> word;
					float c = std::stof(word);

					normals.push_back(Point3(a, b, c));*/
				}
				else if (line.substr(0, 2) == "f ")
				{
					std::istringstream l(line);
					string skip, word;
					l >> skip; // Skip "f"

					Face face;
					while (l >> word)
					{
						std::size_t pos = word.find('/');
						int index = std::stoi(word.substr(0, pos)) - 1; // 1-indexed to 0-indexed
						face.v.push_back(index);
					}

					if (onlyTri && face.v.size() > 3) onlyTri = false;
					if (onlyQuad && face.v.size() < 4) onlyQuad = false;

					faces.push_back(face);
				}
			}
		}

		file.close();

		Mesh out(std::move(vertices), std::move(faces));
		out.maxPoint = Point3(maxX, maxY, maxZ);
		out.minPoint = Point3(minX, minY, minZ);
		out.trisOnly = onlyTri;
		out.quadsOnly = onlyQuad;

		return out;
	}
};
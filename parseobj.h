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

		
		std::vector<Primitive> prims;
		std::vector<float> trivertsgl;
		
		if (file.is_open()) 
		{
			// skip line that just says "OFF"
			string skip;
			std::getline(file, skip);

			// get numVertices, numFaces, numEdges from file
			long numVertices, numFaces, numEdges;

			string line = "";
			std::getline(file, line);
			string word;

			std::istringstream l(line);
			l >> word;
			numVertices = std::stol(word);

			l >> word;
			numFaces = std::stol(word);

			l >> word;
			numEdges = std::stol(word);

			std::vector<Point3> vertices;
			// get vertices from file
			for (long i = 0; i < numVertices; i++) 
			{
				std::getline(file, line);
				std::istringstream l(line);

				string word;

				l >> word;
				float a = std::stof(word);
				l >> word;
				float b = std::stof(word);
				l >> word;
				float c = std::stof(word);

				vertices.push_back(Point3(a, b, c));
			}

			for (int i = 0; i < numFaces; i++)
			{
				std::getline(file, line);
				std::istringstream l(line);

				string word;
				l >> word;
				int num = std::stoi(word);

				std::vector<Point3> pts;
				for (int j = 0; j < num; j++)
				{
					l >> word;
					int vertexIdx = std::stoi(word);
					pts.push_back(vertices[vertexIdx]);
				}

				// 1. Eerste driehoek (werkt voor zowel 3- als 4-punts vlakken)
				if (pts.size() >= 3)
				{
					for (int idx : {0, 1, 2})
					{
						trivertsgl.push_back(pts[idx].x());
						trivertsgl.push_back(pts[idx].y());
						trivertsgl.push_back(pts[idx].z());
					}
				}

				// 2. Tweede driehoek als het een quad (4 hoekpunten) is: hoekpunten 0, 2, 3
				if (pts.size() == 4)
				{
					for (int idx : {0, 2, 3})
					{
						trivertsgl.push_back(pts[idx].x());
						trivertsgl.push_back(pts[idx].y());
						trivertsgl.push_back(pts[idx].z());
					}
				}

				// Sla de primitief op
				prims.push_back(Primitive(pts));
			}
		}

		file.close();
		return Mesh(prims, trivertsgl);
	}
};
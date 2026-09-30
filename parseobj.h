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
		std::vector<Point3> vertices;
		std::vector<Point3> normals;

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

					normals.push_back(Point3(a, b, c));
				}
				else if (line.substr(0, 2) == "f ")
				{
					std::istringstream l(line);
					std::vector<string> temp;
					std::vector<Point3> verts;
					std::vector<Vec3> norms;

					string word;
					
					string skip;
					l >> skip;

					while (l >> word)
						temp.push_back(word);

					if (onlyTri && temp.size() > 3) onlyTri = false;
					if (onlyQuad && temp.size() < 4) onlyQuad = false;

					for (int i = 0; i < temp.size(); i++)
					{
						string w = temp[i];
						std::size_t pos = w.find('/');
						int index = std::stoi(w.substr(0, pos)) - 1;
						//std::cout << index << " " << vertices.size() << std::endl;
						Point3 vert = vertices[index];
						verts.push_back(vert);

						trivertsgl.push_back(vert.x());
						trivertsgl.push_back(vert.y());
						trivertsgl.push_back(vert.z());

						//std::cout << index << " " << norms.size() << std::endl;
						if (normals.size() > 0)
							norms.push_back(normals[index]);
					}

					prims.push_back(Primitive(verts, norms));
				}
			}
			file.close();
			return Mesh(prims, trivertsgl);
		}


		/*
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
		*/
		
		file.close();

		Mesh out = Mesh(prims, trivertsgl);
		out.maxPoint = Point3(maxX, maxY, maxZ);
		out.minPoint = Point3(minX, minY, minZ);
		out.trisOnly = onlyTri;
		out.quadsOnly = onlyQuad;

		return Mesh(prims, trivertsgl);
	}
};
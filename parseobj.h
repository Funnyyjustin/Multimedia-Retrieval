#pragma once

#include "object.h"
#include <filesystem>
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <cstdlib>

using string = std::string;

class Parser 
{
public:
	Mesh parseOFF(const std::string& path) 
	{
		std::ifstream file(path, std::ios::binary | std::ios::ate);
		if (!file.is_open())
		{
			return Mesh({}, {});
		}

		std::streamsize size = file.tellg();
		if (size <= 0)
		{
			return Mesh({}, {});
		}
		file.seekg(0, std::ios::beg);

		std::vector<char> buffer(static_cast<size_t>(size) + 1);
		if (!file.read(buffer.data(), size))
		{
			return Mesh({}, {});
		}
		buffer[static_cast<size_t>(size)] = '\0';
		file.close();

		std::vector<Point3> vertices;
		std::vector<Face> faces;
		vertices.reserve(static_cast<size_t>(size) / 60);
		faces.reserve(static_cast<size_t>(size) / 40);

		float minX = INFINITY; float maxX = -INFINITY;
		float minY = INFINITY; float maxY = -INFINITY;
		float minZ = INFINITY; float maxZ = -INFINITY;

		bool onlyTri = true;
		bool onlyQuad = true;

		char* p = buffer.data();
		char* end = p + size;

		std::vector<int> polyIndices;
		polyIndices.reserve(8);

		while (p < end)
		{
			// Skip leading horizontal whitespace
			while (p < end && (*p == ' ' || *p == '\t' || *p == '\r')) p++;
			if (p >= end) break;

			if (*p == 'v' && *(p + 1) == ' ')
			{
				p += 2;
				char* next = nullptr;
				float a = std::strtof(p, &next); p = next;
				float b = std::strtof(p, &next); p = next;
				float c = std::strtof(p, &next); p = next;

				if (a > maxX) maxX = a; if (a < minX) minX = a;
				if (b > maxY) maxY = b; if (b < minY) minY = b;
				if (c > maxZ) maxZ = c; if (c < minZ) minZ = c;

				vertices.push_back(Point3(a, b, c));
			}
			else if (*p == 'f' && *(p + 1) == ' ')
			{
				p += 2;
				polyIndices.clear();

				while (p < end && *p != '\n' && *p != '\r')
				{
					while (p < end && (*p == ' ' || *p == '\t')) p++;
					if (p >= end || *p == '\n' || *p == '\r') break;

					char* next = nullptr;
					long idx = std::strtol(p, &next, 10);
					if (next == p)
					{
						// Skip non-digit token
						while (p < end && *p != ' ' && *p != '\t' && *p != '\n' && *p != '\r') p++;
						continue;
					}
					p = next;
					polyIndices.push_back(static_cast<int>(idx - 1));

					// Skip texture/normal component (e.g. /12/34)
					while (p < end && *p != ' ' && *p != '\t' && *p != '\n' && *p != '\r') p++;
				}

				if (polyIndices.size() >= 3)
				{
					if (onlyTri && polyIndices.size() > 3) onlyTri = false;
					if (onlyQuad && polyIndices.size() < 4) onlyQuad = false;

					for (size_t i = 1; i + 1 < polyIndices.size(); ++i)
					{
						Face triFace;
						triFace.v = { polyIndices[0], polyIndices[i], polyIndices[i + 1] };
						faces.push_back(std::move(triFace));
					}
				}
			}

			// Advance to next line
			while (p < end && *p != '\n') p++;
			if (p < end && *p == '\n') p++;
		}

		Mesh out(std::move(vertices), std::move(faces));
		out.maxPoint = Point3(maxX, maxY, maxZ);
		out.minPoint = Point3(minX, minY, minZ);
		out.trisOnly = onlyTri;
		out.quadsOnly = onlyQuad;

		return out;
	}
};
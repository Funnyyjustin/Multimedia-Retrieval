#pragma once
#include <random>
#include <vector>
#include <algorithm>
#include <cmath>
#include "vec3.h"
#include "object.h"

class MeshSampler
{
private:
    const Mesh& mesh;
    std::vector<double> cumulativeAreas;
    std::mt19937 rng;
    std::uniform_real_distribution<double> dist01{ 0.0, 1.0 };

public:
    MeshSampler(const Mesh& m, unsigned int seed = 1337) 
        : mesh(m), rng(seed)
    {
        double sum = 0.0;
        cumulativeAreas.reserve(mesh.faces.size());
        for (const auto& f : mesh.faces)
        {
            sum += f.area;
            cumulativeAreas.push_back(sum);
        }
    }

    // Generate 1 uniform point on the surface of the mesh
    Point3 samplePoint()
    {
        if (cumulativeAreas.empty() || cumulativeAreas.back() <= 0.0)
            return Point3(0, 0, 0);

        double rArea = dist01(rng) * cumulativeAreas.back();
        auto it = std::lower_bound(cumulativeAreas.begin(), cumulativeAreas.end(), rArea);
        size_t faceIdx = std::min<size_t>(std::distance(cumulativeAreas.begin(), it), mesh.faces.size() - 1);

        const auto& face = mesh.faces[faceIdx];
        const Point3& A = mesh.vertices[face.v[0]];
        const Point3& B = mesh.vertices[face.v[1]];
        const Point3& C = mesh.vertices[face.v[2]];

        double r1 = dist01(rng);
        double r2 = dist01(rng);
        double sqrtR1 = std::sqrt(r1);

        double u = 1.0 - sqrtR1;
        double v = sqrtR1 * (1.0 - r2);
        double w = sqrtR1 * r2;

        return A * u + B * v + C * w;
    }

    // Generate a batch of N points
    std::vector<Point3> samplePoints(size_t n)
    {
        std::vector<Point3> pts(n);
        for (size_t i = 0; i < n; ++i)
        {
            pts[i] = samplePoint();
        }
        return pts;
    }
};

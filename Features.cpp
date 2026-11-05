#include "Features.h"
#include "Historgam.h"
#include "MeshSampler.h"
#include "object.h"

#include <filesystem>
#include <vector>
#include <algorithm>
#include <fstream>
#include <omp.h>
#include <cstdio>
#include <cmath>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

const size_t NUM_BINS = 20;

double calculateCompactness(Mesh& mesh)
{
    double area = mesh.area;
    double volume = mesh.volume;

    if (volume <= 1e-9 || std::isnan(volume) || std::isnan(area)) return 1000.0;

    double comp = (area * area * area) / (36.0 * M_PI * volume * volume);
    return std::min(comp, 1000.0);
}

double calculateVolumeAABB(Mesh& mesh)
{
    auto min = mesh.minPoint;
    auto max = mesh.maxPoint;

    return ((max.e[0] - min.e[0]) * (max.e[1] - min.e[1]) * (max.e[2] - min.e[2]));
}

double calculateRectangularity(Mesh& mesh)
{
    double volume = mesh.volume;
    double volumeAABB = calculateVolumeAABB(mesh);
    if (volumeAABB <= 1e-9 || std::isnan(volumeAABB) || std::isnan(volume)) return 0.0;
    return volume / volumeAABB;
}

double getDiameter(Mesh& mesh)
{
    double maxDistSquared = 0.0;
    const auto& vertices = mesh.vertices;
    if (vertices.empty()) return 0.0;

    for (size_t i = 0; i < vertices.size(); ++i)
    {
        if (std::isnan(vertices[i].e[0])) return 0.0;
        for (size_t j = i + 1; j < vertices.size(); ++j)
        {
            double dx = vertices[i].e[0] - vertices[j].e[0];
            double dy = vertices[i].e[1] - vertices[j].e[1];
            double dz = vertices[i].e[2] - vertices[j].e[2];

            double distSquared = dx * dx + dy * dy + dz * dz;

            maxDistSquared = std::max(maxDistSquared, distSquared);
        }
    }

    return std::sqrt(maxDistSquared);
}

double calculateConvexity(Mesh& mesh)
{
    if (mesh.vertices.size() < 4 || mesh.faces.size() < 4 || mesh.volume <= 1e-9 || std::isnan(mesh.volume) || std::isnan(mesh.area))
    {
        return 0.0;
    }

    for (const auto& v : mesh.vertices)
    {
        if (std::isnan(v.x()) || std::isnan(v.y()) || std::isnan(v.z()) ||
            std::isinf(v.x()) || std::isinf(v.y()) || std::isinf(v.z()))
        {
            return 0.0;
        }
    }

    // 1. Uniek filename per OpenMP thread to prevent race conditions
    int tid = omp_get_thread_num();
    std::string inFileName = "temp_inp_" + std::to_string(tid) + ".txt";
    std::string outFileName = "temp_out_" + std::to_string(tid) + ".txt";

    // 2. write the mesh vertices to the text files for qhull
    {
        std::ofstream fo(inFileName);
        if (!fo.is_open()) 
            return 0.0;

        fo << "3\n";
        fo << mesh.vertices.size() << "\n";
        for (const auto& v : mesh.vertices)
        {
            fo << v.x() << " " << v.y() << " " << v.z() << "\n";
        }
    }

    // 3. execut qhull directly from the qhull/bin map (redirect stderr to nul)
    std::string cmd = "qhull\\bin\\qhull.exe TI " + inFileName + " QJ FS TO " + outFileName + " 2> nul";
    int ret = system(cmd.c_str());

    double hullVolume = 0.0;
    if (ret == 0)
    {
        // 4. Read the volume from the FS output from qhull (format: line 1 = 0, line 2 = 2 <area> <volume>)
        std::ifstream fi(outFileName);
        if (fi.is_open())
        {
            int dummy1 = 0, dummy2 = 0;
            double hullArea = 0.0;
            if (fi >> dummy1 >> dummy2 >> hullArea >> hullVolume)
            {
                // read successfully
            }
        }
    }

    // 5. Clean up the temp text files
    std::remove(inFileName.c_str());
    std::remove(outFileName.c_str());

    // 6. Convexity = mesh volume / hull volume
    if (hullVolume <= 0.0 || std::isnan(hullVolume))
    {
        return 0.0;
    }

    double convexity = mesh.volume / hullVolume;
    return std::clamp(convexity, 0.0, 1.0);
}

double calculateEccentricity(Mesh& mesh)
{
    double lambda1 = mesh.eigenvalues.x();
    double lambda3 = mesh.eigenvalues.z();

    if (lambda3 <= 1e-9 || std::isnan(lambda1) || std::isnan(lambda3) || std::isinf(lambda1) || std::isinf(lambda3))
    {
        return 1.0;
    }

    return lambda1 / lambda3;
}

// D1: Distance between barycentre and 1 random point
double evalD1(const Point3 pts[1], const Point3& barycenter)
{
    return (pts[0] - barycenter).length();
}

// D2: Distance between 2 random points
double evalD2(const Point3 pts[2])
{
    return (pts[0] - pts[1]).length();
}

// A3: Angle between 3 random points
double evalA3(const Point3 pts[3])
{
    Vec3 v1 = unit_vector(pts[0] - pts[1]);
    Vec3 v2 = unit_vector(pts[2] - pts[1]);
    double cosTheta = std::clamp(dot(v1, v2), -1.0, 1.0);
    return std::acos(cosTheta); // Hoek in radialen [0, PI]
}

// D3: Square root of area of a triangle formed by 3 points
double evalD3(const Point3 pts[3])
{
    Vec3 crossProd = cross(pts[1] - pts[0], pts[2] - pts[0]);
    double area = 0.5 * crossProd.length();
    return std::sqrt(area);
}

// D4: Cube root of volume formed by 4 points
double evalD4(const Point3 pts[4])
{
    Vec3 crossProd = cross(pts[2] - pts[0], pts[3] - pts[0]);
    double vol = std::abs(dot(pts[1] - pts[0], crossProd)) / 6.0;
    return std::cbrt(vol);
}

std::vector<double> calculateBarycenterDist(const std::vector<Point3>& pts, const Point3& barycenter) // D1
{
    std::vector<double> vals;
    vals.reserve(pts.size());
    for (const auto& p : pts) {
        vals.push_back((p - barycenter).length());
    }
    return computeFixedHistogram(vals, NUM_BINS, 0.0, 0.90);
}

std::vector<double> calculateVerticesDist(const std::vector<Point3>& pts) // D2
{
    std::vector<double> vals;
    size_t numPairs = pts.size() / 2;
    vals.reserve(numPairs);
    for (size_t i = 0; i < numPairs; ++i) {
        vals.push_back(evalD2(&pts[2 * i]));
    }
    return computeFixedHistogram(vals, NUM_BINS, 0.0, 1.75);
}

std::vector<double> calculateAngle(const std::vector<Point3>& pts) // A3
{
    std::vector<double> vals;
    size_t numTriplets = pts.size() / 3;
    vals.reserve(numTriplets);
    for (size_t i = 0; i < numTriplets; ++i) {
        vals.push_back(evalA3(&pts[3 * i]));
    }
    return computeFixedHistogram(vals, NUM_BINS, 0.0, M_PI);
}

std::vector<double> calculateRootArea(const std::vector<Point3>& pts) // D3
{
    std::vector<double> vals;
    size_t numTriplets = pts.size() / 3;
    vals.reserve(numTriplets);
    for (size_t i = 0; i < numTriplets; ++i) {
        vals.push_back(evalD3(&pts[3 * i]));
    }
    return computeFixedHistogram(vals, NUM_BINS, 0.0, 0.75);
}

std::vector<double> calculateRootVolume(const std::vector<Point3>& pts) // D4
{
    std::vector<double> vals;
    size_t numQuads = pts.size() / 4;
    vals.reserve(numQuads);
    for (size_t i = 0; i < numQuads; ++i) {
        vals.push_back(evalD4(&pts[4 * i]));
    }
    return computeFixedHistogram(vals, NUM_BINS, 0.0, 0.60);
}

std::vector<double> featureExtract(Mesh& mesh)
{
    std::vector<double> vec;

    // 1. Elementaire descriptors (6 waarden)
    vec.push_back(mesh.area);
    vec.push_back(calculateCompactness(mesh));
    vec.push_back(calculateRectangularity(mesh));
    vec.push_back(getDiameter(mesh));
    vec.push_back(calculateConvexity(mesh));
    vec.push_back(calculateEccentricity(mesh));

    // 2. Monte Carlo sampling (1x execution per mesh)
    MeshSampler sampler(mesh);
    std::vector<Point3> samples = sampler.samplePoints(60000);

    // 3. Add shape property histogrammen (5 x 20 bins)
    auto appendHistogram = [&vec](const std::vector<double>& hist) {
        vec.insert(vec.end(), hist.begin(), hist.end());
    };

    appendHistogram(calculateAngle(samples));                           // A3 (20 bins)
    appendHistogram(calculateBarycenterDist(samples, mesh.barycenter)); // D1 (20 bins)
    appendHistogram(calculateVerticesDist(samples));                   // D2 (20 bins)
    appendHistogram(calculateRootArea(samples));                       // D3 (20 bins)
    appendHistogram(calculateRootVolume(samples));                     // D4 (20 bins)

    return vec; // Totaal 106 values
}

void featureExtraction(std::vector<Mesh>& meshes, std::vector<std::vector<double>>& elem_desc)
{
    //std::vector<std::vector<double>> vec;

    for (std::size_t i = 0, e = meshes.size(); i != e; ++i)
    {
        std::cout << "extracting features for mesh: " << i << "/" << meshes.size() << std::endl;
        elem_desc.push_back(featureExtract(meshes[i]));
    }

    //return vec;
}

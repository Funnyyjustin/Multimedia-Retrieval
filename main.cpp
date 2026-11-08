#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <filesystem>
#include <iostream>
#include <vector>
#include <algorithm>
#include <limits>
#include <omp.h>
#include <atomic>
#include <math.h>
#include "parseobj.h"
#include "object.h"
#include "camera.h"
#include "fileSelector.h"
#include "resampling.h"
#include "Histogram.h"
#include "Features.h"
#include "QuickSort.h"
#include "FeatureNormStand.h"
#include "Matching.h"
#include "matchheur.h"
#include "globals.h"
#include "projection.h"

#define GLM_ENABLE_EXPERIMENTAL
#include <glm/gtx/pca.hpp>
#include <glm/glm.hpp>

std::vector<std::string> paths;

const char* fragmentShaderSource = { "#version 330 core\n"
"in vec3 FragPosEye;\n"
"uniform vec4 uColor;\n"
"uniform bool uUseLighting;\n"
"out vec4 FragColor;\n"
"void main()\n"
"{\n"
"   if (!uUseLighting) {\n"
"       FragColor = uColor;\n"
"       return;\n"
"   }\n"
"   // Compute face normal using screen-space derivatives\n"
"   vec3 normal = normalize(cross(dFdx(FragPosEye), dFdy(FragPosEye)));\n"
"   if (normal.z < 0.0) normal = -normal;\n" // Ensure normal points towards camera
"\n"
"   // Directional light from slightly above and right of camera\n"
"   vec3 lightDir = normalize(vec3(0.3, 0.6, 0.8));\n"
"   vec3 viewDir = normalize(-FragPosEye);\n"
"\n"
"   // 1. Ambient\n"
"   vec3 ambient = 0.25 * uColor.rgb;\n"
"\n"
"   // 2. Diffuse\n"
"   float diff = max(dot(normal, lightDir), 0.0);\n"
"   vec3 diffuse = diff * uColor.rgb;\n"
"\n"
"   // 3. Specular (Blinn-Phong)\n"
"   vec3 halfwayDir = normalize(lightDir + viewDir);\n"
"   float spec = pow(max(dot(normal, halfwayDir), 0.0), 32.0);\n"
"   vec3 specular = vec3(0.2) * spec;\n"
"\n"
"   FragColor = vec4(ambient + diffuse + specular, uColor.a);\n"
"}\n\0" } ;
const char* vertexShaderSource = { "#version 330 core\n"
"layout (location = 0) in vec3 aPos;\n"
"uniform mat4 uMVP;\n"
"uniform mat4 uModelView;\n"
"out vec3 FragPosEye;\n"
"void main()\n"
"{\n"
"   FragPosEye = vec3(uModelView * vec4(aPos, 1.0));\n"
"   gl_Position = uMVP * vec4(aPos, 1.0);\n"
"}\0" } ;

namespace fs = std::filesystem;
int application_mode;

// Helper functions

std::vector<int> flatten(std::vector<std::vector<int>>& vec)
{
    std::vector<int> result;

    for (const auto& v : vec) {
        result.insert(result.end(), v.begin(), v.end());
    }

    return result;
}

std::vector<Mesh> flatten(std::vector<std::vector<Mesh>>& vec)
{
    std::vector<Mesh> result;

    for (const auto& v : vec) {
        result.insert(result.end(), v.begin(), v.end());
    }

    return result;
}

void exportToCSV(const std::vector<Mesh>& meshes, const std::vector<int>& faces, const std::vector<int>& verts, const std::vector<int>& types, const std::vector<Point3>& barycentres, const std::vector<Vec3>& scales, const std::string& filename)
{
    std::ofstream file(filename + ".csv");

    if (!file.is_open())
        return;

    file << "Shape id; Number of faces; Number of vertices; Face type; Volume; Area; Barycentre X; Barycentre Y; Barycenter Z; Scale X; Scale Y; Scale Z; EV1 x; EV1 y; EV1 z; EV2 x; EV2 y; EV2 z; EV3 x; EV3 y; EV3 z; sign x; sign y; sign z\n";

    size_t maxSize = std::max({ faces.size(), verts.size(), types.size() });

    for (size_t i = 0; i < maxSize; ++i)
    {
        file << i << ";";

        if (i < faces.size())
            file << faces[i];

        file << ";";

        if (i < verts.size())
            file << verts[i];

        file << ";";

        if (i < types.size())
            file << types[i];

        file << ";";

        file << meshes[i].volume;

        file << ";";

        file << meshes[i].area;

        file << ";";

        if (i < barycentres.size())
        {
            file << barycentres[i].x();
            file << ";";
            file << barycentres[i].y();
            file << ";";
            file << barycentres[i].z();
        } else
        {
            file << ";";
            file << ";";
        }
        
        file << ";";
        
        if (i < scales.size())
        {
            file << scales[i].x();
            file << ";";
            file << scales[i].y();
            file << ";";
            file << scales[i].z();
        } else
        {
            file << ";";
            file << ";";
        }

        file << ";";

        file << meshes[i].ev1.e[0];
        file << ";";
        file << meshes[i].ev1.e[1];
        file << ";";
        file << meshes[i].ev1.e[2];
        file << ";";

        file << meshes[i].ev2.e[0];
        file << ";";
        file << meshes[i].ev2.e[1];
        file << ";";
        file << meshes[i].ev2.e[2];
        file << ";";

        file << meshes[i].ev3.e[0];
        file << ";";
        file << meshes[i].ev3.e[1];
        file << ";";
        file << meshes[i].ev3.e[2];
        file << ";";

        file << meshes[i].sx;
        file << ";";
        file << meshes[i].sy;
        file << ";";
        file << meshes[i].sz;
        file << ";";

        file << "\n";
    }
}

void exportFeaturesToCSV(const std::vector<Mesh>& meshes, const std::vector<std::vector<double>>& features, const std::string& filename)
{
    std::string fullPath = filename;
    if (fullPath.size() < 4 || fullPath.substr(fullPath.size() - 4) != ".csv")
    {
        fullPath += ".csv";
    }

    std::ofstream file(fullPath);
    if (!file.is_open())
    {
        std::cerr << "Error: Could not open file " << fullPath << " for writing.\n";
        return;
    }

    if (features.empty())
    {
        std::cerr << "Warning: Features list is empty.\n";
        return;
    }

    // Build CSV Header
    file << "Shape id;Class";
    if (features[0].size() == numFeatures)
    {
        file << ";Surface Area;Compactness;Rectangularity;Diameter;Convexity;Eccentricity";
        for (int i = 0; i < numHistogramBins; ++i) file << ";A3_" << i;
        for (int i = 0; i < numHistogramBins; ++i) file << ";D1_" << i;
        for (int i = 0; i < numHistogramBins; ++i) file << ";D2_" << i;
        for (int i = 0; i < numHistogramBins; ++i) file << ";D3_" << i;
        for (int i = 0; i < numHistogramBins; ++i) file << ";D4_" << i;
    }
    else
    {
        for (size_t col = 0; col < features[0].size(); ++col)
        {
            file << ";Feature_" << col;
        }
    }
    file << "\n";

    // Write data rows
    for (size_t i = 0; i < features.size(); ++i)
    {
        file << i << ";";
        if (i < meshes.size())
            file << meshes[i].getLabel();
        else
            file << "Unknown";

        for (double val : features[i])
        {
            file << ";" << val;
        }
        file << "\n";
    }

    std::cout << "Successfully exported " << features.size() << " feature vectors to " << fullPath << "\n";
}

void promptSaveFeatures(const std::vector<Mesh>& meshes, const std::vector<std::vector<double>>& features)
{
    std::cout << "\nFeature extraction completed for " << features.size() << " meshes.\n";
    std::cout << "Export features to CSV file? (y/n):\n";
    char choice;
    std::cin >> choice;

    if (choice == 'y' || choice == 'Y')
    {
        std::cout << "Enter CSV filename:\n";
        std::string filename;
        std::cin >> filename;
        exportFeaturesToCSV(meshes, features, filename);
    }
    else if (choice != 'n' && choice != 'N')
    {
        std::cout << "Invalid input. Skipping CSV export.\n";
    }
}

void exportEvaluationToCSV(const std::vector<std::pair<float, float>>& points, const float& total_area, const std::string& filename)
{
    std::string fullPath = filename;
    if (fullPath.size() < 4 || fullPath.substr(fullPath.size() - 4) != ".csv")
    {
        fullPath += ".csv";
    }

    std::ofstream file(fullPath);
    if (!file.is_open())
    {
        std::cerr << "Error: Could not open file " << fullPath << " for writing.\n";
        return;
    }

    if (points.empty())
    {
        std::cerr << "Warning: Points list is empty.\n";
        return;
    }

    file << "Plot point id;Sensitivity;Specificity;;" << total_area << "\n";

    // Write data rows
    for (size_t i = 0; i < points.size(); ++i)
    {
        file << i << ";" << points[i].first << ";" << points[i].second << "\n";
    }

    std::cout << "Successfully exported " << points.size() << " evaluation plot points to " << fullPath << "\n";
}

void exportEvaluationPerClassToCSV(const std::vector<std::vector<std::pair<float,float>>>& points_per_class,
    const std::vector<float>& total_area_per_class,
    const std::vector<string>& labels,
    const std::vector<int>& obj_counts,
    const std::string& filename)
{
    std::string fullPath = filename;
    if (fullPath.size() < 4 || fullPath.substr(fullPath.size() - 4) != ".csv")
    {
        fullPath += ".csv";
    }

    std::ofstream file(fullPath);
    if (!file.is_open())
    {
        std::cerr << "Error: Could not open file " << fullPath << " for writing.\n";
        return;
    }

    if (points_per_class.empty())
    {
        std::cerr << "Warning: Points list is empty.\n";
        return;
    }

    file << "Class;Sensitivity;Specificity;AUC\n";

    // Write data rows
    int current_class = 0;
    for (int i = 0; i < points_per_class.size(); i++)
    {
        auto str = labels[current_class];
        auto& entry = points_per_class[i];
        for (int j = 0; j < entry.size(); j++)
        {
            file << str << ";" << entry[j].first << ";" << entry[j].second << "\n";
        }
        file << str << ";;;" << total_area_per_class[i] << "\n";
        file << "\n";
        current_class += obj_counts[i];
    }

    std::cout << "Successfully exported " << points_per_class.size() << " classes' plot points to " << fullPath << "\n";
}

// Database analyse functions

std::vector<Mesh> parseDatabase(fs::path path)
{
    int shape_limit = 0;
    int class_limit = 0;

    std::cout << "Type in the max number of classes to parse (0-69):\n";
    std::cin >> class_limit;

    std::cout << "Type in the max number of shapes per class to parse (0-143):\n";
    std::cin >> shape_limit;

    struct Task {
        fs::path filePath;
        std::string className;
    };
    std::vector<Task> tasks;

    std::cout << "Scanning database files...\n";

    int class_index = 0;
    for (const auto& folder : fs::directory_iterator(path))
    {
        if (class_index >= class_limit)
            break;

        if (!folder.is_directory())
            continue;

        int shape_index = 0;
        for (const auto& file : fs::directory_iterator(folder.path()))
        {
            if (shape_index >= shape_limit)
                break;

            tasks.push_back({ file.path(), folder.path().filename().string() });
            shape_index++;
        }

        class_index++;
    }

    int total = static_cast<int>(tasks.size());
    std::vector<Mesh> meshes(total);
    std::atomic<int> parsedCount{0};

    std::cout << "Parsing " << total << " files in parallel...\n";

    #pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < total; ++i)
    {
        Parser p;
        meshes[i] = p.parseOFF(tasks[i].filePath.string());
        meshes[i].setLabel(tasks[i].className);

        int done = ++parsedCount;
        if (done % 100 == 0 || done == total)
        {
            #pragma omp critical
            std::cout << "[Parsing] Completed " << done << " / " << total << " files.\n";
        }
    }

    std::cout << "Finished parsing database.\n";
    return meshes;
}

void analyseDatabase(const std::vector<Mesh>& meshes)
{
    std::cout << "Analysing database..\n";

    int n = static_cast<int>(meshes.size());
    std::vector<int> faces(n);
    std::vector<int> verts(n);
    std::vector<int> types(n);
    std::vector<Point3> barycentres(n);
    std::vector<Vec3> scales(n);

    // Compute metrics in parallel
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < n; ++i)
    {
        const Mesh& m = meshes[i];
        faces[i] = m.primsSize();
        verts[i] = m.vertsSize();

        if (m.quadsOnly) types[i] = 0;
        else if (m.trisOnly) types[i] = 1;
        else types[i] = 2;

        /*Point3 cumulative_point(0, 0, 0);
        for (const auto& point : m.vertices)
        {
            cumulative_point += point;
        }
        if (!m.vertices.empty())
        {
            cumulative_point /= static_cast<double>(m.vertices.size());
        }
        barycentres[i] = cumulative_point;*/
        barycentres[i] = m.barycenter;

        scales[i] = { m.maxPoint - m.minPoint };
    }

    // Print summary to console cleanly
    /*for (int i = 0; i < n; ++i)
    {
        const Mesh& m = meshes[i];
        std::string type = (types[i] == 0) ? "Quads only" : (types[i] == 1 ? "Tris only" : "Mix of tris and quads");
        std::cout << "Class: " << m.getLabel() << "\n";
        std::cout << "Number of faces: " << faces[i] << "\n";
        std::cout << "Number of vertices: " << verts[i] << "\n";
        std::cout << "Type of faces: " << type << "\n";
        std::cout << "AABB: \n";
        std::cout << "  min: " << m.minPoint.e[0] << " " << m.minPoint.e[1] << " " << m.minPoint.e[2] << "\n";
        std::cout << "  max: " << m.maxPoint.e[0] << " " << m.maxPoint.e[1] << " " << m.maxPoint.e[2] << "\n";
        std::cout << "\n";
    }*/

    std::cout << "Export to CSV file? y/n\n";
    char c;
    std::cin >> c;

    if (c == 'y')
    {
        std::cout << "Enter CSV file name:\n";
        std::string filename;
        std::cin >> filename;
        exportToCSV(meshes, faces, verts, types, barycentres, scales, filename);
    }
    else if (c != 'y' && c != 'n')
        std::cout << "Invalid input. Try again.\n";
}

// Resample functions

void resampleMeshes(std::vector<Mesh>& meshes, int minVert = minVertCount, int maxVert = maxVertCount)
{
    int total = static_cast<int>(meshes.size());
    std::cout << "Resampling " << total << " meshes in parallel...\n";

    std::atomic<int> completed{0};

    #pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < total; ++i)
    {
        resampleMesh(meshes[i], minVert, maxVert);

        int done = ++completed;
        if (done % 20 == 0 || done == total)
        {
            #pragma omp critical
            std::cout << "[Resampling] Completed " << done << " / " << total << " meshes (" << (done * 100 / total) << "%)\n";
        }
    }

    std::cout << "Finished resampling meshes.\n";
}

// Precomputation functions

Point3 calculateCentroid(const Face& face, const Mesh& mesh)
{
    Point3 p(0, 0, 0);
    for (const auto& v : face.v)
    {
        p += mesh.vertices[v];
    }
    p /= 3;
    return p;
}

double calculateArea(const Face& face, const Mesh& mesh)
{
    // Get the three vertices
    Point3 a = mesh.vertices[face.v[0]];
    Point3 b = mesh.vertices[face.v[1]];
    Point3 c = mesh.vertices[face.v[2]];

    // Calculate two edge vectors of the face
    Vec3 sideone = b - a;
    Vec3 sidetwo = c - a;

    // Calculate the cross product
    Vec3 crossproduct = cross(sideone, sidetwo);

    // Calculate area
    auto r = 0.5f * crossproduct.length();

    return r;
}

Point3 calculateBarycenter(Mesh& input)
{
    if (input.faces.empty()) return Point3(0, 0, 0);

    Point3 barycenter(0, 0, 0);
    double total_area = 0.0;

    for (int i = 0; i < static_cast<int>(input.faces.size()); i++)
    {
        auto& f = input.faces[i];
        barycenter += f.centroid * f.area;
        total_area += f.area;
    }

    if (total_area <= 1e-9 || std::isnan(total_area))
    {
        input.area = 0.0;
        return Point3(0, 0, 0);
    }

    barycenter /= total_area;
    input.area = total_area;

    return barycenter;
}

double calculateVolume(Mesh& input)
{
    double signedVolume = 0.0;
    double absVolume = 0.0;

    for (const auto& f : input.faces)
    {
        if (f.v.size() < 3) continue;

        const Vec3& a = input.vertices[f.v[0]] - input.barycenter;
        const Vec3& b = input.vertices[f.v[1]] - input.barycenter;
        const Vec3& c = input.vertices[f.v[2]] - input.barycenter;

        double tetVol = dot(a, cross(b, c)) / 6.0;
        signedVolume += tetVol;
        absVolume += std::abs(tetVol);
    }

    double vol = std::abs(signedVolume);

    if (vol <= 1e-6 && absVolume > 1e-6)
    {
        vol = 0.5 * absVolume;
    }

    return vol;
}

void precomputeMesh(Mesh& input)
{
    // For each face, precompute the centroid and the area
    for (int i = 0; i < static_cast<int>(input.faces.size()); i++)
    {
        auto& f = input.faces[i];
        f.area = calculateArea(f, input);
        f.centroid = calculateCentroid(f, input);
    }

    // Calculate barycenter and volume of each mesh
    input.barycenter = calculateBarycenter(input);
    input.volume = calculateVolume(input);
}

void precomputeMeshes(std::vector<Mesh>& meshes)
{
    int total = static_cast<int>(meshes.size());
    std::cout << "Precomputing " << total << " meshes in parallel...\n";

    #pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < total; ++i)
    {
        precomputeMesh(meshes[i]);
    }

    std::cout << "Finished precomputing meshes.\n";
}

// Normalisation functions + helper

void calculateAABB(Mesh& input)
{
    if (input.vertices.empty()) return;

    float x_min, x_max, y_min, y_max, z_min, z_max;
    x_min = x_max = static_cast<float>(input.vertices[0].x());
    y_min = y_max = static_cast<float>(input.vertices[0].y());
    z_min = z_max = static_cast<float>(input.vertices[0].z());

    for (const auto& v : input.vertices)
    {
        x_min = std::min(x_min, static_cast<float>(v.x()));
        x_max = std::max(x_max, static_cast<float>(v.x()));
        y_min = std::min(y_min, static_cast<float>(v.y()));
        y_max = std::max(y_max, static_cast<float>(v.y()));
        z_min = std::min(z_min, static_cast<float>(v.z()));
        z_max = std::max(z_max, static_cast<float>(v.z()));
    }

    input.minPoint = Point3(x_min, y_min, z_min);
    input.maxPoint = Point3(x_max, y_max, z_max);
}

void translateBarycenter(Mesh& input)
{
    //std::cout << "Translating..\n";

    if (input.vertices.empty()) return;

    // translate the unique vertices
    for (auto& v : input.vertices)
    {
        v -= input.barycenter;
    }

    // also translate the centroids of all faces
    for (auto& f : input.faces)
    {
        f.centroid -= input.barycenter;
    }

    Point3 barycenter_new = calculateBarycenter(input);
    input.barycenter = barycenter_new;

    // Update old AABB into new one
    calculateAABB(input);

    if (application_mode == 1)
    {
        //input.updateGLBuffers();
    }
}

void scaleUniformly(Mesh& input)
{
    //std::cout << "Scaling..\n";

    if (input.vertices.empty()) return;

    Point3 minpoint = input.minPoint;
    Point3 maxpoint = input.maxPoint;

    float largest_dim = std::max({maxpoint.e[0] - minpoint.e[0], maxpoint.e[1] - minpoint.e[1], maxpoint.e[2] - minpoint.e[2]});
    if (largest_dim <= 1e-9f || std::isnan(largest_dim)) return;

    double factor = 1.0 / largest_dim;
    for (auto& v : input.vertices)
    {
        v *= factor;
    }

    double total_area = 0;
    for (auto& f : input.faces)
    {
        f.centroid *= factor;
        f.area *= factor * factor; // square because both edge vectors needed to calculate the area are scaled by the factor
    }

    input.area *= factor * factor;
    input.volume *= factor * factor * factor;

    // Update old AABB into new one
    calculateAABB(input);

    if (application_mode == 1)
    {
        //input.updateGLBuffers();
    }
}

// calculate area weighted covariance matrix for all triangles
glm::dmat3 computeAreaWeightedCovariance(const Mesh& input)
{
    glm::dmat3 covMat(0.0);
    double totalArea = 0.0;
    
    for (const auto& f : input.faces)
    {
        if (f.v.size() < 3 || f.area <= 0.0 || std::isnan(f.area))
            continue;
        const Point3& p0 = input.vertices[f.v[0]];
        const Point3& p1 = input.vertices[f.v[1]];
        const Point3& p2 = input.vertices[f.v[2]];
        const Point3& c = f.centroid;
        
        double w = f.area;
        totalArea += w;
        
        // Quadratic integral over the triangle
        for (int row = 0; row < 3; ++row)
        {
            for (int col = 0; col < 3; ++col)
            {
                double term = (p0.e[row] * p0.e[col] + p1.e[row] * p1.e[col] + p2.e[row] * p2.e[col] + 9.0 * c.e[row] * c.e[col]) / 12.0;
                covMat[col][row] += w * term;
            }
        }
    }
    
    // fallback
    if (totalArea <= 1e-12)
    {
        std::vector<glm::dvec3> v;
        v.reserve(input.vertices.size());
        for (const auto& vert : input.vertices )
            v.emplace_back(vert.x(), vert.y(), vert.z());
        glm::dvec3 C(0.0, 0.0, 0.0);
        return glm::computeCovarianceMatrix(v.data(), v.size(), C);
    }
    
    return covMat / totalArea;
}

void alignAxes(Mesh& input)
{
    //std::cout << "Rotating..\n";
    if (input.vertices.empty()) return;

    // Step 1: Compute Covariance Matrix (3x3)
    // Can be calculated by first putting all vertices in a Nx3 matrix V (so each row of the matrix is a xyz of a vertex in the shape)
    // Then we can get C = (1 / N) * (V^t * V), where N is the number of vertices
    
    glm::dmat3 covMat = computeAreaWeightedCovariance(input);
    
    // Step 2: Perform Eigen decomposition
    // We have to solve det(C - l * I) = 0, where I is the identity matrix and l is lambda --> solve for lambda
    // Since C is a 3x3 matrix, we get -l^3 + c2 * lambda^2 + c1 * lambda + c0 = 0 to get the eigenvalues (l1, l2, l3)
    // To get the eigenvectors, we plug the eigenvalues back into the original equations: (C - li * I) * vi = 0,
    // where li is the ith eigenvalue, and vi is the corresponding eigenvector.
    // We can use Gaussian elimination for this. After applying that, we have to normalize the result.
    // Largest eigenvalue means that its corresponding eigenvector represents the longest dimension of the shape --> same principal for the other values/vectors
    
    glm::dvec3 evals(1.0, 1.0, 1.0);
    glm::dmat3 evecs; 
    int count = glm::findEigenvaluesSymReal(covMat, evals, evecs);
    if (count != 3 || std::isnan(evals[0]) || std::isnan(evals[1]) || std::isnan(evals[2])) {
        input.eigenvalues = Vec3(1.0, 1.0, 1.0);
        return;
    }
    
    // Sort from largest to smallest (evlas[0] >= evals[1] >= evals[2])
    glm::sortEigenvalues(evals, evecs);

    glm::dvec3 e1 = glm::normalize(evecs[0]);
    glm::dvec3 e2 = glm::normalize(evecs[1]);
    glm::dvec3 e3 = glm::normalize(evecs[2]);
    
    // Step 3: Put the three eigenvectors together in a 3x3 rotation matrix R
    // The left column should correspond to the eigenvector with the largest eigenvalue --> same idea for the other two, left to right, large to small
    // We can then rotate all the points in our shape by applying R^t * V (the V from step 1), which gives us a matrix of all vertices of the shape, properly rotated
    
    // quarantee righthanded coordinate system
    if (glm::dot(glm::cross(e1, e2), e3) < 0.0) {
        e3 = -e3;
    }
    
    //Rotation Matrix
    glm::dmat3 R(e1, e2, e3);
    glm::dmat3 Rt = glm::transpose(R);
    
    for (auto& v : input.vertices) {
        glm::dvec3 p(v.x(), v.y(), v.z());
        glm::dvec3 pRot = Rt * p;
        v = Point3(pRot.x, pRot.y, pRot.z);
    }

    for (auto& f : input.faces) {
        auto& c = f.centroid;
        glm::dvec3 p(c.x(), c.y(), c.z());
        glm::dvec3 pRot = Rt * p;
        c = Point3(pRot.x, pRot.y, pRot.z);
    }

    // save the eigenvalues
    input.ev1 = Vec3(1.0, 0.0, 0.0);
    input.ev2 = Vec3(0.0, 1.0, 0.0);
    input.ev3 = Vec3(0.0, 0.0, 1.0);
    input.eigenvalues = Vec3(evals[0], evals[1], evals[2]);

    // Update AABB
    calculateAABB(input);
    if (application_mode == 1)
    {
        //input.updateGLBuffers();
    }
}

int sign(float i)
{
    if (i < 0)
        return -1;
    return 1;
}

void flipShape(Mesh& input)
{
    //std::cout << "Flipping..\n";

    // The idea is to have the sides of the shapes that have more mass always be on the same side of some axis
    // We can do this by first calculating the centers of all triangles
    // Then, for x, y and z, we calculate fi = sum_t sign(C t,i)(C t,i)^2
    // where i = {x, y, z}, and (C t, i) is the ith coordinate of center of triangle t
    // Basically, all we do, is for each axis, calculate how much its weights towards either the positive or negative side
    // We can then make a 3x3 matrix F, where the diagonals are sign(fx), sign(fy) and sign(fz)
    // We multiply each set of vertex coordinates with F to apply the flipping
    
    float fx = 0;
    float fy = 0;
    float fz = 0;

    // calculate for each axis area weighted momentum
    for (const auto& face : input.faces)
    {
        double w = (face.area > 0.0 && !std::isnan(face.area)) ? face.area : 0.0;
        
        double cx = face.centroid.x();
        double cy = face.centroid.y();
        double cz = face.centroid.z();

        // Then, for x, y and z, we calculate fi = sum_t sign(C t,i)(C t,i)^2
        fx += w * sign(static_cast<float>(cx)) * (cx * cx);
        fy += w * sign(static_cast<float>(cy)) * (cy * cy);
        fz += w * sign(static_cast<float>(cz)) * (cz * cz);
    }
    
    int sfx = sign(static_cast<float>(fx));
    int sfy = sign(static_cast<float>(fy));
    int sfz = sign(static_cast<float>(fz));

    for (Point3& v : input.vertices)
    {
        v = Point3(sfx * v.x(), sfy * v.y(), sfz * v.z());
    }

    for (auto& f : input.faces)
    {
        auto& c = f.centroid;
        c = Point3(sfx * c.x(), sfy * c.y(), sfz * c.z());
    }

    // Calculate new signs for each axis
    fx = 0; fy = 0; fz = 0;

    for (const auto& face : input.faces)
    {
        double w = (face.area > 0.0 && !std::isnan(face.area)) ? face.area : 0.0;
        double cx = face.centroid.x();
        double cy = face.centroid.y();
        double cz = face.centroid.z();

        fx += w * sign(static_cast<float>(cx)) * (cx * cx);
        fy += w * sign(static_cast<float>(cy)) * (cy * cy);
        fz += w * sign(static_cast<float>(cz)) * (cz * cz);
    }

    input.sx = sign(static_cast<float>(fx));
    input.sy = sign(static_cast<float>(fy));
    input.sz = sign(static_cast<float>(fz));

    // Update old AABB into new one
    calculateAABB(input);
    if (application_mode == 1)
    {
        //input.updateGLBuffers();
    }
}

void normalizeMesh(Mesh& input)
{
    translateBarycenter(input);
    alignAxes(input);
    flipShape(input);
    scaleUniformly(input);

    if (application_mode == 1)
        input.updateGLBuffers();
}

void normalizeMeshes(std::vector<Mesh>& meshes)
{
    int total = static_cast<int>(meshes.size());
    std::cout << "Normalizing " << total << " meshes in parallel...\n";

    #pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < total; ++i)
    {
        normalizeMesh(meshes[i]);
    }

    std::cout << "Finished normalizing meshes.\n";
}

std::vector<std::vector<float>> getDataDeviations()
{
    std::ifstream file(histogramsFilePath);
    std::vector<std::vector<float>> data;
    float sumFaces = 0;
    float sumVerts = 0;
    int total = 0;

    // Read csv file
    if (file.is_open())
    {
        string line;
        std::getline(file, line); // skip headers
        while (std::getline(file, line))
        {
            std::vector<float> row;
            std::stringstream ss(line);

            string word;
            float f;
            // read id
            std::getline(ss, word, ';');
            f = std::stof(word);
            row.push_back(f);

            // read number of faces for current shape
            std::getline(ss, word, ';');
            f = std::stof(word);
            row.push_back(f);
            sumFaces += f;

            // read number of vertices for current shape
            std::getline(ss, word, ';');
            f = std::stof(word);
            row.push_back(f);
            sumVerts += f;

            total++;
            data.push_back(row);
        }
    }

    // calculate the standard deviation for the faces and vertices
    float avgFaces = sumFaces / total;
    float avgVerts = sumVerts / total;
    std::cout << "Average number of vertices: " << avgVerts << std::endl;
    std::cout << "Average number of faces: " << avgFaces << std::endl;

    float facesSTDsum = 0;
    float vertsSTDsum = 0;

    for (int i = 0; i < data.size(); i++)
    {
        facesSTDsum += powf(data[i][1] - avgFaces, 2);
        vertsSTDsum += powf(data[i][2] - avgVerts, 2);
    }

    facesSTDsum /= total;
    vertsSTDsum /= total;

    float sigmaFaces = sqrtf(facesSTDsum);
    float sigmaVerts = sqrtf(vertsSTDsum);

    // calculate for each shape how many standard deviations from average
    for (int i = 0; i < data.size(); i++)
    {
        data[i][1] = fabs(data[i][1] - avgFaces) / sigmaFaces;
        data[i][2] = fabs(data[i][2] - avgVerts) / sigmaVerts;
    }



    return data;
}

// Render functions

void renderObject(fs::path selectedFile)
{
    Parser p;
    Mesh obj = p.parseOFF(selectedFile.string());
    normalizeMesh(obj);
    obj.updateGLBuffers();
    GLsizei vertexCount = static_cast<GLsizei>(obj.trivertsgl.size() / 3);
    std::cout << "Number of vertices: " << obj.vertsSize() << std::endl;
    std::cout << "Number of faces: " << obj.primsSize() << std::endl;

    glfwInit();
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

    // Create window
    GLFWwindow* window = glfwCreateWindow(800, 600, "Multimedia Retrieval", NULL, NULL);
    glfwSetMouseButtonCallback(window, mouse_button_callback);
    glfwSetCursorPosCallback(window, cursor_position_callback);
    glfwSetScrollCallback(window, scroll_callback);
    glfwSetKeyCallback(window, key_callback);

    if (window == NULL)
    {
        std::cout << "Failed to create GLFW window" << std::endl;
        glfwTerminate();
        return;
    }
    glfwMakeContextCurrent(window);

    // Initialise GLAD
    if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress))
    {
        std::cout << "Failed to initialize GLAD" << std::endl;
        return;
    }

    glViewport(0, 0, 800, 600);
    glEnable(GL_DEPTH_TEST);

    // Print controls to console
    std::cout << "\n=== Controls ===\n";
    std::cout << "  Left Mouse Drag  : Orbit camera\n";
    std::cout << "  Right Mouse Drag : Pan camera\n";
    std::cout << "  Scroll Wheel     : Zoom\n";
    std::cout << "  W / A / S / D    : Move camera\n";
    std::cout << "  1                : Toggle Edges\n";
    std::cout << "  2                : Toggle Vertices\n";
    std::cout << "  3                : Toggle Faces\n";
    std::cout << "  L                : 1 Step Loop Subdivision (Upsample)\n";
    std::cout << "  U                : 1 Step Simple Subdivision (Upsample)\n";
    std::cout << "  K                : 1 Step Decimation (Downsample)\n";
    std::cout << "  R                : Auto Resample (target 5k - 10k vertices)\n";
    std::cout << "================\n\n";

	std::cout << "File: " << selectedFile.string() << std::endl;

    // Load in mesh
    std::vector<float> vertices = obj.trivertsgl;

    // Compile and load shaders
    unsigned int vertexShader = glCreateShader(GL_VERTEX_SHADER);
    glShaderSource(vertexShader, 1, &vertexShaderSource, NULL);
    glCompileShader(vertexShader);

    unsigned int fragmentShader = glCreateShader(GL_FRAGMENT_SHADER);
    glShaderSource(fragmentShader, 1, &fragmentShaderSource, NULL);
    glCompileShader(fragmentShader);

    unsigned int shaderProgram = glCreateProgram();
    glAttachShader(shaderProgram, vertexShader);
    glAttachShader(shaderProgram, fragmentShader);
    glLinkProgram(shaderProgram);

    glDeleteShader(vertexShader);
    glDeleteShader(fragmentShader);

    // VAO, VBO
    unsigned int VAO, VBO;
    glGenVertexArrays(1, &VAO);
    glGenBuffers(1, &VBO);

    glBindVertexArray(VAO);

    glBindBuffer(GL_ARRAY_BUFFER, VBO);
    glBufferData(GL_ARRAY_BUFFER, vertices.size() * sizeof(float), vertices.data(), GL_STATIC_DRAW);

    // Data reading
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);

    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindVertexArray(0);

    // Show window
    // Render loop
    int mvpLoc = glGetUniformLocation(shaderProgram, "uMVP");
    int modelViewLoc = glGetUniformLocation(shaderProgram, "uModelView");
    int useLightingLoc = glGetUniformLocation(shaderProgram, "uUseLighting");
    GLint colorLoc = glGetUniformLocation(shaderProgram, "uColor");

    float lastFrameTime = 0.0f;

    while (!glfwWindowShouldClose(window))
    {
        // Calculate delta time
        float currentFrameTime = static_cast<float>(glfwGetTime());
        float deltaTime = currentFrameTime - lastFrameTime;
        lastFrameTime = currentFrameTime;

        // Process camera input
        processCameraKeyboard(window, deltaTime);

        // Process resampling input
        static bool lKeyPressed = false;
        if (glfwGetKey(window, GLFW_KEY_L) == GLFW_PRESS)
        {
            if (!lKeyPressed)
            {
                lKeyPressed = true;
                std::cout << "\n[Resample] Performing 1 step of Loop subdivision..." << std::endl;
                upsampleMeshLoop(obj);
                glBindBuffer(GL_ARRAY_BUFFER, VBO);
                glBufferData(GL_ARRAY_BUFFER, obj.trivertsgl.size() * sizeof(float), obj.trivertsgl.data(), GL_STATIC_DRAW);
                vertexCount = static_cast<GLsizei>(obj.trivertsgl.size() / 3);
                std::cout << "[Resample] Done! Vertices: " << obj.vertsSize() << ", Faces: " << obj.primsSize() << std::endl;
            }
        }
        else if (glfwGetKey(window, GLFW_KEY_L) == GLFW_RELEASE)
        {
            lKeyPressed = false;
        }

        static bool uKeyPressed = false;
        if (glfwGetKey(window, GLFW_KEY_U) == GLFW_PRESS)
        {
            if (!uKeyPressed)
            {
                uKeyPressed = true;
                std::cout << "\n[Resample] Performing 1 step of Simple subdivision..." << std::endl;
                upsampleMeshSimple(obj);
                glBindBuffer(GL_ARRAY_BUFFER, VBO);
                glBufferData(GL_ARRAY_BUFFER, obj.trivertsgl.size() * sizeof(float), obj.trivertsgl.data(), GL_STATIC_DRAW);
                vertexCount = static_cast<GLsizei>(obj.trivertsgl.size() / 3);
                std::cout << "[Resample] Done! Vertices: " << obj.vertsSize() << ", Faces: " << obj.primsSize() << std::endl;
            }
        }
        else if (glfwGetKey(window, GLFW_KEY_U) == GLFW_RELEASE)
        {
            uKeyPressed = false;
        }

        static bool kKeyPressed = false;
        if (glfwGetKey(window, GLFW_KEY_K) == GLFW_PRESS)
        {
            if (!kKeyPressed)
            {
                kKeyPressed = true;
                std::cout << "\n[Resample] Performing 1 step of Downsampling (Decimation)..." << std::endl;
                downsampleMesh(obj);
                glBindBuffer(GL_ARRAY_BUFFER, VBO);
                glBufferData(GL_ARRAY_BUFFER, obj.trivertsgl.size() * sizeof(float), obj.trivertsgl.data(), GL_STATIC_DRAW);
                vertexCount = static_cast<GLsizei>(obj.trivertsgl.size() / 3);
                std::cout << "[Resample] Done! Vertices: " << obj.vertsSize() << ", Faces: " << obj.primsSize() << std::endl;
            }
        }
        else if (glfwGetKey(window, GLFW_KEY_K) == GLFW_RELEASE)
        {
            kKeyPressed = false;
        }

        static bool rKeyPressed = false;
        if (glfwGetKey(window, GLFW_KEY_R) == GLFW_PRESS)
        {
            if (!rKeyPressed)
            {
                rKeyPressed = true;
                std::cout << "\n[Resample] Auto-resampling mesh towards 5k - 10k vertices..." << std::endl;
                resampleMesh(obj);
                glBindBuffer(GL_ARRAY_BUFFER, VBO);
                glBufferData(GL_ARRAY_BUFFER, obj.trivertsgl.size() * sizeof(float), obj.trivertsgl.data(), GL_STATIC_DRAW);
                vertexCount = static_cast<GLsizei>(obj.trivertsgl.size() / 3);
                std::cout << "[Resample] Done! Vertices: " << obj.vertsSize() << ", Faces: " << obj.primsSize() << std::endl;
            }
        }
        else if (glfwGetKey(window, GLFW_KEY_R) == GLFW_RELEASE)
        {
            rKeyPressed = false;
        }

        glClearColor(1.0f, 1.0f, 1.0f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        glUseProgram(shaderProgram); \

            // Get camera position
            glm::vec3 cameraPos, camRight, camUp;
        getCameraVectors(cameraPos, camRight, camUp);

        // Build matrices
        glm::mat4 model = glm::mat4(1.0f);
        glm::mat4 view = glm::lookAt(cameraPos, cameraTarget, glm::vec3(0.0f, 1.0f, 0.0f));
        glm::mat4 modelView = view * model;

        int width, height;
        glfwGetFramebufferSize(window, &width, &height);
        float aspect = (height > 0) ? (float)width / (float)height : 1.0f;
        glm::mat4 projection = glm::perspective(glm::radians(45.0f), aspect, 0.1f, 100.0f);
        glm::mat4 mvp = projection * view * model;

        // Write to shader
        glUniformMatrix4fv(mvpLoc, 1, GL_FALSE, glm::value_ptr(mvp));
        glUniformMatrix4fv(modelViewLoc, 1, GL_FALSE, glm::value_ptr(modelView));

        // Draw models
        glBindVertexArray(VAO);

        // Draw faces (with shading)
        if (showFace)
        {
            glUniform1i(useLightingLoc, 1);
            glUniform4f(colorLoc, 1.0f, 0.5f, 0.2f, 1.0f);
            glDrawArrays(GL_TRIANGLES, 0, vertexCount);
        }

        // Draw edges (unlit)
        if (showEdge)
        {
            glUniform1i(useLightingLoc, 0);
            glUniform4f(colorLoc, 0.2f, 0.1f, 0.05f, 1.0f);
            glLineWidth(2.0f);
            glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
            glDrawArrays(GL_TRIANGLES, 0, vertexCount);
            glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
        }

        // Draw nodes (unlit)
        if (showNode)
        {
            glUniform1i(useLightingLoc, 0);
            glUniform4f(colorLoc, 1.0f, 0.0f, 0.0f, 1.0f);
            glPointSize(4.0f);
            glDrawArrays(GL_POINTS, 0, vertexCount);
        }


        glfwSwapBuffers(window);
        glfwPollEvents();
    }

    // Clean memory
    glDeleteVertexArrays(1, &VAO);
    glDeleteBuffers(1, &VBO);
    glDeleteProgram(shaderProgram);

    glfwTerminate();
}

void promptRenderObject()
{
    fs::path selectedFile = selectModelFile("ObjectFiles");
    if (selectedFile.empty())
    {
        std::cout << "Error: File is empty.\n";
        promptRenderObject();
    }

    std::cout << "Loading file: " << selectedFile.string() << "\n";
    renderObject(selectedFile);
    promptRenderObject();
}

void renderObject(int id)
{
    fs::path path = paths[id];
    renderObject(path);
}

void renderObjects(std::vector<int> ids)
{
    for (int id : ids)
    {
        renderObject(id);
    }
}

// Main functions

int main()
{
    std::vector<int> obj_counts = makeidpathcsv();
    paths = readPaths();

    omp_set_num_threads(omp_get_max_threads());

    std::cout << "Multimedia Retrieval tool\n";
    std::cout << "Select mode to begin: \n";
    std::cout << "1. Rendering mode\n";
    std::cout << "2. Feature extraction mode\n";
    std::cout << "3. Histogram export mode\n";
    std::cout << "4. Outlier mode \n";
    std::cout << "5. Query mode \n";
    std::cout << "6. Evaluation mode \n";
    std::cout << "7. project mode \n";

    std::cin >> application_mode;


    // Rendering mode
    if (application_mode == 1)
    {
        promptRenderObject();
    }

    // Feature extraction mode
    else if (application_mode == 2)
    {
        // Analyse initial database
        fs::path objectFiles = "ObjectFiles";
        std::vector<Mesh> meshes = parseDatabase(objectFiles);
        //analyseDatabase(meshes);

        // Resample meshes
        resampleMeshes(meshes);
        //analyseDatabase(meshes);

        // Precompute values of the meshes
        precomputeMeshes(meshes);
        //analyseDatabase(meshes);

        // Normalization --> remeshing, translation, (rotation), (flipping), scaling
        normalizeMeshes(meshes);
        //analyseDatabase(meshes);

        // Feature extraction for all meshes
        std::vector<std::vector<double>> elem_desc;
        featureExtraction(meshes, elem_desc);
        
        // save raw features
        exportFeaturesToCSV(meshes, elem_desc, "raw_features.csv");
        
        // create and save min-max normalized features
        auto norm_desc = elem_desc;
        normalizeFeatures(norm_desc);
        exportFeaturesToCSV(meshes, norm_desc, fullFeaturesNormFilePath.string());

        // create and save z-score normalized features
        auto stand_desc = elem_desc;
        standardizeFeatures(stand_desc);
        exportFeaturesToCSV(meshes, stand_desc, fullFeaturesStandFilePath.string());
    }
    
    // Histogram mode
    else if (application_mode == 3)
    {
        runHistogramMode();
    }

    // Outlier mode
    else if (application_mode == 4) 
    {
        std::cout << "Reading .csv file..." << std::endl;
        std::vector<std::vector<float>> dataDeviations = getDataDeviations();
        std::cout << "Read csv and calculated standard deviations" << std::endl;
        // after sorting, low number of deviations away from average are low indeces and high number of deviations away are high indeces
        int index = 2; // 1 = faces, 2 = vertices

        quicksort(dataDeviations, index, 0, dataDeviations.size() - 1);

        std::vector<int> outliers;
        outliers.push_back(dataDeviations[0][0]);
        for (int i = 0; dataDeviations[dataDeviations.size() - 1 - i][index] >= 3; i++)
        {
            int elem = dataDeviations.size() - 1 - i;
            outliers.push_back(dataDeviations[elem][0]);
        }

        renderObjects(outliers);
    }

    // Querying mode
	else if (application_mode == 5)
	{
		fs::path path = promptPathOrSelect();

        Parser p;
		Mesh queryMesh = p.parseOFF(path.string());

        resampleMesh(queryMesh);
        precomputeMesh(queryMesh);
		normalizeMesh(queryMesh);
		renderObject(path);
		
        auto queryFeatures = featureExtract(queryMesh);
        auto temp = queryFeatures;
	    StandardizationData stdData = readStandardizationData(standardizationDataFilePath.string());
	    standardizeWithData(queryFeatures, stdData);
	    normalizeHistogram(queryFeatures);
	    FeatureData featureData = readFeatures(fullFeaturesStandFilePath);
	    
        int N = static_cast<int>(featureData.features.size());
        int D = (N > 0) ? static_cast<int>(featureData.features[0].size()) : 0;
        if (N == 0 || D == 0)
        {
            std::cerr << "Error: No features found in database.\n";
            return 0;
        }

        std::cout << "\n=== Select Search Method ===\n";
        std::cout << "1. ANN (Approximate Nearest Neighbors kd-tree)\n";
        std::cout << "2. Linear / Brute-force Search (makeDistanceVector + quicksort)\n";
        std::cout << "Enter choice (1 or 2): ";
        int searchMethod = 1;
        std::cin >> searchMethod;

        if (searchMethod == 1)
        {
            FeatureWeights weights = readWeights();
            double weightSum = 0.0;
            for (double wVal : weights) weightSum += wVal;
            if (weightSum < 1e-9)
            {
                weights = computeWeightsFromDatabase(featureData.features);
                saveWeights(weights);
            }
            
            ANNpointArray dataPts = annAllocPts(N, D);
            for (int n = 0; n < N; n++)
            {
                std::vector<double> embededRow = embedFeatureForANN(featureData.features[n], weights);
                for (int d = 0; d < D; d++)
                {
                    dataPts[n][d] = embededRow[d];
                }
            }

            ANNkd_tree* kdTree = new ANNkd_tree(dataPts, N, D);
            
            std::vector<double> embededQuery = embedFeatureForANN(queryFeatures, weights);

            ANNpoint queryPt = annAllocPt(D);
            for (int d = 0; d < D; d++)
            {
                queryPt[d] = embededQuery[d];
            }

            std::cout << "\nSelect ANN search type:\n";
            std::cout << "  k - K-nearest neighbors\n";
            std::cout << "  r - Radius / Range search\n";
            std::cout << "Choice (k/r): ";
            std::string searchType;
            std::cin >> searchType;

            std::vector<int> results;

            if (searchType == "k" || searchType == "K")
            {
                std::cout << "Enter how many results (K): ";
                int k = 10;
                std::cin >> k;
                if (k > N) k = N;
                if (k < 1) k = 1;

                ANNidxArray nnIdx = new ANNidx[k];
                ANNdistArray dists = new ANNdist[k];

                kdTree->annkSearch(queryPt, k, nnIdx, dists, 0.0);

                std::cout << "\n--- ANN K-NN Search Results (K = " << k << ") ---\n";
                for (int i = 0; i < k; i++)
                {
                    if (nnIdx[i] != ANN_NULL_IDX && nnIdx[i] >= 0 && nnIdx[i] < N)
                    {
                        results.push_back(nnIdx[i]);
                        float dist = std::sqrt(static_cast<float>(dists[i]));
                        std::cout << (i + 1) << ". file " << (nnIdx[i] < (int)paths.size() ? paths[nnIdx[i]] : std::to_string(nnIdx[i]))
                                  << "  dist " << dist
                                  << "  label " << featureData.labels[nnIdx[i]] << "\n";
                    }
                }

                delete[] nnIdx;
                delete[] dists;
            }
            else if (searchType == "r" || searchType == "R")
            {
                double r = 1.0;
                std::cout << "Enter search radius (R): ";
                std::cin >> r;

                int maxK = N;
                std::cout << "Enter max number of results to return (1-" << N << ", default " << N << "): ";
                int inputMax;
                if (std::cin >> inputMax && inputMax > 0 && inputMax <= N)
                {
                    maxK = inputMax;
                }

                ANNidxArray nnIdx = new ANNidx[maxK];
                ANNdistArray dists = new ANNdist[maxK];
                double sqRad = r * r;

                // annkFRSearch retrieves points within distance R (squared distance <= sqRad)
                int ptsFound = kdTree->annkFRSearch(queryPt, sqRad, maxK, nnIdx, dists, 0.0);

                std::cout << "\n--- ANN Radius Search Results (R = " << r << ", found: " << ptsFound << ") ---\n";
                int toShow = std::min(ptsFound, maxK);
                for (int i = 0; i < toShow; i++)
                {
                    if (nnIdx[i] != ANN_NULL_IDX && nnIdx[i] >= 0 && nnIdx[i] < N)
                    {
                        results.push_back(nnIdx[i]);
                        float dist = std::sqrt(static_cast<float>(dists[i]));
                        std::cout << (i + 1) << ". file " << (nnIdx[i] < (int)paths.size() ? paths[nnIdx[i]] : std::to_string(nnIdx[i]))
                                  << "  dist " << dist
                                  << "  label " << featureData.labels[nnIdx[i]] << "\n";
                    }
                }

                delete[] nnIdx;
                delete[] dists;
            }
            else
            {
                std::cout << "Invalid search type selected.\n";
            }

            delete kdTree;
            annDeallocPts(dataPts);
            annDeallocPt(queryPt);
            annClose();

            if (!results.empty())
            {
                renderObjects(results);
            }
            else
            {
                std::cout << "No matching shapes found to render.\n";
            }
        }
        else // Option 2: Linear / Brute-force search
        {
            saveWeights(computeWeightsFromDatabase(featureData.features));

            std::vector<std::vector<float>> distances = makeDistanceVector(featureData.features, queryFeatures, readWeights());

            quicksort(distances, 1, 0, distances.size() - 1);

            std::cout << "Enter how many results: ";
            int k = 10;
            std::cin >> k;
            if (k > (int)distances.size()) k = (int)distances.size();
            if (k < 1) k = 1;

            std::cout << "\n--- Linear Search Results (K = " << k << ") ---\n";
            std::vector<int> kbest;
            for (int i = 0; i < k; i++)
            {
                int idx = (int)distances[i][0];
                kbest.push_back(idx);
                std::cout << (i + 1) << ". file " << (idx < (int)paths.size() ? paths[idx] : std::to_string(idx))
                          << "  dist " << distances[i][1]
                          << "  label " << featureData.labels[idx] << "\n";
            }

            renderObjects(kbest);
        }
	} 
    // Evaluation mode
    else if (application_mode == 6)
    {
        std::cout << "Evaluating system..\n";
        std::cout << "Enter evaluation metric (1-2483): \n";
        int metric;
        std::cin >> metric;
        if (metric < 1) metric = 1;

        std::cout << "\n=== Select Evaluation Search Method ===\n";
        std::cout << "1. ANN (Approximate Nearest Neighbors kd-tree)\n";
        std::cout << "2. Linear / Brute-force Search (makeDistanceVector + quicksort)\n";
        std::cout << "Enter choice (1 or 2): ";
        int evalMethod = 2;
        std::cin >> evalMethod;

        // Prepare database
        auto database = readFeatures(fullFeaturesStandFilePath);
        auto& db_feats = database.features;
        int N = static_cast<int>(db_feats.size());
        int D = (N > 0) ? static_cast<int>(db_feats[0].size()) : 0;
        if (N == 0 || D == 0)
        {
            std::cerr << "Error: No features found in database.\n";
            return 0;
        }

        FeatureWeights weights = computeWeightsFromDatabase(db_feats);
        saveWeights(weights);

        // Precompute ranked neighbor indices (sorted closest -> furthest) for each query i
        std::vector<std::vector<int>> ranked_indices(N, std::vector<int>(N));

        if (evalMethod == 1)
        {
            double eps = 0.0;
            std::cout << "Enter ANN epsilon (0.0 = exact kd-tree, e.g. 0.1 = approximate): ";
            std::cin >> eps;

            std::cout << "Building ANN kd-tree and ranking all " << N << " queries...\n";
            ANNpointArray dataPts = annAllocPts(N, D);
            for (int n = 0; n < N; n++)
            {
                std::vector<double> embededRow = embedFeatureForANN(db_feats[n], weights);
                for (int d = 0; d < D; d++)
                {
                    dataPts[n][d] = embededRow[d];
                }
            }

            ANNkd_tree* kdTree = new ANNkd_tree(dataPts, N, D);
            ANNpoint queryPt = annAllocPt(D);
            ANNidxArray nnIdx = new ANNidx[N];
            ANNdistArray dists = new ANNdist[N];

            for (int i = 0; i < N; i++)
            {
                std::vector<double> embededQuery = embedFeatureForANN(db_feats[i], weights);
                for (int d = 0; d < D; d++)
                {
                    queryPt[d] = embededQuery[d];
                }

                kdTree->annkSearch(queryPt, N, nnIdx, dists, eps);
                for (int j = 0; j < N; j++)
                {
                    ranked_indices[i][j] = (nnIdx[j] != ANN_NULL_IDX && nnIdx[j] >= 0 && nnIdx[j] < N) ? nnIdx[j] : j;
                }
            }

            delete[] nnIdx;
            delete[] dists;
            delete kdTree;
            annDeallocPts(dataPts);
            annDeallocPt(queryPt);
            annClose();
        }
        else
        {
            std::cout << "Ranking all " << N << " queries using Linear Search + Quicksort in parallel...\n";
            #pragma omp parallel for schedule(static)
            for (int i = 0; i < N; i++)
            {
                auto distances = makeDistanceVector(db_feats, db_feats[i], weights);
                quicksort(distances, 1, 0, distances.size() - 1);
                for (int j = 0; j < N; j++)
                {
                    ranked_indices[i][j] = static_cast<int>(distances[j][0]);
                }
            }
        }

        std::vector<std::pair<float, float>> plot_points;
        std::vector<std::vector<std::pair<float, float>>> plot_points_per_class(69);

        // Build k steps (always include N as final point so ROC reaches Sensitivity = 1.0)
        std::vector<int> k_steps;
        for (int k = 0; k <= N; k += metric)
        {
            k_steps.push_back(k);
        }
        if (k_steps.empty() || k_steps.back() != N)
        {
            k_steps.push_back(N);
        }

        // k = number of items returned in the query
        for (int k : k_steps)
        {
            int per = (((float)k) / (float)N) * 100.0f;
            std::cout << "Evaluating k = " << k << " (" << per << "%)\n";

            int current_class = 0;
            int current_obj = 0;

            float total_tp = 0, total_fp = 0, total_fn = 0, total_tn = 0;
            float class_tp = 0, class_fp = 0, class_fn = 0, class_tn = 0;

            // For each item in the database, match it against the entire database and calculate sens/spec
            for (int i = 0; i <= N; i++)
            {
                // Onto new class, so save results of previous class
                if (current_obj >= obj_counts[current_class])
                {
                    float sens_class = class_tp / (class_tp + class_fn);
                    float spec_class = class_tn / (class_tn + class_fp);

                    plot_points_per_class[current_class].push_back(std::make_pair(sens_class, spec_class));

                    class_tp = 0;
                    class_fp = 0;
                    class_fn = 0;
                    class_tn = 0;

                    current_obj = 0;
                    current_class++;

                    if (i == N) break;
                }

                float tp = 0, fn = 0, fp = 0, tn = 0;

                const auto& query_label = database.labels[i];

                // Get the k-best items based on the precomputed ranking
                for (int j = 0; j < k; j++)
                {
                    int id = ranked_indices[i][j];
                    const auto& obj_label = database.labels[id];

                    // True positive if labels are identical, false positive otherwise
                    if (query_label == obj_label)
                        tp++;
                    else fp++;
                }

                // Get number of objects in the class of the queried object
                auto num_objects_query_class = obj_counts[current_class];
                fn = num_objects_query_class - tp; // False negatives is the number of objects not returned in the query that are in the class of the queried object
                tn = N - tp - fp - fn; // True negatives are everything except the tps, fps and fns

                total_tp += tp;
                total_fp += fp;
                total_fn += fn;
                total_tn += tn;

                class_tp += tp;
                class_fp += fp;
                class_fn += fn;
                class_tn += tn;

                current_obj++;
            }

            // Calculate sens and spec over the entire database for some value k
            float sens = total_tp / (total_tp + total_fn);
            float spec = total_tn / (total_tn + total_fp);

            plot_points.push_back(std::make_pair(sens, spec));
        }

        // We can get area under curve by using the trapezoidal rule
        // First, we calculate the area of the trapezoid between every two consecutive points
        // Then, sum all trapezoid areas
        float total_area = 0;
        for (int i = 0; i < plot_points.size() - 1; i++)
        {
            auto& p1 = plot_points[i];
            auto& p2 = plot_points[i + 1];

            total_area += ((p1.second + p2.second) / 2) * (p2.first - p1.first);
        }

        std::vector<float> total_area_per_class;
        for (int i = 0; i < plot_points_per_class.size(); i++)
        {
            auto& v = plot_points_per_class[i];
            float total_area_class = 0;
            for (int j = 0; j < v.size() - 1; j++)
            {
                auto& p1 = v[j];
                auto& p2 = v[j + 1];

                total_area_class += ((p1.second + p2.second) / 2) * (p2.first - p1.first);
            }
            total_area_per_class.push_back(total_area_class);
        }

        std::cout << "Evaluation score: " << total_area << "\n";

        // Save to CSV
        std::cout << "Save to CSV file? y/n:\n";
        char c;
        std::cin >> c;

        if (c == 'y')
        {
            std::cout << "Enter filename:\n";
            string filename;
            std::cin >> filename;

            if (filename.size() > 0)
            {
                exportEvaluationToCSV(plot_points, total_area, filename);
                exportEvaluationPerClassToCSV(plot_points_per_class, total_area_per_class, database.labels, obj_counts, filename + "_per_class");
            }
                
            else std::cout << "Cannot save to CSV file.\n";
        }

        std::cout << "Finished evaluation of system.\n";
    }
    // Projection mode
    else if (application_mode == 7)
	{
		std::cout << "\n=== Mode 6: 2D t-SNE Projection & Scatterplot ===\n";
		std::vector<double> dummy;
		FeatureData featureData = promptNormOrStandFeatures(dummy);

		auto points = Projection::getOrComputeEmbedding(featureData.features, featureData.labels, "tsne_embedding.csv");
		if (!points.empty())
		{
			Projection::runScatterplot(points, paths);
		}
	}
	
    return 0;
}





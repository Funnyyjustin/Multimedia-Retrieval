#include "resampling.h"
#include "vec3.h"

const int minVertCount = 5000;
const int maxVertCount = 10000;
const float subdivideFraction = 0.5;

void resampleMesh(Mesh& input)
{
    // TODO: step 2.3
    // First, check if the mesh needs to be resampled --> set limits for #vertices that do not suffice
    // According to the assignment, a good #vertices (or #samples) for a mesh is around 5k - 10k, but more research on this is needed
    // If no need for resampling, return
    // Else, resample according to the given algorithms in the assignment
    // Upscale if too few, downscale if too many
    if (input.vertsSize() < minVertCount)
    {
        while (input.vertsSize() < minVertCount)
        {
            int previousCount = input.vertsSize();
            upsampleMeshSimple(input);

            if (input.vertsSize() <= previousCount)
                break;
        }
    } else if (input.vertsSize() > maxVertCount)
    {
        while (input.vertsSize() > maxVertCount)
        {
            int previousCount = input.vertsSize();
            downsampleMesh(input);

            if (input.vertsSize() >= previousCount)
                break;
        }
    }
    
    
}   

void upsampleMeshSimple(Mesh& input)
{
    float avgArea = computeAverageArea(input);
    float minArea = subdivideFraction * avgArea;
    std::vector<Primitive> new_primitives;
    
    for (const Primitive& prim : input.primitives)
    {
        const float area = computeArea(prim);
        if (area > minArea)
        {
            if (prim.vertices.size() == 4)
            {
                const Point3& v0 = prim.vertices[0]; const Point3& v1 = prim.vertices[1]; const Point3& v2 = prim.vertices[2]; const Point3& v3 = prim.vertices[3];
                const Primitive tri1 = Primitive({v0, v1, v3}, {}); const Primitive tri2 = Primitive({v1, v2, v3}, {});
                new_primitives.push_back(tri1);
                new_primitives.push_back(tri2);
            }
            const Point3& v0 = prim.vertices[0]; const Point3& v1 = prim.vertices[1]; const Point3& v2 = prim.vertices[2];
            const Point3 center = (v0 + v1 + v2) / 3.0f;
            new_primitives.push_back(Primitive({v0, v1, center}, {}));
            new_primitives.push_back(Primitive({v1, v2, center}, {}));
            new_primitives.push_back(Primitive({v2, v0, center}, {}));
            
        } else
        {
            new_primitives.push_back(prim);
        }
    }
    input.primitives = new_primitives;
    input.updateGLBuffers();
}

void upsampleMeshLoop(Mesh& input)
{
    
}
void downsampleMesh(Mesh& input)
{
    
}

float computeArea(const Primitive& prim)
{
    const Vec3 v0 = prim.vertices[0]; const Vec3 v1 = prim.vertices[1]; const Vec3 v2 = prim.vertices[2];
    float area = cross(v1 - v0, v2 - v0).length() * 0.5f;
    if (prim.vertices.size() == 4)
    {
        const Vec3 v3 = prim.vertices[3];
        area += cross(v2 - v0, v3 - v0).length() * 0.5f;
    }
    
    return area;
}

float computeAverageArea(const Mesh& mesh)
{
    float totalArea = 0;
    for (const Primitive& prim : mesh.primitives)
    {
        totalArea += computeArea(prim);
    }
    return totalArea / mesh.primitives.size();
}
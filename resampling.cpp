#include "resampling.h"
#include <unordered_map>
#include <cstdint>
#include <cmath>
#include <algorithm>
#include <queue>
#include <unordered_set>
#include "vec3.h"

const float subdivideFraction = 0.5f;

struct FastEdge
{
    int u = -1, v = -1;
    int w1 = -1, w2 = -1;
    int newVertexIdx = -1;
};

struct DecimateEdge 
{
    int u;
    int v;
    float cost;
    Point3 targetPos; 
};

inline uint64_t makeEdgeKey(int u, int v)
{
    if (u > v) std::swap(u, v);
    return (static_cast<uint64_t>(u) << 32) | static_cast<uint32_t>(v);
}

// Alpha formula from the loop sample code
inline float calculateAlpha(int n)
{
    if (n > 3)
    {
        float center = 0.375f + (0.25f * std::cos(6.2831853f / static_cast<float>(n)));
        return (0.625f - (center * center)) / static_cast<float>(n);
    }
    return 0.1875f; // 3/16
}

void resampleMesh(Mesh& input, int minVert, int maxVert)
{
    if (minVert > maxVert) std::swap(minVert, maxVert);

    // 1. If mesh has fewer vertices than minVert, upsample until it reaches at least minVert
    while (static_cast<int>(input.vertices.size()) < minVert)
    {
        int previousCount = input.vertsSize();
        // upsampleMeshSimple(input);
        upsampleMeshLoop(input);

        // Break if upsampling did not increase vertex count to prevent an infinite loop
        if (input.vertsSize() <= previousCount)
            break;
    } 

    // 2. If mesh has more vertices than maxVert (either originally or after upsampling overshot),
    // downsample until it is within [minVert, maxVert].
    while (static_cast<int>(input.vertices.size()) > maxVert)
    {
        int previousCount = input.vertsSize();
        downsampleMesh(input, minVert);

        // Break if decimation cannot remove any more vertices to prevent an infinite loop
        if (input.vertsSize() >= previousCount)
            break;
    }
}   

void upsampleMeshSimple(Mesh& input)
{
    float avgArea = computeAverageArea(input);
    float minArea = subdivideFraction * avgArea;

    std::vector<Face> new_faces;
    new_faces.reserve(input.faces.size() * 3);

    std::vector<Point3> new_vertices = input.vertices;
    new_vertices.reserve(input.vertices.size() + input.faces.size());

    for (const auto& f : input.faces)
    {
        const Point3& v0 = input.vertices[f.v[0]];
        const Point3& v1 = input.vertices[f.v[1]];
        const Point3& v2 = input.vertices[f.v[2]];
        float area = static_cast<float>(cross(v1 - v0, v2 - v0).length() * 0.5);
        if (f.v.size() == 4)
        {
            const Point3& v3 = input.vertices[f.v[3]];
            area += static_cast<float>(cross(v2 - v0, v3 - v0).length() * 0.5);
        }

        if (area > minArea)
        {
            if (f.v.size() == 4)
            {
                // Splits quad into 2 triangles
                new_faces.push_back(Face{ {f.v[0], f.v[1], f.v[3]} });
                new_faces.push_back(Face{ {f.v[1], f.v[2], f.v[3]} });
                continue;
            }

            // Calculate center and add to new_vertices
            Point3 center = (v0 + v1 + v2) / 3.0f;
            int cIdx = static_cast<int>(new_vertices.size());
            new_vertices.push_back(center);

            // Add the 3 new faces
            new_faces.push_back(Face{ {f.v[0], f.v[1], cIdx} });
            new_faces.push_back(Face{ {f.v[1], f.v[2], cIdx} });
            new_faces.push_back(Face{ {f.v[2], f.v[0], cIdx} });
        }
        else
        {
            new_faces.push_back(f);
        }
    }

    input.vertices = std::move(new_vertices);
    input.faces = std::move(new_faces);
    input.updateGLBuffers();
}

void upsampleMeshLoop(Mesh& input)
{
    // Get the vertices and faces
    const auto& oldVerts = input.vertices;
    const auto& oldFaces = input.faces;

    std::unordered_map<uint64_t, FastEdge> edgeMap;
    edgeMap.reserve(oldFaces.size() * 2);

    // 1. Fill edges
    for (const auto& f : oldFaces) {
        for (int i = 0; i < 3; ++i) {
            int u = f.v[i];
            int v = f.v[(i + 1) % 3];
            int w = f.v[(i + 2) % 3];

            uint64_t key = makeEdgeKey(u, v);
            auto& edge = edgeMap[key];
            edge.u = std::min(u, v);
            edge.v = std::max(u, v);

            if (edge.w1 == -1) edge.w1 = w;
            else edge.w2 = w;
        }
    }

    // Update neighbours and edge points
    std::vector<Point3> neighborSum(oldVerts.size(), Point3(0, 0, 0));
    std::vector<int> valence(oldVerts.size(), 0);
    std::vector<std::vector<int>> boundaryNeighbors(oldVerts.size());
    std::vector<Point3> newPositions;
    newPositions.reserve(oldVerts.size() + edgeMap.size());
    newPositions = oldVerts;

    // 2. Calculate edge-points and sum neighbours
    for (auto& pair : edgeMap) {
        FastEdge& edge = pair.second;
        int u = edge.u;
        int v = edge.v;
        
        if (edge.w2 == -1) {
            boundaryNeighbors[u].push_back(v);
            boundaryNeighbors[v].push_back(u);
        }
        
        const Point3& Pu = oldVerts[u];
        const Point3& Pv = oldVerts[v];

        neighborSum[u] += Pv; valence[u]++;
        neighborSum[v] += Pu; valence[v]++;

        Point3 edgePoint;
        if (edge.w2 != -1) {
            const Point3& Pw1 = oldVerts[edge.w1];
            const Point3& Pw2 = oldVerts[edge.w2];
            edgePoint = (Pu + Pv) * 0.375 + (Pw1 + Pw2) * 0.125;
        } else {
            edgePoint = (Pu + Pv) * 0.5;
        }

        edge.newVertexIdx = static_cast<int>(newPositions.size());
        newPositions.push_back(edgePoint);
    }

    // 3. Vertex smoothing
    for (size_t i = 0; i < oldVerts.size(); ++i) {
        if (boundaryNeighbors[i].size() == 2) {
            int b1 = boundaryNeighbors[i][0];
            int b2 = boundaryNeighbors[i][1];
            newPositions[i] = oldVerts[i] * 0.75f + (oldVerts[b1] + oldVerts[b2]) * 0.125f;
        } else if (!boundaryNeighbors[i].empty()) {
            newPositions[i] = oldVerts[i];
        } else {
            int n = valence[i];
            if (n > 0) {
                float alpha = calculateAlpha(n);
                newPositions[i] = oldVerts[i] * (1.0f - n * alpha) + neighborSum[i] * alpha;
            }
        }
    }

    // Create new faces
    std::vector<Face> newFaces;
    newFaces.reserve(oldFaces.size() * 4);

    for (const auto& f : oldFaces) {
        int v0 = f.v[0], v1 = f.v[1], v2 = f.v[2];
        int e0 = edgeMap[makeEdgeKey(v0, v1)].newVertexIdx;
        int e1 = edgeMap[makeEdgeKey(v1, v2)].newVertexIdx;
        int e2 = edgeMap[makeEdgeKey(v2, v0)].newVertexIdx;

        newFaces.push_back(Face{ {v0, e0, e2} });
        newFaces.push_back(Face{ {v1, e1, e0} });
        newFaces.push_back(Face{ {v2, e2, e1} });
        newFaces.push_back(Face{ {e0, e1, e2} });
    }

    // Update mesh with new geometry
    input.vertices = std::move(newPositions);
    input.faces = std::move(newFaces);
    input.updateGLBuffers();
}

void downsampleMesh(Mesh& input, int targetVertices)
{
    if (input.vertices.empty() || input.faces.empty()) return;

    int currentVertices = static_cast<int>(input.vertices.size());
    if (targetVertices > 0 && currentVertices <= targetVertices) return;

    // 1. Build vertex-to-face adjacency list
    std::vector<std::vector<int>> vertexFaces(input.vertices.size());
    for (int fIdx = 0; fIdx < static_cast<int>(input.faces.size()); ++fIdx)
    {
        for (int vIdx : input.faces[fIdx].v)
        {
            vertexFaces[vIdx].push_back(fIdx);
        }
    }

    // 2. Setup priority queue (min-heap ordered by edge collapse cost)
    auto compareEdges = [](const DecimateEdge& a, const DecimateEdge& b) {
        return a.cost > b.cost;
    };
    std::priority_queue<DecimateEdge, std::vector<DecimateEdge>, decltype(compareEdges)> pq(compareEdges);

    // 3. Collect unique edges and populate the priority queue
    std::unordered_set<uint64_t> seenEdges;
    for (const auto& face : input.faces)
    {
        int numV = static_cast<int>(face.v.size());
        for (int i = 0; i < numV; ++i)
        {
            int u = face.v[i];
            int v = face.v[(i + 1) % numV];
            uint64_t key = makeEdgeKey(u, v);

            if (seenEdges.insert(key).second)
            {
                int minIdx = std::min(u, v);
                int maxIdx = std::max(u, v);
                Point3 target = (input.vertices[minIdx] + input.vertices[maxIdx]) * 0.5;
                float cost = static_cast<float>((input.vertices[minIdx] - input.vertices[maxIdx]).length());
                pq.push(DecimateEdge{ minIdx, maxIdx, cost, target });
            }
        }
    }

    // 4. Track removed vertices and faces
    std::vector<bool> vertexRemoved(input.vertices.size(), false);
    std::vector<bool> faceRemoved(input.faces.size(), false);

    // Reduce vertices by up to 20% per step, respecting targetVertices lower bound
    int stepTarget;
    if (targetVertices > 0)
    {
        stepTarget = std::max(targetVertices, static_cast<int>(currentVertices * 0.8f));
    }
    else
    {
        stepTarget = std::max(1, static_cast<int>(currentVertices * 0.8f));
    }

    // 5. Edge collapse decimation loop
    while (currentVertices > stepTarget && !pq.empty())
    {
        DecimateEdge edge = pq.top();
        pq.pop();

        int u = edge.u;
        int v = edge.v;

        // Skip if either vertex has already been collapsed
        if (vertexRemoved[u] || vertexRemoved[v])
            continue;

        // Update vertex position to the collapsed target midpoint
        input.vertices[v] = edge.targetPos;

        // Update faces adjacent to vertex u
        for (int fIdx : vertexFaces[u])
        {
            if (faceRemoved[fIdx]) continue;
            auto& face = input.faces[fIdx];

            // If face contains both u and v, it degenerates into a line segment
            bool containsV = false;
            for (int idx : face.v)
            {
                if (idx == v)
                {
                    containsV = true;
                    break;
                }
            }

            if (containsV)
            {
                faceRemoved[fIdx] = true;
            }
            else
            {
                // Replace vertex u with v in remaining adjacent faces
                for (int& idx : face.v)
                {
                    if (idx == u) idx = v;
                }
                vertexFaces[v].push_back(fIdx);
            }
        }

        // Mark vertex u as removed
        vertexRemoved[u] = true;
        currentVertices--;
    }

    // 6. Compact mesh and rebuild contiguous indices
    std::vector<Point3> newVertices;
    std::vector<int> indexMap(input.vertices.size(), -1);

    for (size_t i = 0; i < input.vertices.size(); ++i)
    {
        if (!vertexRemoved[i])
        {
            indexMap[i] = static_cast<int>(newVertices.size());
            newVertices.push_back(input.vertices[i]);
        }
    }

    std::vector<Face> newFaces;
    for (size_t i = 0; i < input.faces.size(); ++i)
    {
        if (!faceRemoved[i])
        {
            Face f = input.faces[i];
            for (int& idx : f.v)
            {
                idx = indexMap[idx];
            }
            newFaces.push_back(f);
        }
    }

    input.vertices = std::move(newVertices);
    input.faces = std::move(newFaces);
    input.updateGLBuffers();
}

float computeAverageArea(const Mesh& mesh)
{
    if (mesh.faces.empty()) return 0.0f;
    float totalArea = 0.0f;
    for (const auto& f : mesh.faces)
    {
        const Point3& v0 = mesh.vertices[f.v[0]];
        const Point3& v1 = mesh.vertices[f.v[1]];
        const Point3& v2 = mesh.vertices[f.v[2]];
        float area = static_cast<float>(cross(v1 - v0, v2 - v0).length() * 0.5);
        if (f.v.size() == 4)
        {
            const Point3& v3 = mesh.vertices[f.v[3]];
            area += static_cast<float>(cross(v2 - v0, v3 - v0).length() * 0.5);
        }
        totalArea += area;
    }
    return totalArea / static_cast<float>(mesh.faces.size());
}

#pragma once
#include <vector>
#include <string>

struct ProjectedPoint {
    int id;
    std::string label;
    float x; // Normalized to [-0.85, 0.85] for OpenGL
    float y;
    float r, g, b; // Color per class
};

class Projection {
public:
    // 1. Executes t-SNE or loads the cached CSV
    static std::vector<ProjectedPoint> getOrComputeEmbedding(
        const std::vector<std::vector<double>>& features,
        const std::vector<std::string>& labels,
        const std::string& cachePath = "tsne_embedding.csv",
        double perplexity = 50.0,
        double theta = 0.5,
        int maxIter = 5000
    );

    // 2. Opens the interactive 2D scatterplot window with brushing
    static void runScatterplot(
        const std::vector<ProjectedPoint>& points,
        const std::vector<std::string>& paths
    );
};

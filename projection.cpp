#include "projection.h"
#include "tsne/tsne.h"
#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <array>
#include <iostream>
#include <fstream>
#include <sstream>
#include <cmath>
#include <map>
#include <algorithm>
#include <iomanip>

static void hsvToRgb(double h, double s, double v, float& r, float& g, float& b)
{
    if (s <= 0.0) {
        r = g = b = static_cast<float>(v);
        return;
    }
    while (h >= 360.0) h -= 360.0;
    while (h < 0.0) h += 360.0;

    double c = v * s;
    double x = c * (1.0 - std::abs(std::fmod(h / 60.0, 2.0) - 1.0));
    double m = v - c;

    double r1 = 0, g1 = 0, b1 = 0;
    if (h < 60.0) { r1 = c; g1 = x; b1 = 0; }
    else if (h < 120.0) { r1 = x; g1 = c; b1 = 0; }
    else if (h < 180.0) { r1 = 0; g1 = c; b1 = x; }
    else if (h < 240.0) { r1 = 0; g1 = x; b1 = c; }
    else if (h < 300.0) { r1 = x; g1 = 0; b1 = c; }
    else { r1 = c; g1 = 0; b1 = x; }

    r = static_cast<float>(r1 + m);
    g = static_cast<float>(g1 + m);
    b = static_cast<float>(b1 + m);
}

std::vector<ProjectedPoint> Projection::getOrComputeEmbedding(
    const std::vector<std::vector<double>>& features,
    const std::vector<std::string>& labels,
    const std::string& cachePath,
    double perplexity,
    double theta,
    int maxIter)
{
    struct RawPoint {
        int id;
        std::string label;
        double x;
        double y;
    };
    std::vector<RawPoint> rawPoints;

    bool loadFromCache = false;
    std::ifstream testFile(cachePath);
    if (testFile.is_open())
    {
        std::cout << "\nFound existing t-SNE cache: '" << cachePath << "'.\n";
        std::cout << "Load cached 2D coordinates? (y/n, default 'y'): ";
        std::string choice;
        std::cin >> choice;
        if (choice.empty() || choice == "y" || choice == "Y")
        {
            loadFromCache = true;
        }
        testFile.close();
    }

    if (loadFromCache)
    {
        std::ifstream file(cachePath);
        std::string line;
        std::getline(file, line); // header: ID;Class;X;Y

        while (std::getline(file, line))
        {
            if (line.empty()) continue;
            std::stringstream ss(line);
            std::string idStr, labelStr, xStr, yStr;

            if (std::getline(ss, idStr, ';') &&
                std::getline(ss, labelStr, ';') &&
                std::getline(ss, xStr, ';') &&
                std::getline(ss, yStr, ';'))
            {
                RawPoint rp;
                rp.id = std::stoi(idStr);
                rp.label = labelStr;
                rp.x = std::stod(xStr);
                rp.y = std::stod(yStr);
                rawPoints.push_back(rp);
            }
        }
        std::cout << "Loaded " << rawPoints.size() << " points from cache.\n";
    }
    else
    {
        int N = static_cast<int>(features.size());
        if (N == 0) {
            std::cerr << "Error: No features found to project.\n";
            return {};
        }
        int D = static_cast<int>(features[0].size());

        std::cout << "\nPreparing data for Barnes-Hut t-SNE (" << N << " points, " << D << " dimensions)...\n";
        std::cout << "Parameters: Perplexity = " << perplexity << ", Theta = " << theta << ", Max Iter = " << maxIter << "\n";

        double* X = new double[N * D];
        for (int i = 0; i < N; ++i)
        {
            for (int d = 0; d < D; ++d)
            {
                X[i * D + d] = features[i][d];
            }
        }

        double* Y = new double[N * 2];

        TSNE tsne;
        std::cout << "Running t-SNE dimensionality reduction...\n";
        tsne.run(X, N, D, Y, 2, perplexity, theta, 42, false, maxIter);
        std::cout << "t-SNE computation completed!\n";

        std::ofstream out(cachePath);
        if (out.is_open())
        {
            out << "ID;Class;X;Y\n";
            for (int i = 0; i < N; ++i)
            {
                std::string label = (i < (int)labels.size()) ? labels[i] : "Unknown";
                out << i << ";" << label << ";" << std::fixed << std::setprecision(6)
                    << Y[i * 2 + 0] << ";" << Y[i * 2 + 1] << "\n";

                RawPoint rp;
                rp.id = i;
                rp.label = label;
                rp.x = Y[i * 2 + 0];
                rp.y = Y[i * 2 + 1];
                rawPoints.push_back(rp);
            }
            out.close();
            std::cout << "Saved 2D coordinates to '" << cachePath << "'.\n";
        }

        delete[] X;
        delete[] Y;
    }

    if (rawPoints.empty()) {
        std::cerr << "Error: No 2D points available.\n";
        return {};
    }

    // 1. Find min and max for normalization to [-0.85, 0.85]
    double minX = rawPoints[0].x, maxX = rawPoints[0].x;
    double minY = rawPoints[0].y, maxY = rawPoints[0].y;
    for (const auto& pt : rawPoints)
    {
        if (pt.x < minX) minX = pt.x;
        if (pt.x > maxX) maxX = pt.x;
        if (pt.y < minY) minY = pt.y;
        if (pt.y > maxY) maxY = pt.y;
    }

    double rangeX = (maxX - minX > 1e-7) ? (maxX - minX) : 1.0;
    double rangeY = (maxY - minY > 1e-7) ? (maxY - minY) : 1.0;

    // 2. Collect unique classes and map each to a distinct HSV color
    std::map<std::string, int> classMap;
    for (const auto& pt : rawPoints)
    {
        if (classMap.find(pt.label) == classMap.end())
        {
            int newIdx = static_cast<int>(classMap.size());
            classMap[pt.label] = newIdx;
        }
    }

    int numClasses = std::max(1, static_cast<int>(classMap.size()));
    std::map<std::string, std::array<float, 3>> classColors;
    for (const auto& pair : classMap)
    {
        double hue = (360.0 * pair.second) / numClasses;
        float r = 1.0f, g = 1.0f, b = 1.0f;
        hsvToRgb(hue, 0.85, 0.95, r, g, b);
        classColors[pair.first] = { r, g, b };
    }

    // 3. Assemble normalized ProjectedPoints
    std::vector<ProjectedPoint> results;
    results.reserve(rawPoints.size());
    for (const auto& rp : rawPoints)
    {
        ProjectedPoint pp;
        pp.id = rp.id;
        pp.label = rp.label;
        pp.x = static_cast<float>(-0.85 + 1.70 * ((rp.x - minX) / rangeX));
        pp.y = static_cast<float>(-0.85 + 1.70 * ((rp.y - minY) / rangeY));

        auto col = classColors[rp.label];
        pp.r = col[0];
        pp.g = col[1];
        pp.b = col[2];

        results.push_back(pp);
    }

    return results;
}

static const char* scatterVertexShaderSource =
"#version 330 core\n"
"layout (location = 0) in vec2 aPos;\n"
"layout (location = 1) in vec3 aColor;\n"
"out vec3 vColor;\n"
"uniform float uPointSize;\n"
"void main()\n"
"{\n"
"   gl_Position = vec4(aPos, 0.0, 1.0);\n"
"   gl_PointSize = uPointSize;\n"
"   vColor = aColor;\n"
"}\n";

static const char* scatterFragmentShaderSource =
"#version 330 core\n"
"in vec3 vColor;\n"
"out vec4 FragColor;\n"
"uniform bool uIsHighlight;\n"
"void main()\n"
"{\n"
"   vec2 coord = gl_PointCoord - vec2(0.5);\n"
"   float dist = length(coord);\n"
"   if (dist > 0.5) {\n"
"       discard;\n"
"   }\n"
"   if (uIsHighlight) {\n"
"       FragColor = vec4(1.0, 1.0, 0.0, 1.0);\n" // Yellow highlight ring
"   } else {\n"
"       FragColor = vec4(vColor, 1.0);\n"
"   }\n"
"}\n";

void Projection::runScatterplot(
    const std::vector<ProjectedPoint>& points,
    const std::vector<std::string>& paths)
{
    if (points.empty())
    {
        std::cerr << "No points to render in scatterplot.\n";
        return;
    }

    glfwInit();
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

    GLFWwindow* window = glfwCreateWindow(1024, 768, "CBSR 2D t-SNE Projection | Hover to brush points", NULL, NULL);
    if (!window)
    {
        std::cerr << "Failed to create GLFW window.\n";
        glfwTerminate();
        return;
    }

    glfwMakeContextCurrent(window);

    if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress))
    {
        std::cerr << "Failed to initialize GLAD.\n";
        glfwDestroyWindow(window);
        glfwTerminate();
        return;
    }

    glViewport(0, 0, 1024, 768);
    glEnable(GL_PROGRAM_POINT_SIZE);

    // Build shader program
    GLuint vertexShader = glCreateShader(GL_VERTEX_SHADER);
    glShaderSource(vertexShader, 1, &scatterVertexShaderSource, NULL);
    glCompileShader(vertexShader);

    GLuint fragmentShader = glCreateShader(GL_FRAGMENT_SHADER);
    glShaderSource(fragmentShader, 1, &scatterFragmentShaderSource, NULL);
    glCompileShader(fragmentShader);

    GLuint shaderProgram = glCreateProgram();
    glAttachShader(shaderProgram, vertexShader);
    glAttachShader(shaderProgram, fragmentShader);
    glLinkProgram(shaderProgram);

    glDeleteShader(vertexShader);
    glDeleteShader(fragmentShader);

    GLint pointSizeLoc = glGetUniformLocation(shaderProgram, "uPointSize");
    GLint highlightLoc = glGetUniformLocation(shaderProgram, "uIsHighlight");

    // Vertex data: [x, y, r, g, b]
    std::vector<float> vertexData;
    vertexData.reserve(points.size() * 5);
    for (const auto& p : points)
    {
        vertexData.push_back(p.x);
        vertexData.push_back(p.y);
        vertexData.push_back(p.r);
        vertexData.push_back(p.g);
        vertexData.push_back(p.b);
    }

    GLuint VAO, VBO;
    glGenVertexArrays(1, &VAO);
    glGenBuffers(1, &VBO);

    glBindVertexArray(VAO);
    glBindBuffer(GL_ARRAY_BUFFER, VBO);
    glBufferData(GL_ARRAY_BUFFER, vertexData.size() * sizeof(float), vertexData.data(), GL_STATIC_DRAW);

    // Position attribute
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 5 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);

    // Color attribute
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 5 * sizeof(float), (void*)(2 * sizeof(float)));
    glEnableVertexAttribArray(1);

    std::cout << "\n=======================================================\n";
    std::cout << "  2D t-SNE Scatterplot Active\n";
    std::cout << "  - Hover over any point to brush and inspect details.\n";
    std::cout << "  - Close the scatterplot window to return to menu.\n";
    std::cout << "=======================================================\n";

    int lastHovered = -2;

    while (!glfwWindowShouldClose(window))
    {
        int winW, winH;
        glfwGetFramebufferSize(window, &winW, &winH);
        glViewport(0, 0, winW, winH);

        // Brushing: find closest point to mouse cursor
        double mouseX, mouseY;
        glfwGetCursorPos(window, &mouseX, &mouseY);

        int hoveredIdx = -1;
        if (winW > 0 && winH > 0)
        {
            float ndcX = static_cast<float>((2.0 * mouseX / winW) - 1.0);
            float ndcY = static_cast<float>(1.0 - (2.0 * mouseY / winH));

            float minSqDist = 1e9f;
            float thresholdSq = 0.035f * 0.035f; // Threshold in NDC

            for (size_t i = 0; i < points.size(); ++i)
            {
                float dx = points[i].x - ndcX;
                float dy = points[i].y - ndcY;
                float d2 = dx * dx + dy * dy;
                if (d2 < minSqDist)
                {
                    minSqDist = d2;
                    if (d2 <= thresholdSq)
                    {
                        hoveredIdx = static_cast<int>(i);
                    }
                }
            }
        }

        // Update title if hover target changed
        if (hoveredIdx != lastHovered)
        {
            lastHovered = hoveredIdx;
            if (hoveredIdx >= 0)
            {
                int pid = points[hoveredIdx].id;
                std::string fileStr = (pid < (int)paths.size()) ? paths[pid] : ("ID " + std::to_string(pid));
                std::string title = "CBSR t-SNE | ID: " + std::to_string(pid) +
                                    " | Class: " + points[hoveredIdx].label +
                                    " | File: " + fileStr;
                glfwSetWindowTitle(window, title.c_str());
            }
            else
            {
                glfwSetWindowTitle(window, "CBSR 2D t-SNE Projection | Hover to brush points");
            }
        }

        // Dark grey background for high contrast
        glClearColor(0.12f, 0.12f, 0.14f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT);

        glUseProgram(shaderProgram);
        glBindVertexArray(VAO);

        // 1. Draw regular points
        glUniform1i(highlightLoc, GL_FALSE);
        glUniform1f(pointSizeLoc, 8.0f);
        glDrawArrays(GL_POINTS, 0, static_cast<GLsizei>(points.size()));

        // 2. Draw highlighted hovered point (larger and bright)
        if (hoveredIdx >= 0)
        {
            glUniform1i(highlightLoc, GL_TRUE);
            glUniform1f(pointSizeLoc, 20.0f);
            glDrawArrays(GL_POINTS, hoveredIdx, 1);

            // Inner core for contrast
            glUniform1i(highlightLoc, GL_FALSE);
            glUniform1f(pointSizeLoc, 10.0f);
            glDrawArrays(GL_POINTS, hoveredIdx, 1);
        }

        glfwSwapBuffers(window);
        glfwPollEvents();
    }

    glDeleteVertexArrays(1, &VAO);
    glDeleteBuffers(1, &VBO);
    glDeleteProgram(shaderProgram);

    glfwDestroyWindow(window);
    glfwTerminate();
}
#pragma once

#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl3.h"

#include <vector>
#include <string>
#include <map>
#include <unordered_map>
#include <algorithm>
#include <cmath>
#include <fstream>
#include <sstream>
#include <iostream>
#include <iomanip>
#include <chrono>

// ============================================================================
// UI & PRESENTATION SUITE FOR 3D MULTIMEDIA RETRIEVAL
// Designed for fast, interactive live demonstration using precomputed data.
// ============================================================================

namespace DemoUI {

struct ShapeRecord {
    int id = 0;
    std::string path;
    std::string fileName;
    std::string className;

    // Pre-resampling stats (from histograms.csv)
    int origFaces = 0;
    int origVerts = 0;
    int origType = 1; // 0 = quads, 1 = tris, 2 = mix
    float zFaces = 0.0f;
    float zVerts = 0.0f;

    // Post-resampling / pre-normalization stats (from full_features_1.csv)
    int resampledFaces = 0;
    int resampledVerts = 0;
    float preNormBaryDist = 0.0f;
    float preNormMaxScale = 1.0f;

    // Post-normalization stats (from flipandrotate.csv)
    int normFaces = 0;
    int normVerts = 0;
    float normVolume = 0.0f;
    float normArea = 0.0f;
    float normBaryX = 0.0f, normBaryY = 0.0f, normBaryZ = 0.0f;
    float normScaleX = 1.0f, normScaleY = 1.0f, normScaleZ = 1.0f;
    float normSignX = 1.0f, normSignY = 1.0f, normSignZ = 1.0f;
};

struct ClassInfo {
    std::string name;
    int startIdx = 0;
    int count = 0;
    std::vector<int> shapeIds;
    float color[3] = { 0.3f, 0.7f, 1.0f };

    // Precomputed evaluation AUCs
    float aucStd = 0.0f;
    float aucImp = 0.0f;
    float aucAnn = 0.0f;
};

struct QueryMatch {
    int shapeId = 0;
    float totalDist = 0.0f;
    std::array<double, NUM_GROUPS> groupDists{};
    bool sameClass = false;
};

struct EvalCurve {
    std::string name;
    float auc = 0.0f;
    std::vector<std::pair<float, float>> points; // (Sensitivity, Specificity)
    std::map<std::string, std::vector<std::pair<float, float>>> classPoints;
    std::map<std::string, float> classAuc;
};

struct GLMeshEntry {
    GLuint vao = 0;
    GLuint vbo = 0;
    GLsizei vertexCount = 0;
    int numVerts = 0;
    int numFaces = 0;
    Point3 minPt{ -0.5, -0.5, -0.5 };
    Point3 maxPt{ 0.5, 0.5, 0.5 };
    Point3 barycenter{ 0.0, 0.0, 0.0 };
};

struct RenderTargetFBO {
    GLuint fbo = 0;
    GLuint colorTex = 0;
    GLuint depthRbo = 0;
    int width = 0;
    int height = 0;

    void ensureSize(int w, int h) {
        w = std::max(64, w);
        h = std::max(64, h);
        if (fbo != 0 && width == w && height == h) return;

        if (fbo == 0) {
            glGenFramebuffers(1, &fbo);
            glGenTextures(1, &colorTex);
            glGenRenderbuffers(1, &depthRbo);
        }

        width = w;
        height = h;

        glBindFramebuffer(GL_FRAMEBUFFER, fbo);

        glBindTexture(GL_TEXTURE_2D, colorTex);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, colorTex, 0);

        glBindRenderbuffer(GL_RENDERBUFFER, depthRbo);
        glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH24_STENCIL8, width, height);
        glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_STENCIL_ATTACHMENT, GL_RENDERBUFFER, depthRbo);

        glBindFramebuffer(GL_FRAMEBUFFER, 0);
    }

    void destroy() {
        if (colorTex) glDeleteTextures(1, &colorTex);
        if (depthRbo) glDeleteRenderbuffers(1, &depthRbo);
        if (fbo) glDeleteFramebuffers(1, &fbo);
        colorTex = depthRbo = fbo = 0;
    }
};

static const char* lineVertexShaderSrc =
    "#version 330 core\n"
    "layout (location = 0) in vec3 aPos;\n"
    "layout (location = 1) in vec3 aCol;\n"
    "uniform mat4 uMVP;\n"
    "out vec3 vCol;\n"
    "void main() {\n"
    "   vCol = aCol;\n"
    "   gl_Position = uMVP * vec4(aPos, 1.0);\n"
    "}\n";

static const char* lineFragmentShaderSrc =
    "#version 330 core\n"
    "in vec3 vCol;\n"
    "out vec4 FragColor;\n"
    "void main() {\n"
    "   FragColor = vec4(vCol, 1.0);\n"
    "}\n";

inline float safeStof(const std::string& s, float defVal = 0.0f) {
    if (s.empty()) return defVal;
    try {
        std::string clean = s;
        for (char& c : clean) if (c == ',') c = '.';
        return std::stof(clean);
    } catch (...) {
        return defVal;
    }
}

class MultimediaDemoApp {
public:
    // Precomputed Database
    std::vector<ShapeRecord> shapes;
    std::vector<ClassInfo> classes;
    std::map<std::string, int> classToIndex;

    FeatureData featRaw;
    FeatureData featStand;
    FeatureData featNorm;
    FeatureWeights defaultWeights{};
    FeatureWeights activeWeights{};
    StandardizationData stdData;
    NormalizationData normData;

    // Precomputed distributions for Tab 2 (Database Preprocessing Verification)
    std::vector<float> histOrigVertsBins;
    std::vector<float> histResampledVertsBins;
    std::vector<float> histPreBaryBins;
    std::vector<float> histPostBaryBins;
    std::vector<float> histPreScaleBins;
    std::vector<float> histPostScaleBins;
    float avgOrigVerts = 0.0f, avgOrigFaces = 0.0f;
    float avgResampledVerts = 0.0f, avgResampledFaces = 0.0f;
    std::vector<int> outlierHighVerts;
    std::vector<int> outlierLowVerts;

    // Precomputed t-SNE Embedding (Mode 7)
    std::vector<ProjectedPoint> tsnePoints;

    // Precomputed Evaluations (Mode 6)
    EvalCurve evalStd;
    EvalCurve evalImp;
    EvalCurve evalAnn;

    // Fast ANN kd-tree
    ANNpointArray annDataPts = nullptr;
    ANNkd_tree* annKdTree = nullptr;
    int annN = 0;
    int annD = 0;
    bool annUseStand = true;

    // Active UI State
    int activeTab = 0;
    int requestedTab = -1;
    int selectedClassIdx = 0;
    int selectedShapeId = 0;
    int compareShapeId = 1;
    char classFilterBuf[64] = "";

    // Query settings
    int searchMethod = 0; // 0 = Linear (Weighted EMD+L2), 1 = ANN kd-Tree
    int annSearchType = 0; // 0 = K-NN, 1 = Radius
    int featureSetChoice = 0; // 0 = Standardized, 1 = Min-Max Normalized
    int queryK = 8;
    float queryRadius = 0.25f;
    float annEpsilon = 0.0f;
    bool autoQueryOnSelect = true;
    double lastQueryTimeMs = 0.0;
    std::vector<QueryMatch> queryResults;

    // 3D Viewport & Interactive Lab Mesh
    GLuint meshShaderProg = 0;
    GLuint lineShaderProg = 0;
    GLuint overlayVao = 0, overlayVbo = 0;

    RenderTargetFBO mainFbo;
    static constexpr int MAX_THUMB_FBOS = 12;
    RenderTargetFBO thumbFbos[MAX_THUMB_FBOS];
    std::unordered_map<int, GLMeshEntry> normMeshCache;

    // Interactive Lab Mesh (for Tab 2 step-by-step normalization/resampling)
    Mesh labMesh;
    GLMeshEntry labMeshGL;
    int labLoadedShapeId = -1;
    bool labIsNormalized = true;
    std::string labStatusText = "Normalized (Translated + PCA Aligned + Flipped + Scaled)";

    // Camera state for Main 3D Viewport
    float camDist = 1.85f;
    float camYaw = 0.45f;
    float camPitch = 0.25f;
    glm::vec3 camTarget{ 0.0f, 0.0f, 0.0f };
    bool renderFaces = true;
    bool renderEdges = false;
    bool renderVertices = false;
    bool renderAxesAndBox = true;
    bool autoRotateTurntable = true;
    float turntableSpeed = 0.6f;

    // Tab 3 (Histograms) state
    bool showQueryHist = true;
    bool showCompareHist = true;
    bool showClassMeanHist = true;
    bool showClassSpaghetti = false;

    // Tab 4 (t-SNE) state
    float tsneZoom = 1.0f;
    ImVec2 tsnePan{ 0.0f, 0.0f };
    float tsnePointRadius = 4.5f;
    int tsneFilterMode = 0; // 0 = All classes, 1 = Selected Class Only, 2 = Query & Top-K Neighbors
    int tsneHoveredShapeId = -1;

    // Sidebar Class Browser state
    int classSortMode = 0;          // 0 = Alphabetical (Name), 1 = AUC
    bool classSortAscending = true; // For Name: true=A-Z, false=Z-A; For AUC: false=High-Low, true=Low-High

    static ImVec4 getAucColor(float auc) {
        if (auc >= 0.90f) {
            return ImVec4(0.25f, 0.95f, 0.40f, 1.0f); // Bright Emerald Green (Excellent >= 0.90)
        } else if (auc >= 0.80f) {
            return ImVec4(0.55f, 0.92f, 0.35f, 1.0f); // Lime Green (Good >= 0.80)
        } else if (auc >= 0.70f) {
            return ImVec4(0.98f, 0.85f, 0.25f, 1.0f); // Bright Yellow / Amber (Moderate >= 0.70)
        } else if (auc >= 0.60f) {
            return ImVec4(1.00f, 0.55f, 0.25f, 1.0f); // Orange (Fair >= 0.60)
        } else {
            return ImVec4(0.95f, 0.35f, 0.35f, 1.0f); // Coral Red (Poor < 0.60)
        }
    }

    // Tab 5 (Evaluation) state
    bool showEvalStd = true;
    bool showEvalImp = true;
    bool showEvalAnn = true;
    bool showEvalSelectedClass = true;
    bool plotStandardROC = false; // default = false: Specificity (TNR, CSV Plot), true = 1 - Specificity

    // Startup timing
    double startupLoadTimeMs = 0.0;

    void init(const std::vector<std::string>& pathList, const std::vector<int>& objCounts) {
        auto t0 = std::chrono::high_resolution_clock::now();

        initShaders();
        loadDatabaseAndFeatures(pathList, objCounts);
        loadPreprocessingStats();
        loadTsneEmbedding();
        loadEvaluations();
        rebuildANNTree(true);

        // Select a great default demo shape (e.g. Bicycle or Jet)
        int defaultClass = 0;
        if (classToIndex.count("Bicycle")) defaultClass = classToIndex["Bicycle"];
        else if (classToIndex.count("Jet")) defaultClass = classToIndex["Jet"];
        selectedClassIdx = defaultClass;
        if (!classes.empty() && !classes[selectedClassIdx].shapeIds.empty()) {
            selectedShapeId = classes[selectedClassIdx].shapeIds[0];
            loadLabMesh(selectedShapeId, true);
            runQuery();
        }

        auto t1 = std::chrono::high_resolution_clock::now();
        startupLoadTimeMs = std::chrono::duration<double, std::milli>(t1 - t0).count();
        std::cout << "[DemoUI] Precomputed database loaded in " << startupLoadTimeMs << " ms.\n";
    }

    void cleanup() {
        mainFbo.destroy();
        for (int i = 0; i < MAX_THUMB_FBOS; ++i) thumbFbos[i].destroy();
        for (auto& kv : normMeshCache) {
            if (kv.second.vao) glDeleteVertexArrays(1, &kv.second.vao);
            if (kv.second.vbo) glDeleteBuffers(1, &kv.second.vbo);
        }
        normMeshCache.clear();
        if (labMeshGL.vao) glDeleteVertexArrays(1, &labMeshGL.vao);
        if (labMeshGL.vbo) glDeleteBuffers(1, &labMeshGL.vbo);
        if (overlayVao) glDeleteVertexArrays(1, &overlayVao);
        if (overlayVbo) glDeleteBuffers(1, &overlayVbo);
        if (meshShaderProg) glDeleteProgram(meshShaderProg);
        if (lineShaderProg) glDeleteProgram(lineShaderProg);
        if (annKdTree) { delete annKdTree; annKdTree = nullptr; }
        if (annDataPts) { annDeallocPts(annDataPts); annDataPts = nullptr; }
    }

private:
    static void hsvToRgb(double h, double s, double v, float& r, float& g, float& b) {
        if (s <= 0.0) { r = g = b = static_cast<float>(v); return; }
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

    void initShaders() {
        unsigned int vs = glCreateShader(GL_VERTEX_SHADER);
        glShaderSource(vs, 1, &vertexShaderSource, NULL);
        glCompileShader(vs);

        unsigned int fs = glCreateShader(GL_FRAGMENT_SHADER);
        glShaderSource(fs, 1, &fragmentShaderSource, NULL);
        glCompileShader(fs);

        meshShaderProg = glCreateProgram();
        glAttachShader(meshShaderProg, vs);
        glAttachShader(meshShaderProg, fs);
        glLinkProgram(meshShaderProg);
        glDeleteShader(vs);
        glDeleteShader(fs);

        unsigned int lvs = glCreateShader(GL_VERTEX_SHADER);
        glShaderSource(lvs, 1, &lineVertexShaderSrc, NULL);
        glCompileShader(lvs);

        unsigned int lfs = glCreateShader(GL_FRAGMENT_SHADER);
        glShaderSource(lfs, 1, &lineFragmentShaderSrc, NULL);
        glCompileShader(lfs);

        lineShaderProg = glCreateProgram();
        glAttachShader(lineShaderProg, lvs);
        glAttachShader(lineShaderProg, lfs);
        glLinkProgram(lineShaderProg);
        glDeleteShader(lvs);
        glDeleteShader(lfs);

        glGenVertexArrays(1, &overlayVao);
        glGenBuffers(1, &overlayVbo);
    }

    void loadDatabaseAndFeatures(const std::vector<std::string>& pathList, const std::vector<int>& objCounts) {
        featRaw = readFeatures("raw_features.csv");
        featStand = readFeatures(fullFeaturesStandFilePath);
        featNorm = readFeatures(fullFeaturesNormFilePath);
        stdData = readStandardizationData(standardizationDataFilePath.string());
        normData = readNormalizationData(normalizationDataFilePath.string());

        defaultWeights = readWeights();
        double wSum = 0.0;
        for (double w : defaultWeights) wSum += w;
        if (wSum < 1e-9 && !featStand.features.empty()) {
            defaultWeights = computeWeightsFromDatabase(featStand.features);
        }
        normalizeWeights(defaultWeights);
        activeWeights = defaultWeights;

        int N = static_cast<int>(pathList.size());
        shapes.resize(N);
        for (int i = 0; i < N; ++i) {
            shapes[i].id = i;
            shapes[i].path = pathList[i];
            fs::path p(pathList[i]);
            shapes[i].fileName = p.filename().string();
            shapes[i].className = p.parent_path().filename().string();
            if (shapes[i].className.empty() && i < (int)featStand.labels.size()) {
                shapes[i].className = featStand.labels[i];
            }

            if (classToIndex.find(shapes[i].className) == classToIndex.end()) {
                int cIdx = static_cast<int>(classes.size());
                classToIndex[shapes[i].className] = cIdx;
                ClassInfo ci;
                ci.name = shapes[i].className;
                ci.startIdx = i;
                ci.count = 0;
                classes.push_back(ci);
            }
            int cIdx = classToIndex[shapes[i].className];
            classes[cIdx].count++;
            classes[cIdx].shapeIds.push_back(i);
        }

        int numC = std::max(1, static_cast<int>(classes.size()));
        for (int c = 0; c < numC; ++c) {
            double hue = (360.0 * c) / numC;
            hsvToRgb(hue, 0.80, 0.95, classes[c].color[0], classes[c].color[1], classes[c].color[2]);
        }
    }

    static std::vector<float> makeHistogramBins(const std::vector<float>& vals, int bins, float minV, float maxV) {
        std::vector<float> out(bins, 0.0f);
        if (vals.empty() || maxV <= minV) return out;
        for (float v : vals) {
            int b = static_cast<int>(((v - minV) / (maxV - minV)) * bins);
            if (b < 0) b = 0;
            if (b >= bins) b = bins - 1;
            out[b] += 1.0f;
        }
        return out;
    }

    void loadPreprocessingStats() {
        int N = static_cast<int>(shapes.size());
        std::vector<float> origVertsVec;
        std::vector<float> resampledVertsVec;
        std::vector<float> preBaryVec;
        std::vector<float> postBaryVec;
        std::vector<float> preScaleVec;
        std::vector<float> postScaleVec;

        // 1. Read histograms.csv (Original Faces & Vertices)
        {
            std::ifstream f(histogramsFilePath);
            if (f.is_open()) {
                std::string line;
                std::getline(f, line);
                double sumF = 0.0, sumV = 0.0;
                int cnt = 0;
                while (std::getline(f, line)) {
                    if (line.empty()) continue;
                    std::stringstream ss(line);
                    std::string idS, fS, vS, tS;
                    std::getline(ss, idS, ';');
                    std::getline(ss, fS, ';');
                    std::getline(ss, vS, ';');
                    std::getline(ss, tS, ';');
                    int id = static_cast<int>(safeStof(idS, -1));
                    if (id >= 0 && id < N) {
                        shapes[id].origFaces = static_cast<int>(safeStof(fS));
                        shapes[id].origVerts = static_cast<int>(safeStof(vS));
                        shapes[id].origType = static_cast<int>(safeStof(tS, 1));
                        sumF += shapes[id].origFaces;
                        sumV += shapes[id].origVerts;
                        origVertsVec.push_back(static_cast<float>(shapes[id].origVerts));
                        cnt++;
                    }
                }
                if (cnt > 0) {
                    avgOrigFaces = static_cast<float>(sumF / cnt);
                    avgOrigVerts = static_cast<float>(sumV / cnt);
                    double varF = 0.0, varV = 0.0;
                    for (int i = 0; i < N; ++i) {
                        varF += (shapes[i].origFaces - avgOrigFaces) * (shapes[i].origFaces - avgOrigFaces);
                        varV += (shapes[i].origVerts - avgOrigVerts) * (shapes[i].origVerts - avgOrigVerts);
                    }
                    float sdF = std::sqrt(static_cast<float>(varF / cnt));
                    float sdV = std::sqrt(static_cast<float>(varV / cnt));
                    for (int i = 0; i < N; ++i) {
                        shapes[i].zFaces = (sdF > 1e-6f) ? std::fabs(shapes[i].origFaces - avgOrigFaces) / sdF : 0.0f;
                        shapes[i].zVerts = (sdV > 1e-6f) ? std::fabs(shapes[i].origVerts - avgOrigVerts) / sdV : 0.0f;
                    }
                }
            }
        }

        // Build sorted outlier lists
        std::vector<int> allIds(N);
        for (int i = 0; i < N; ++i) allIds[i] = i;
        std::sort(allIds.begin(), allIds.end(), [&](int a, int b) {
            return shapes[a].zVerts > shapes[b].zVerts;
        });
        for (int i = 0; i < std::min(N, 40); ++i) {
            if (shapes[allIds[i]].zVerts >= 2.0f) outlierHighVerts.push_back(allIds[i]);
        }
        for (int i = N - 1; i >= std::max(0, N - 20); --i) {
            outlierLowVerts.push_back(allIds[i]);
        }

        // 2. Read full_features_1.csv (After Resampling, Before Normalization)
        {
            std::ifstream f("full_features_1.csv");
            if (f.is_open()) {
                std::string line;
                std::getline(f, line);
                while (std::getline(f, line)) {
                    if (line.empty()) continue;
                    std::stringstream ss(line);
                    std::string tok;
                    std::vector<std::string> cols;
                    while (std::getline(ss, tok, ';')) cols.push_back(tok);
                    if (cols.size() >= 12) {
                        int id = static_cast<int>(safeStof(cols[0], -1));
                        if (id >= 0 && id < N) {
                            shapes[id].resampledFaces = static_cast<int>(safeStof(cols[1]));
                            shapes[id].resampledVerts = static_cast<int>(safeStof(cols[2]));
                            float bx = safeStof(cols[6]), by = safeStof(cols[7]), bz = safeStof(cols[8]);
                            float sx = safeStof(cols[9]), sy = safeStof(cols[10]), sz = safeStof(cols[11]);
                            shapes[id].preNormBaryDist = std::sqrt(bx * bx + by * by + bz * bz);
                            shapes[id].preNormMaxScale = std::max({ sx, sy, sz });
                            preBaryVec.push_back(shapes[id].preNormBaryDist);
                            preScaleVec.push_back(shapes[id].preNormMaxScale);
                        }
                    }
                }
            }
        }

        // 3. Read flipandrotate.csv (After Full Normalization)
        {
            std::ifstream f("flipandrotate.csv");
            if (f.is_open()) {
                std::string line;
                std::getline(f, line);
                double sumF = 0.0, sumV = 0.0;
                int cnt = 0;
                while (std::getline(f, line)) {
                    if (line.empty()) continue;
                    std::stringstream ss(line);
                    std::string tok;
                    std::vector<std::string> cols;
                    while (std::getline(ss, tok, ';')) cols.push_back(tok);
                    if (cols.size() >= 12) {
                        int id = static_cast<int>(safeStof(cols[0], -1));
                        if (id >= 0 && id < N) {
                            shapes[id].normFaces = static_cast<int>(safeStof(cols[1]));
                            shapes[id].normVerts = static_cast<int>(safeStof(cols[2]));
                            shapes[id].normVolume = safeStof(cols[4]);
                            shapes[id].normArea = safeStof(cols[5]);
                            shapes[id].normBaryX = safeStof(cols[6]);
                            shapes[id].normBaryY = safeStof(cols[7]);
                            shapes[id].normBaryZ = safeStof(cols[8]);
                            shapes[id].normScaleX = safeStof(cols[9]);
                            shapes[id].normScaleY = safeStof(cols[10]);
                            shapes[id].normScaleZ = safeStof(cols[11]);
                            if (cols.size() >= 24) {
                                shapes[id].normSignX = safeStof(cols[21], 1.0f);
                                shapes[id].normSignY = safeStof(cols[22], 1.0f);
                                shapes[id].normSignZ = safeStof(cols[23], 1.0f);
                            }
                            sumF += shapes[id].normFaces;
                            sumV += shapes[id].normVerts;
                            resampledVertsVec.push_back(static_cast<float>(shapes[id].normVerts));
                            float bDist = std::sqrt(
                                shapes[id].normBaryX * shapes[id].normBaryX +
                                shapes[id].normBaryY * shapes[id].normBaryY +
                                shapes[id].normBaryZ * shapes[id].normBaryZ);
                            postBaryVec.push_back(bDist);
                            postScaleVec.push_back(std::max({ shapes[id].normScaleX, shapes[id].normScaleY, shapes[id].normScaleZ }));
                            cnt++;
                        }
                    }
                }
                if (cnt > 0) {
                    avgResampledFaces = static_cast<float>(sumF / cnt);
                    avgResampledVerts = static_cast<float>(sumV / cnt);
                }
            }
        }

        histOrigVertsBins = makeHistogramBins(origVertsVec, 30, 0.0f, 30000.0f);
        histResampledVertsBins = makeHistogramBins(resampledVertsVec, 30, 0.0f, 15000.0f);
        histPreBaryBins = makeHistogramBins(preBaryVec, 25, 0.0f, 2.0f);
        histPostBaryBins = makeHistogramBins(postBaryVec, 25, 0.0f, 0.01f);
        histPreScaleBins = makeHistogramBins(preScaleVec, 25, 0.0f, 2.0f);
        histPostScaleBins = makeHistogramBins(postScaleVec, 25, 0.0f, 1.2f);
    }

    void loadTsneEmbedding() {
        std::ifstream file("tsne_embedding.csv");
        if (!file.is_open()) return;

        struct RawPt { int id; std::string label; double x, y; };
        std::vector<RawPt> raw;
        std::string line;
        std::getline(file, line);
        while (std::getline(file, line)) {
            if (line.empty()) continue;
            std::stringstream ss(line);
            std::string idS, lS, xS, yS;
            if (std::getline(ss, idS, ';') && std::getline(ss, lS, ';') &&
                std::getline(ss, xS, ';') && std::getline(ss, yS, ';')) {
                RawPt rp;
                rp.id = static_cast<int>(safeStof(idS));
                rp.label = lS;
                rp.x = safeStof(xS);
                rp.y = safeStof(yS);
                raw.push_back(rp);
            }
        }
        if (raw.empty()) return;

        double minX = raw[0].x, maxX = raw[0].x, minY = raw[0].y, maxY = raw[0].y;
        for (const auto& p : raw) {
            minX = std::min(minX, p.x); maxX = std::max(maxX, p.x);
            minY = std::min(minY, p.y); maxY = std::max(maxY, p.y);
        }
        double rX = std::max(1e-7, maxX - minX);
        double rY = std::max(1e-7, maxY - minY);

        tsnePoints.reserve(raw.size());
        for (const auto& rp : raw) {
            ProjectedPoint pp;
            pp.id = rp.id;
            pp.label = rp.label;
            pp.x = static_cast<float>(-0.9 + 1.8 * ((rp.x - minX) / rX));
            pp.y = static_cast<float>(-0.9 + 1.8 * ((rp.y - minY) / rY));
            if (classToIndex.count(rp.label)) {
                int cIdx = classToIndex[rp.label];
                pp.r = classes[cIdx].color[0];
                pp.g = classes[cIdx].color[1];
                pp.b = classes[cIdx].color[2];
            } else {
                pp.r = 0.7f; pp.g = 0.7f; pp.b = 0.7f;
            }
            tsnePoints.push_back(pp);
        }
    }

    static EvalCurve parseEvalCSV(const std::string& globalFile, const std::string& perClassFile, const std::string& title) {
        EvalCurve ec;
        ec.name = title;

        {
            std::ifstream f(globalFile);
            if (f.is_open()) {
                std::string line;
                if (std::getline(f, line)) {
                    std::stringstream ss(line);
                    std::string tok;
                    std::vector<std::string> hdr;
                    while (std::getline(ss, tok, ';')) hdr.push_back(tok);
                    if (hdr.size() >= 5) ec.auc = safeStof(hdr[4]);
                }
                while (std::getline(f, line)) {
                    if (line.empty()) continue;
                    std::stringstream ss(line);
                    std::string idS, sensS, specS;
                    if (std::getline(ss, idS, ';') && std::getline(ss, sensS, ';') && std::getline(ss, specS, ';')) {
                        ec.points.push_back({ safeStof(sensS), safeStof(specS) });
                    }
                }
            }
        }

        {
            std::ifstream f(perClassFile);
            if (f.is_open()) {
                std::string line;
                std::getline(f, line); // Class;Sensitivity;Specificity;AUC
                while (std::getline(f, line)) {
                    if (line.empty()) continue;
                    std::stringstream ss(line);
                    std::string cName, sensS, specS, aucS;
                    std::getline(ss, cName, ';');
                    std::getline(ss, sensS, ';');
                    std::getline(ss, specS, ';');
                    std::getline(ss, aucS, ';');
                    if (cName.empty()) continue;
                    if (!aucS.empty()) {
                        ec.classAuc[cName] = safeStof(aucS);
                    } else if (!sensS.empty() && !specS.empty()) {
                        ec.classPoints[cName].push_back({ safeStof(sensS), safeStof(specS) });
                    }
                }
            }
        }
        return ec;
    }

    void loadEvaluations() {
        evalStd = parseEvalCSV("eval_linear.csv", "eval_linear_per_class.csv", "Standardized Linear Search");
        evalImp = parseEvalCSV("Eval_after_improvements.csv", "Eval_after_improvements_per_class.csv", "After Improvements (Step k=10)");
        evalAnn = parseEvalCSV("eval_ANN.csv", "eval_ANN_per_class.csv", "ANN kd-Tree (Hellinger Embedding)");

        for (auto& ci : classes) {
            if (evalStd.classAuc.count(ci.name)) ci.aucStd = evalStd.classAuc[ci.name];
            if (evalImp.classAuc.count(ci.name)) ci.aucImp = evalImp.classAuc[ci.name];
            if (evalAnn.classAuc.count(ci.name)) ci.aucAnn = evalAnn.classAuc[ci.name];
        }
    }

public:
    void rebuildANNTree(bool useStandFeatures) {
        const auto& db = useStandFeatures ? featStand.features : featNorm.features;
        if (db.empty() || db[0].empty()) return;

        if (annKdTree) { delete annKdTree; annKdTree = nullptr; }
        if (annDataPts) { annDeallocPts(annDataPts); annDataPts = nullptr; }

        annUseStand = useStandFeatures;
        annN = static_cast<int>(db.size());
        annD = static_cast<int>(db[0].size());
        annDataPts = annAllocPts(annN, annD);

        for (int n = 0; n < annN; ++n) {
            std::vector<double> emb = embedFeatureForANN(db[n], activeWeights);
            for (int d = 0; d < annD; ++d) {
                annDataPts[n][d] = emb[d];
            }
        }
        annKdTree = new ANNkd_tree(annDataPts, annN, annD);
    }

    void runQuery() {
        const auto& db = (featureSetChoice == 0) ? featStand : featNorm;
        int N = static_cast<int>(db.features.size());
        if (selectedShapeId < 0 || selectedShapeId >= N) return;

        auto t0 = std::chrono::high_resolution_clock::now();
        queryResults.clear();

        const auto& queryVec = db.features[selectedShapeId];
        const std::string& queryClass = shapes[selectedShapeId].className;

        if (searchMethod == 1) {
            // ANN kd-Tree search
            if (!annKdTree || annUseStand != (featureSetChoice == 0)) {
                rebuildANNTree(featureSetChoice == 0);
            }
            std::vector<double> embQuery = embedFeatureForANN(queryVec, activeWeights);
            ANNpoint qPt = annAllocPt(annD);
            for (int d = 0; d < annD; ++d) qPt[d] = embQuery[d];

            int k = std::clamp(queryK, 1, annN);
            if (annSearchType == 0) {
                ANNidxArray nnIdx = new ANNidx[k];
                ANNdistArray dists = new ANNdist[k];
                annKdTree->annkSearch(qPt, k, nnIdx, dists, annEpsilon);
                for (int i = 0; i < k; ++i) {
                    int id = nnIdx[i];
                    if (id >= 0 && id < N) {
                        QueryMatch qm;
                        qm.shapeId = id;
                        qm.totalDist = std::sqrt(static_cast<float>(dists[i]));
                        qm.groupDists = groupDistances(queryVec, db.features[id]);
                        qm.sameClass = (shapes[id].className == queryClass);
                        queryResults.push_back(qm);
                    }
                }
                delete[] nnIdx;
                delete[] dists;
            } else {
                int maxK = std::min(50, annN);
                ANNidxArray nnIdx = new ANNidx[maxK];
                ANNdistArray dists = new ANNdist[maxK];
                double sqRad = static_cast<double>(queryRadius) * static_cast<double>(queryRadius);
                int found = annKdTree->annkFRSearch(qPt, sqRad, maxK, nnIdx, dists, annEpsilon);
                int toShow = std::min(found, maxK);
                for (int i = 0; i < toShow; ++i) {
                    int id = nnIdx[i];
                    if (id >= 0 && id < N) {
                        QueryMatch qm;
                        qm.shapeId = id;
                        qm.totalDist = std::sqrt(static_cast<float>(dists[i]));
                        qm.groupDists = groupDistances(queryVec, db.features[id]);
                        qm.sameClass = (shapes[id].className == queryClass);
                        queryResults.push_back(qm);
                    }
                }
                delete[] nnIdx;
                delete[] dists;
            }
            annDeallocPt(qPt);
        } else {
            // Linear Search (Weighted EMD + L2)
            std::vector<std::pair<float, int>> distPairs(N);
            #pragma omp parallel for schedule(static)
            for (int i = 0; i < N; ++i) {
                float d = static_cast<float>(weightedDistance(db.features[i], queryVec, activeWeights));
                distPairs[i] = { d, i };
            }
            int k = std::clamp(queryK, 1, N);
            std::partial_sort(distPairs.begin(), distPairs.begin() + k, distPairs.end(),
                [](const auto& a, const auto& b) { return a.first < b.first; });

            for (int i = 0; i < k; ++i) {
                int id = distPairs[i].second;
                QueryMatch qm;
                qm.shapeId = id;
                qm.totalDist = distPairs[i].first;
                qm.groupDists = groupDistances(queryVec, db.features[id]);
                qm.sameClass = (shapes[id].className == queryClass);
                queryResults.push_back(qm);
            }
        }

        auto t1 = std::chrono::high_resolution_clock::now();
        lastQueryTimeMs = std::chrono::duration<double, std::milli>(t1 - t0).count();

        if (queryResults.size() > 1) {
            compareShapeId = queryResults[1].shapeId;
        } else if (!queryResults.empty()) {
            compareShapeId = queryResults[0].shapeId;
        }

        preloadThumbnails();
    }

    static void uploadMeshToGL(Mesh& m, GLMeshEntry& entry) {
        m.updateGLBuffers();
        if (entry.vao == 0) {
            glGenVertexArrays(1, &entry.vao);
            glGenBuffers(1, &entry.vbo);
        }
        glBindVertexArray(entry.vao);
        glBindBuffer(GL_ARRAY_BUFFER, entry.vbo);
        glBufferData(GL_ARRAY_BUFFER, m.trivertsgl.size() * sizeof(float), m.trivertsgl.data(), GL_STATIC_DRAW);
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0);
        glEnableVertexAttribArray(0);
        glBindBuffer(GL_ARRAY_BUFFER, 0);
        glBindVertexArray(0);

        entry.vertexCount = static_cast<GLsizei>(m.trivertsgl.size() / 3);
        entry.numVerts = m.vertsSize();
        entry.numFaces = m.primsSize();
        entry.minPt = m.minPoint;
        entry.maxPt = m.maxPoint;
        entry.barycenter = m.barycenter;
    }

    void preloadThumbnails() {
        std::vector<int> neededIds;
        if (normMeshCache.find(selectedShapeId) == normMeshCache.end()) {
            neededIds.push_back(selectedShapeId);
        }
        int numThumbs = std::min((int)queryResults.size(), MAX_THUMB_FBOS);
        for (int i = 0; i < numThumbs; ++i) {
            int sid = queryResults[i].shapeId;
            if (normMeshCache.find(sid) == normMeshCache.end()) {
                if (std::find(neededIds.begin(), neededIds.end(), sid) == neededIds.end()) {
                    neededIds.push_back(sid);
                }
            }
        }
        if (neededIds.empty()) return;

        // Ensure cache capacity
        while (normMeshCache.size() + neededIds.size() > 512 && !normMeshCache.empty()) {
            auto first = normMeshCache.begin();
            if (first->second.vao) glDeleteVertexArrays(1, &first->second.vao);
            if (first->second.vbo) glDeleteBuffers(1, &first->second.vbo);
            normMeshCache.erase(first);
        }

        std::vector<Mesh> loadedMeshes(neededIds.size());
        #pragma omp parallel for schedule(dynamic)
        for (int i = 0; i < (int)neededIds.size(); ++i) {
            int sid = neededIds[i];
            if (sid >= 0 && sid < (int)shapes.size()) {
                Parser p;
                loadedMeshes[i] = p.parseOFF(shapes[sid].path);
                loadedMeshes[i].setLabel(shapes[sid].className);
                precomputeMesh(loadedMeshes[i]);
                calculateAABB(loadedMeshes[i]);
                normalizeMesh(loadedMeshes[i]);
            }
        }

        // Upload to OpenGL on main thread
        for (int i = 0; i < (int)neededIds.size(); ++i) {
            int sid = neededIds[i];
            GLMeshEntry entry;
            uploadMeshToGL(loadedMeshes[i], entry);
            normMeshCache[sid] = entry;
            if (sid == selectedShapeId && labLoadedShapeId != selectedShapeId) {
                labMesh = loadedMeshes[i];
                labMeshGL = entry;
                labLoadedShapeId = selectedShapeId;
                labIsNormalized = true;
                labStatusText = "Fully Normalized (Centered + PCA Aligned + Flipped + Unit Scaled)";
            }
        }
    }

    const GLMeshEntry& getOrLoadNormMesh(int shapeId) {
        auto it = normMeshCache.find(shapeId);
        if (it != normMeshCache.end()) return it->second;

        // Limit cache size to 512 meshes to keep GPU memory bounded
        if (normMeshCache.size() >= 512) {
            auto first = normMeshCache.begin();
            if (first->second.vao) glDeleteVertexArrays(1, &first->second.vao);
            if (first->second.vbo) glDeleteBuffers(1, &first->second.vbo);
            normMeshCache.erase(first);
        }

        GLMeshEntry entry;
        if (shapeId >= 0 && shapeId < (int)shapes.size()) {
            Parser p;
            Mesh m = p.parseOFF(shapes[shapeId].path);
            precomputeMesh(m);
            normalizeMesh(m);
            uploadMeshToGL(m, entry);
        }
        normMeshCache[shapeId] = entry;
        return normMeshCache[shapeId];
    }

    void loadLabMesh(int shapeId, bool normalizeImmediately) {
        if (shapeId < 0 || shapeId >= (int)shapes.size()) return;
        Parser p;
        labMesh = p.parseOFF(shapes[shapeId].path);
        labMesh.setLabel(shapes[shapeId].className);
        precomputeMesh(labMesh);
        calculateAABB(labMesh);

        if (normalizeImmediately) {
            normalizeMesh(labMesh);
            labIsNormalized = true;
            labStatusText = "Fully Normalized (Centered + PCA Aligned + Flipped + Unit Scaled)";
        } else {
            // Auto-fit camera to raw mesh if it is far from origin or huge
            labIsNormalized = false;
            labStatusText = "Original Raw Mesh (Un-normalized from OFF/OBJ)";
        }
        labLoadedShapeId = shapeId;
        uploadMeshToGL(labMesh, labMeshGL);
    }

    void renderMeshToFBO(
        RenderTargetFBO& target,
        const GLMeshEntry& meshEntry,
        float yaw, float pitch, float dist, glm::vec3 lookTarget,
        bool drawFaces, bool drawEdges, bool drawVerts, bool drawBoxAndAxes,
        const glm::vec3& faceColor = glm::vec3(0.95f, 0.52f, 0.20f),
        const glm::vec4& bgColor = glm::vec4(0.11f, 0.12f, 0.15f, 1.0f))
    {
        glBindFramebuffer(GL_FRAMEBUFFER, target.fbo);
        glViewport(0, 0, target.width, target.height);
        glEnable(GL_DEPTH_TEST);
        glClearColor(bgColor.r, bgColor.g, bgColor.b, bgColor.a);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        if (meshEntry.vao == 0 || meshEntry.vertexCount == 0) {
            glBindFramebuffer(GL_FRAMEBUFFER, 0);
            return;
        }

        glm::vec3 camPos;
        camPos.x = lookTarget.x + dist * std::cos(pitch) * std::sin(yaw);
        camPos.y = lookTarget.y + dist * std::sin(pitch);
        camPos.z = lookTarget.z + dist * std::cos(pitch) * std::cos(yaw);

        glm::mat4 model = glm::mat4(1.0f);
        glm::mat4 view = glm::lookAt(camPos, lookTarget, glm::vec3(0.0f, 1.0f, 0.0f));
        glm::mat4 modelView = view * model;
        float aspect = (target.height > 0) ? (float)target.width / (float)target.height : 1.0f;
        glm::mat4 proj = glm::perspective(glm::radians(42.0f), aspect, 0.02f, 200.0f);
        glm::mat4 mvp = proj * view * model;

        glUseProgram(meshShaderProg);
        glUniformMatrix4fv(glGetUniformLocation(meshShaderProg, "uMVP"), 1, GL_FALSE, glm::value_ptr(mvp));
        glUniformMatrix4fv(glGetUniformLocation(meshShaderProg, "uModelView"), 1, GL_FALSE, glm::value_ptr(modelView));
        GLint useLightLoc = glGetUniformLocation(meshShaderProg, "uUseLighting");
        GLint colorLoc = glGetUniformLocation(meshShaderProg, "uColor");

        glBindVertexArray(meshEntry.vao);

        if (drawFaces) {
            glUniform1i(useLightLoc, 1);
            glUniform4f(colorLoc, faceColor.r, faceColor.g, faceColor.b, 1.0f);
            glDrawArrays(GL_TRIANGLES, 0, meshEntry.vertexCount);
        }

        if (drawEdges) {
            glUniform1i(useLightLoc, 0);
            glUniform4f(colorLoc, 0.12f, 0.14f, 0.18f, 1.0f);
            glLineWidth(1.2f);
            glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
            glDrawArrays(GL_TRIANGLES, 0, meshEntry.vertexCount);
            glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
        }

        if (drawVerts) {
            glUniform1i(useLightLoc, 0);
            glUniform4f(colorLoc, 1.0f, 0.25f, 0.25f, 1.0f);
            glPointSize(3.5f);
            glDrawArrays(GL_POINTS, 0, meshEntry.vertexCount);
        }

        if (drawBoxAndAxes) {
            std::vector<float> lines;
            auto addLine = [&](glm::vec3 a, glm::vec3 b, glm::vec3 col) {
                lines.insert(lines.end(), { a.x, a.y, a.z, col.r, col.g, col.b });
                lines.insert(lines.end(), { b.x, b.y, b.z, col.r, col.g, col.b });
            };

            // Coordinate axes (X = Red, Y = Green, Z = Blue)
            float axisLen = 0.65f;
            addLine({ 0, 0, 0 }, { axisLen, 0, 0 }, { 0.95f, 0.25f, 0.25f });
            addLine({ 0, 0, 0 }, { 0, axisLen, 0 }, { 0.25f, 0.90f, 0.35f });
            addLine({ 0, 0, 0 }, { 0, 0, axisLen }, { 0.30f, 0.55f, 1.00f });

            // AABB Wireframe Box
            glm::vec3 mn((float)meshEntry.minPt.x(), (float)meshEntry.minPt.y(), (float)meshEntry.minPt.z());
            glm::vec3 mx((float)meshEntry.maxPt.x(), (float)meshEntry.maxPt.y(), (float)meshEntry.maxPt.z());
            glm::vec3 bc(0.55f, 0.60f, 0.70f);

            glm::vec3 c000(mn.x, mn.y, mn.z), c100(mx.x, mn.y, mn.z), c010(mn.x, mx.y, mn.z), c110(mx.x, mx.y, mn.z);
            glm::vec3 c001(mn.x, mn.y, mx.z), c101(mx.x, mn.y, mx.z), c011(mn.x, mx.y, mx.z), c111(mx.x, mx.y, mx.z);
            addLine(c000, c100, bc); addLine(c100, c110, bc); addLine(c110, c010, bc); addLine(c010, c000, bc);
            addLine(c001, c101, bc); addLine(c101, c111, bc); addLine(c111, c011, bc); addLine(c011, c001, bc);
            addLine(c000, c001, bc); addLine(c100, c101, bc); addLine(c110, c111, bc); addLine(c010, c011, bc);

            // Barycenter cross marker (Yellow)
            glm::vec3 bary((float)meshEntry.barycenter.x(), (float)meshEntry.barycenter.y(), (float)meshEntry.barycenter.z());
            float s = 0.06f;
            addLine(bary - glm::vec3(s, 0, 0), bary + glm::vec3(s, 0, 0), { 1.0f, 0.9f, 0.1f });
            addLine(bary - glm::vec3(0, s, 0), bary + glm::vec3(0, s, 0), { 1.0f, 0.9f, 0.1f });
            addLine(bary - glm::vec3(0, 0, s), bary + glm::vec3(0, 0, s), { 1.0f, 0.9f, 0.1f });

            glUseProgram(lineShaderProg);
            glUniformMatrix4fv(glGetUniformLocation(lineShaderProg, "uMVP"), 1, GL_FALSE, glm::value_ptr(mvp));
            glBindVertexArray(overlayVao);
            glBindBuffer(GL_ARRAY_BUFFER, overlayVbo);
            glBufferData(GL_ARRAY_BUFFER, lines.size() * sizeof(float), lines.data(), GL_DYNAMIC_DRAW);
            glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), (void*)0);
            glEnableVertexAttribArray(0);
            glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), (void*)(3 * sizeof(float)));
            glEnableVertexAttribArray(1);
            glLineWidth(1.5f);
            glDrawArrays(GL_LINES, 0, static_cast<GLsizei>(lines.size() / 6));
        }

        glBindVertexArray(0);
        glBindFramebuffer(GL_FRAMEBUFFER, 0);
    }

    void selectShape(int shapeId, bool syncLab = true) {
        if (shapeId < 0 || shapeId >= (int)shapes.size()) return;
        selectedShapeId = shapeId;
        if (classToIndex.count(shapes[shapeId].className)) {
            selectedClassIdx = classToIndex[shapes[shapeId].className];
        }
        if (syncLab) {
            loadLabMesh(shapeId, true);
        }
        if (autoQueryOnSelect) {
            runQuery();
        }
    }

    void drawInteractiveViewport(const GLMeshEntry& meshEntry, float heightPx, const char* labelId) {
        ImVec2 avail = ImGui::GetContentRegionAvail();
        int vpW = std::max(200, static_cast<int>(avail.x));
        int vpH = std::max(200, static_cast<int>(heightPx));

        mainFbo.ensureSize(vpW, vpH);
        renderMeshToFBO(mainFbo, meshEntry, camYaw, camPitch, camDist, camTarget,
            renderFaces, renderEdges, renderVertices, renderAxesAndBox);

        ImGui::Image((ImTextureID)(intptr_t)mainFbo.colorTex, ImVec2((float)vpW, (float)vpH), ImVec2(0, 1), ImVec2(1, 0));

        if (ImGui::IsItemHovered()) {
            ImGuiIO& io = ImGui::GetIO();
            if (ImGui::IsMouseDragging(ImGuiMouseButton_Left)) {
                autoRotateTurntable = false;
                camYaw -= io.MouseDelta.x * 0.007f;
                camPitch += io.MouseDelta.y * 0.007f;
                const float maxP = glm::radians(88.0f);
                camPitch = std::clamp(camPitch, -maxP, maxP);
            }
            if (ImGui::IsMouseDragging(ImGuiMouseButton_Right)) {
                glm::vec3 camPos;
                camPos.x = camTarget.x + camDist * std::cos(camPitch) * std::sin(camYaw);
                camPos.y = camTarget.y + camDist * std::sin(camPitch);
                camPos.z = camTarget.z + camDist * std::cos(camPitch) * std::cos(camYaw);
                glm::vec3 fwd = glm::normalize(camTarget - camPos);
                glm::vec3 right = glm::normalize(glm::cross(fwd, glm::vec3(0, 1, 0)));
                glm::vec3 up = glm::normalize(glm::cross(right, fwd));
                float panSpd = 0.002f * camDist;
                camTarget -= right * (io.MouseDelta.x * panSpd);
                camTarget += up * (io.MouseDelta.y * panSpd);
            }
            if (std::fabs(io.MouseWheel) > 1e-4f) {
                camDist *= (io.MouseWheel > 0) ? 0.88f : 1.14f;
                camDist = std::clamp(camDist, 0.1f, 80.0f);
            }
        }

        // Viewport Controls Bar
        ImGui::PushID(labelId);
        ImGui::Checkbox("Faces", &renderFaces); ImGui::SameLine();
        ImGui::Checkbox("Wireframe", &renderEdges); ImGui::SameLine();
        ImGui::Checkbox("Vertices", &renderVertices); ImGui::SameLine();
        ImGui::Checkbox("Axes & AABB", &renderAxesAndBox); ImGui::SameLine();
        ImGui::Checkbox("Turntable", &autoRotateTurntable); ImGui::SameLine();
        if (ImGui::SmallButton("Reset Camera")) {
            camYaw = 0.45f; camPitch = 0.25f;
            Point3 span = meshEntry.maxPt - meshEntry.minPt;
            float maxDim = std::max({ (float)span.x(), (float)span.y(), (float)span.z(), 0.5f });
            camDist = maxDim * 1.85f;
            camTarget = glm::vec3((float)meshEntry.barycenter.x(), (float)meshEntry.barycenter.y(), (float)meshEntry.barycenter.z());
        }
        ImGui::PopID();
    }

    // ========================================================================
    // LEFT SIDEBAR: DEMO PRESETS & DATABASE BROWSER
    // ========================================================================
    void drawSidebar() {
        ImGui::BeginChild("LeftSidebar", ImVec2(325.0f, 0), true);

        ImGui::TextColored(ImVec4(0.35f, 0.85f, 1.0f, 1.0f), "DEMO SHORTCUTS & PRESETS");
        ImGui::TextDisabled("1-Click presentation scenarios:");

        float btnW = (ImGui::GetContentRegionAvail().x - 8.0f) * 0.5f;
        auto presetBtn = [&](const char* label, const char* cName, int offset = 0) {
            if (ImGui::Button(label, ImVec2(btnW, 0))) {
                if (classToIndex.count(cName)) {
                    int cIdx = classToIndex[cName];
                    if (!classes[cIdx].shapeIds.empty()) {
                        int idx = std::min(offset, (int)classes[cIdx].shapeIds.size() - 1);
                        selectShape(classes[cIdx].shapeIds[idx]);
                    }
                }
            }
        };

        presetBtn("Bicycle (100% P@1)", "Bicycle", 0); ImGui::SameLine();
        presetBtn("Glasses (85% P@1)", "Glasses", 1);
        presetBtn("Keyboard (85% P@1)", "ComputerKeyboard", 0); ImGui::SameLine();
        presetBtn("Jet (143 models)", "Jet", 2);
        presetBtn("Humanoid (142)", "Humanoid", 0); ImGui::SameLine();
        presetBtn("Sword (75% P@1)", "Sword", 0);
        presetBtn("FloorLamp (Hard)", "FloorLamp", 0); ImGui::SameLine();
        presetBtn("Guitar", "Guitar", 0);

        if (!outlierHighVerts.empty() && !outlierLowVerts.empty()) {
            if (ImGui::Button("Max Verts Outlier", ImVec2(btnW, 0))) {
                selectShape(outlierHighVerts[0]);
            }
            ImGui::SameLine();
            if (ImGui::Button("Average Shape", ImVec2(btnW, 0))) {
                selectShape(outlierLowVerts[0]);
            }
        }

        ImGui::Separator();
        ImGui::TextColored(ImVec4(0.35f, 0.85f, 1.0f, 1.0f), "DATABASE BROWSER (%d shapes)", (int)shapes.size());
        ImGui::SetNextItemWidth(-1);
        ImGui::InputTextWithHint("##ClassFilter", "Filter classes (e.g. Car, Mug)...", classFilterBuf, sizeof(classFilterBuf));

        // Sorting Controls Bar
        ImGui::Text("Sort:");
        ImGui::SameLine();
        bool isSortName = (classSortMode == 0);
        bool isSortAuc = (classSortMode == 1);

        if (isSortName) ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.20f, 0.45f, 0.75f, 1.0f));
        if (ImGui::SmallButton("Name")) {
            if (classSortMode == 0) {
                classSortAscending = !classSortAscending;
            } else {
                classSortMode = 0;
                classSortAscending = true; // default A-Z
            }
        }
        if (isSortName) ImGui::PopStyleColor();
        if (ImGui::IsItemHovered()) ImGui::SetTooltip("Sort alphabetically by class name (click again to reverse)");

        ImGui::SameLine();
        if (isSortAuc) ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.20f, 0.45f, 0.75f, 1.0f));
        if (ImGui::SmallButton("AUC")) {
            if (classSortMode == 1) {
                classSortAscending = !classSortAscending;
            } else {
                classSortMode = 1;
                classSortAscending = false; // default High-to-Low for AUC
            }
        }
        if (isSortAuc) ImGui::PopStyleColor();
        if (ImGui::IsItemHovered()) ImGui::SetTooltip("Sort by class retrieval AUC score (click again to reverse)");

        ImGui::SameLine();
        const char* dirIcon = "";
        if (classSortMode == 0) {
            dirIcon = classSortAscending ? "A->Z" : "Z->A";
        } else {
            dirIcon = classSortAscending ? "Low->High" : "High->Low";
        }
        if (ImGui::SmallButton(dirIcon)) {
            classSortAscending = !classSortAscending;
        }
        if (ImGui::IsItemHovered()) ImGui::SetTooltip("Click to toggle sorting direction (Ascending / Descending)");

        ImGui::Text("Classes (%d):", (int)classes.size());
        ImGui::BeginChild("ClassList", ImVec2(0, 230), true);
        std::string filterStr = classFilterBuf;
        std::transform(filterStr.begin(), filterStr.end(), filterStr.begin(), ::tolower);

        // Sort indices based on current criteria
        std::vector<int> sortedClassIndices(classes.size());
        for (size_t i = 0; i < classes.size(); ++i) sortedClassIndices[i] = static_cast<int>(i);

        std::sort(sortedClassIndices.begin(), sortedClassIndices.end(), [&](int a, int b) {
            if (classSortMode == 0) {
                int cmp = classes[a].name.compare(classes[b].name);
                return classSortAscending ? (cmp < 0) : (cmp > 0);
            } else {
                if (std::fabs(classes[a].aucStd - classes[b].aucStd) > 1e-4f) {
                    return classSortAscending ? (classes[a].aucStd < classes[b].aucStd)
                                              : (classes[a].aucStd > classes[b].aucStd);
                }
                return classes[a].name < classes[b].name;
            }
        });

        for (int c : sortedClassIndices) {
            if (!filterStr.empty()) {
                std::string lowerName = classes[c].name;
                std::transform(lowerName.begin(), lowerName.end(), lowerName.begin(), ::tolower);
                if (lowerName.find(filterStr) == std::string::npos) continue;
            }
            char itemLabel[128];
            std::snprintf(itemLabel, sizeof(itemLabel), "%s (%d)  [AUC %.2f]",
                classes[c].name.c_str(), classes[c].count, classes[c].aucStd);
            bool isSel = (selectedClassIdx == c);
            ImGui::PushStyleColor(ImGuiCol_Text, getAucColor(classes[c].aucStd));
            if (ImGui::Selectable(itemLabel, isSel)) {
                selectedClassIdx = c;
                if (!classes[c].shapeIds.empty()) {
                    selectShape(classes[c].shapeIds[0]);
                }
            }
            ImGui::PopStyleColor();
        }
        ImGui::EndChild();

        if (selectedClassIdx >= 0 && selectedClassIdx < (int)classes.size()) {
            const auto& ci = classes[selectedClassIdx];
            ImGui::Text("Shapes in '%s' (%d):", ci.name.c_str(), ci.count);
            ImGui::BeginChild("ShapeList", ImVec2(0, 0), true);
            for (int sid : ci.shapeIds) {
                char sLabel[128];
                std::snprintf(sLabel, sizeof(sLabel), "#%d - %s (%d v)",
                    sid, shapes[sid].fileName.c_str(), shapes[sid].origVerts);
                if (ImGui::Selectable(sLabel, selectedShapeId == sid)) {
                    selectShape(sid);
                }
            }
            ImGui::EndChild();
        }

        ImGui::EndChild();
    }

    // ========================================================================
    // TAB 1: 3D QUERY & RETRIEVAL STUDIO (MODE 5)
    // ========================================================================
    void drawTabQueryStudio() {
        if (shapes.empty() || selectedShapeId < 0 || selectedShapeId >= (int)shapes.size()) {
            ImGui::TextColored(ImVec4(1.0f, 0.4f, 0.4f, 1.0f), "No shapes loaded in database. Please check working directory.");
            return;
        }

        const auto& qShape = shapes[selectedShapeId];
        const auto& qMeshGL = getOrLoadNormMesh(selectedShapeId);

        float leftPaneW = 620.0f;
        ImGui::BeginChild("QueryStudioLeft", ImVec2(leftPaneW, 0), false);

        // Left Column: Query 3D Viewport + Search Settings
        ImGui::TextColored(ImVec4(0.4f, 0.9f, 1.0f, 1.0f), "ACTIVE QUERY SHAPE: #%d (%s)", qShape.id, qShape.className.c_str());
        ImGui::TextDisabled("File: %s | Orig: %d verts, %d faces", qShape.path.c_str(), qShape.origVerts, qShape.origFaces);

        drawInteractiveViewport(qMeshGL, 500.0f, "QueryVP");

        ImGui::Separator();
        ImGui::TextColored(ImVec4(0.4f, 0.9f, 1.0f, 1.0f), "SEARCH & MATCHING SETTINGS (MODE 5)");

        bool changed = false;
        ImGui::Text("Search Method:"); ImGui::SameLine();
        if (ImGui::RadioButton("Linear (EMD + L2)", searchMethod == 0)) { searchMethod = 0; changed = true; }
        ImGui::SameLine();
        if (ImGui::RadioButton("ANN kd-Tree (Hellinger)", searchMethod == 1)) { searchMethod = 1; changed = true; }

        ImGui::Text("Features:"); ImGui::SameLine();
        if (ImGui::RadioButton("Standardized (Z-score)", featureSetChoice == 0)) { featureSetChoice = 0; changed = true; }
        ImGui::SameLine();
        if (ImGui::RadioButton("Min-Max Norm", featureSetChoice == 1)) { featureSetChoice = 1; changed = true; }

        if (searchMethod == 1) {
            ImGui::Text("ANN Mode:"); ImGui::SameLine();
            if (ImGui::RadioButton("K-Nearest (k-NN)", annSearchType == 0)) { annSearchType = 0; changed = true; }
            ImGui::SameLine();
            if (ImGui::RadioButton("Radius Search (R)", annSearchType == 1)) { annSearchType = 1; changed = true; }
            if (annSearchType == 1) {
                ImGui::SetNextItemWidth(150.0f);
                changed |= ImGui::SliderFloat("Search Radius R", &queryRadius, 0.05f, 1.0f, "%.3f");
            }
            ImGui::SetNextItemWidth(150.0f);
            changed |= ImGui::SliderFloat("ANN Epsilon (0 = exact)", &annEpsilon, 0.0f, 1.0f, "%.2f");
        }

        if (searchMethod == 0 || annSearchType == 0) {
            ImGui::SetNextItemWidth(150.0f);
            changed |= ImGui::SliderInt("Result Count (K)", &queryK, 1, 20);
        }

        if (ImGui::CollapsingHeader("Feature Group Weights (Interactive)", ImGuiTreeNodeFlags_DefaultOpen)) {
            const char* gNames[NUM_GROUPS] = {
                "G0: 6 Elementary Features (EMD)",
                "G1: A3 (Angles between 3 pts)",
                "G2: D1 (Distance to Barycenter)",
                "G3: D2 (Distance between 2 pts)",
                "G4: D3 (sqrt Triangle Area 3 pts)",
                "G5: D4 (cbrt Tetrahedron Vol 4 pts)"
            };
            bool wChanged = false;
            for (size_t g = 0; g < NUM_GROUPS; ++g) {
                float wVal = static_cast<float>(activeWeights[g]);
                ImGui::SetNextItemWidth(140.0f);
                if (ImGui::SliderFloat(gNames[g], &wVal, 0.0f, 1.0f, "%.3f")) {
                    activeWeights[g] = wVal;
                    wChanged = true;
                }
            }
            if (ImGui::Button("Reset Optimal Weights (weights.csv)")) {
                activeWeights = defaultWeights;
                wChanged = true;
            }
            ImGui::SameLine();
            if (ImGui::Button("Equal Weights")) {
                for (double& w : activeWeights) w = 1.0 / NUM_GROUPS;
                wChanged = true;
            }
            if (wChanged) {
                normalizeWeights(activeWeights);
                if (searchMethod == 1) rebuildANNTree(featureSetChoice == 0);
                changed = true;
            }
        }

        if (changed) runQuery();

        ImGui::EndChild(); // End QueryStudioLeft

        ImGui::SameLine();

        // Right Column: Starts at the very top of the tab, right alongside the 3D Viewport!
        ImGui::BeginChild("QueryStudioRight", ImVec2(0, 0), false);

        int sameClassHits = 0;
        int sameClassExclSelf = 0;
        int denomExclSelf = 0;
        for (size_t i = 0; i < queryResults.size(); ++i) {
            if (queryResults[i].sameClass) sameClassHits++;
            if (queryResults[i].shapeId != selectedShapeId) {
                denomExclSelf++;
                if (queryResults[i].sameClass) sameClassExclSelf++;
            }
        }
        float precIncl = queryResults.empty() ? 0.0f : (100.0f * sameClassHits / queryResults.size());
        float precExcl = (denomExclSelf == 0) ? 0.0f : (100.0f * sameClassExclSelf / denomExclSelf);

        // Top-Right Live Query Performance & Analytics Dashboard Banner (Utilizes top-right space)
        ImGui::BeginChild("QueryAnalyticsBanner", ImVec2(0, 78.0f), true);
        ImGui::Columns(4, "QueryMetricsCols", false);

        // Metric 1: Precision@K
        ImGui::TextDisabled("PRECISION@%d", (int)queryResults.size());
        ImGui::TextColored(precIncl >= 70.0f ? ImVec4(0.3f, 0.95f, 0.45f, 1.0f) : ImVec4(1.0f, 0.75f, 0.3f, 1.0f),
            "%.1f%%  (%d / %d matches)", precIncl, sameClassHits, (int)queryResults.size());
        ImGui::ProgressBar(precIncl / 100.0f, ImVec2(-1, 8.0f), "");

        // Metric 2: Search Latency & Engine
        ImGui::NextColumn();
        ImGui::TextDisabled("SEARCH LATENCY");
        ImGui::TextColored(ImVec4(0.4f, 0.9f, 1.0f, 1.0f), "%.3f ms", lastQueryTimeMs);
        ImGui::TextDisabled("%s", searchMethod == 0 ? "Linear Scan (EMD+L2)" : "ANN kd-Tree (Hellinger)");

        // Metric 3: Active Query Profile
        ImGui::NextColumn();
        ImGui::TextDisabled("QUERY PROFILE");
        ImGui::TextColored(ImVec4(1.0f, 0.85f, 0.35f, 1.0f), "#%d: %s", qShape.id, qShape.className.c_str());
        ImGui::TextDisabled("%d verts | %d faces", qShape.origVerts, qShape.origFaces);

        // Metric 4: Distance Spread
        ImGui::NextColumn();
        ImGui::TextDisabled("DISTANCE SPREAD");
        if (!queryResults.empty()) {
            ImGui::Text("%.4f - %.4f", queryResults.front().totalDist, queryResults.back().totalDist);
        } else {
            ImGui::Text("No results");
        }
        ImGui::TextDisabled("Excl. self P: %.1f%%", precExcl);

        ImGui::Columns(1);
        ImGui::EndChild();
        ImGui::Spacing();

        ImGui::TextColored(ImVec4(0.4f, 1.0f, 0.5f, 1.0f),
            "TOP-%d RETRIEVAL MATCHES | Search Time: %.3f ms | Precision@%d: %.1f%% (excl. self: %.1f%%)",
            (int)queryResults.size(), lastQueryTimeMs, (int)queryResults.size(), precIncl, precExcl);

        // 3D Thumbnail Grid for Top-K Results
        float availW = ImGui::GetContentRegionAvail().x;
        float cardW = 225.0f;
        int cols = std::max(1, static_cast<int>(availW / (cardW + 10.0f)));

        int numThumbs = std::min((int)queryResults.size(), MAX_THUMB_FBOS);
        for (int i = 0; i < numThumbs; ++i) {
            const auto& res = queryResults[i];
            const auto& rShape = shapes[res.shapeId];
            const auto& rMeshGL = getOrLoadNormMesh(res.shapeId);

            ImGui::PushID(i);
            ImGui::BeginChild("Card", ImVec2(cardW, 255), true);

            // Header with rank & match color
            ImVec4 hdrCol = res.sameClass ? ImVec4(0.2f, 0.95f, 0.4f, 1.0f) : ImVec4(1.0f, 0.40f, 0.35f, 1.0f);
            ImGui::TextColored(hdrCol, "#%d | %s", i + 1, rShape.className.c_str());
            ImGui::TextDisabled("ID %d (%s) | d = %.4f", res.shapeId, rShape.fileName.c_str(), res.totalDist);

            // Render 3D thumbnail FBO synced to camera
            thumbFbos[i].ensureSize(205, 145);
            glm::vec3 tint = res.sameClass ? glm::vec3(0.30f, 0.82f, 0.48f) : glm::vec3(0.92f, 0.42f, 0.35f);
            if (res.shapeId == selectedShapeId) tint = glm::vec3(0.35f, 0.75f, 1.0f);
            renderMeshToFBO(thumbFbos[i], rMeshGL, camYaw, camPitch, 1.75f, glm::vec3(0.0f),
                true, false, false, false, tint);

            ImGui::Image((ImTextureID)(intptr_t)thumbFbos[i].colorTex, ImVec2(205, 145), ImVec2(0, 1), ImVec2(1, 0));

            if (ImGui::Button("Compare Hist", ImVec2(98, 0))) {
                compareShapeId = res.shapeId;
                requestedTab = 2; // Jump to Tab 3 (Histograms)
                activeTab = 2;
            }
            ImGui::SameLine();
            if (ImGui::Button("Set as Query", ImVec2(98, 0))) {
                selectShape(res.shapeId);
            }

            ImGui::EndChild();
            ImGui::PopID();

            if ((i + 1) % cols != 0 && i + 1 < numThumbs) {
                ImGui::SameLine();
            }
        }

        ImGui::Spacing();
        ImGui::Separator();
        ImGui::TextColored(ImVec4(0.4f, 0.9f, 1.0f, 1.0f), "DETAILED DISTANCE BREAKDOWN PER FEATURE GROUP");

        if (ImGui::BeginTable("DistTable", 11, ImGuiTableFlags_Borders | ImGuiTableFlags_RowBg | ImGuiTableFlags_SizingStretchProp)) {
            ImGui::TableSetupColumn("Rank", ImGuiTableColumnFlags_WidthFixed, 38.0f);
            ImGui::TableSetupColumn("ID", ImGuiTableColumnFlags_WidthFixed, 42.0f);
            ImGui::TableSetupColumn("Class", ImGuiTableColumnFlags_WidthStretch);
            ImGui::TableSetupColumn("Total d", ImGuiTableColumnFlags_WidthFixed, 68.0f);
            ImGui::TableSetupColumn("Single", ImGuiTableColumnFlags_WidthFixed, 55.0f);
            ImGui::TableSetupColumn("A3", ImGuiTableColumnFlags_WidthFixed, 55.0f);
            ImGui::TableSetupColumn("D1", ImGuiTableColumnFlags_WidthFixed, 55.0f);
            ImGui::TableSetupColumn("D2", ImGuiTableColumnFlags_WidthFixed, 55.0f);
            ImGui::TableSetupColumn("D3", ImGuiTableColumnFlags_WidthFixed, 55.0f);
            ImGui::TableSetupColumn("D4", ImGuiTableColumnFlags_WidthFixed, 55.0f);
            ImGui::TableSetupColumn("Action", ImGuiTableColumnFlags_WidthFixed, 75.0f);
            ImGui::TableHeadersRow();

            for (size_t i = 0; i < queryResults.size(); ++i) {
                const auto& r = queryResults[i];
                ImGui::TableNextRow();
                ImGui::TableSetColumnIndex(0); ImGui::Text("%d", (int)i + 1);
                ImGui::TableSetColumnIndex(1); ImGui::Text("%d", r.shapeId);
                ImGui::TableSetColumnIndex(2);
                ImVec4 cCol = r.sameClass ? ImVec4(0.3f, 0.95f, 0.45f, 1.0f) : ImVec4(1.0f, 0.45f, 0.4f, 1.0f);
                ImGui::TextColored(cCol, "%s (%s)", shapes[r.shapeId].className.c_str(), shapes[r.shapeId].fileName.c_str());
                ImGui::TableSetColumnIndex(3); ImGui::Text("%.4f", r.totalDist);
                for (int g = 0; g < 6; ++g) {
                    ImGui::TableSetColumnIndex(4 + g);
                    ImGui::Text("%.3f", r.groupDists[g]);
                }
                ImGui::TableSetColumnIndex(10);
                ImGui::PushID((int)i + 500);
                if (ImGui::SmallButton("Compare")) {
                    compareShapeId = r.shapeId;
                    requestedTab = 2;
                    activeTab = 2;
                }
                ImGui::PopID();
            }
            ImGui::EndTable();
        }

        ImGui::EndChild(); // End QueryStudioRight
    }

    void drawDetailedBarHistogram(
        const char* id,
        const char* title,
        const char* subtitle,
        const std::vector<float>& bins,
        float minX, float maxX,
        const char* unitStr,
        ImU32 barColor = IM_COL32(245, 180, 40, 230),
        float targetMin = -1.0f, float targetMax = -1.0f,
        float targetLine = -1.0f)
    {
        ImGui::BeginChild(id, ImVec2(0, 160.0f), true);

        // Header line
        ImGui::TextColored(ImVec4(0.4f, 0.9f, 1.0f, 1.0f), "%s", title);
        ImGui::TextDisabled("%s", subtitle);

        ImVec2 canvasPos = ImGui::GetCursorScreenPos();
        ImVec2 canvasSize = ImGui::GetContentRegionAvail();
        canvasSize.y = std::max(60.0f, canvasSize.y - 4.0f);

        ImDrawList* dl = ImGui::GetWindowDrawList();
        dl->AddRectFilled(canvasPos, ImVec2(canvasPos.x + canvasSize.x, canvasPos.y + canvasSize.y), IM_COL32(20, 23, 30, 255), 4.0f);
        dl->AddRect(canvasPos, ImVec2(canvasPos.x + canvasSize.x, canvasPos.y + canvasSize.y), IM_COL32(55, 62, 75, 255), 4.0f);

        float padL = 42.0f; // Left margin for Y-axis numbers
        float padR = 14.0f;
        float padT = 8.0f;
        float padB = 22.0f; // Bottom margin for X-axis numbers under the histogram

        float plotW = std::max(20.0f, canvasSize.x - padL - padR);
        float plotH = std::max(20.0f, canvasSize.y - padT - padB);

        int numBins = static_cast<int>(bins.size());
        if (numBins == 0) {
            ImGui::EndChild();
            return;
        }

        float maxCount = 1.0f;
        float totalCount = 0.0f;
        for (float v : bins) {
            if (v > maxCount) maxCount = v;
            totalCount += v;
        }

        // Draw Y-axis grid lines and labels (Frequency / count)
        for (int g = 0; g <= 2; ++g) {
            float frac = g / 2.0f;
            float gy = canvasPos.y + padT + (1.0f - frac) * plotH;
            dl->AddLine(ImVec2(canvasPos.x + padL, gy), ImVec2(canvasPos.x + padL + plotW, gy), IM_COL32(45, 50, 62, 255));
            int val = static_cast<int>(frac * maxCount);
            char yLbl[16];
            std::snprintf(yLbl, sizeof(yLbl), "%d", val);
            dl->AddText(ImVec2(canvasPos.x + 4.0f, gy - 7.0f), IM_COL32(150, 160, 175, 255), yLbl);
        }

        // Target range highlight (e.g. [5k - 10k] or target line at 1.0)
        if (targetMin >= minX && targetMax <= maxX && targetMax > targetMin) {
            float x0 = canvasPos.x + padL + ((targetMin - minX) / (maxX - minX)) * plotW;
            float x1 = canvasPos.x + padL + ((targetMax - minX) / (maxX - minX)) * plotW;
            dl->AddRectFilled(ImVec2(x0, canvasPos.y + padT), ImVec2(x1, canvasPos.y + padT + plotH),
                IM_COL32(40, 200, 120, 45));
            dl->AddLine(ImVec2(x0, canvasPos.y + padT), ImVec2(x0, canvasPos.y + padT + plotH), IM_COL32(40, 220, 120, 180), 1.2f);
            dl->AddLine(ImVec2(x1, canvasPos.y + padT), ImVec2(x1, canvasPos.y + padT + plotH), IM_COL32(40, 220, 120, 180), 1.2f);
        }
        if (targetLine >= minX && targetLine <= maxX) {
            float xTarget = canvasPos.x + padL + ((targetLine - minX) / (maxX - minX)) * plotW;
            dl->AddLine(ImVec2(xTarget, canvasPos.y + padT), ImVec2(xTarget, canvasPos.y + padT + plotH),
                IM_COL32(50, 210, 255, 220), 2.0f);
        }

        // Draw Bars
        float barW = plotW / numBins;
        ImVec2 mousePos = ImGui::GetIO().MousePos;
        int hoveredBin = -1;

        for (int i = 0; i < numBins; ++i) {
            float barH = (bins[i] / maxCount) * plotH;
            float bx0 = canvasPos.x + padL + i * barW;
            float bx1 = bx0 + barW - 1.0f;
            float by1 = canvasPos.y + padT + plotH;
            float by0 = by1 - barH;

            bool isHovered = (mousePos.x >= bx0 && mousePos.x <= bx1 &&
                              mousePos.y >= canvasPos.y + padT && mousePos.y <= by1);
            if (isHovered) hoveredBin = i;

            ImU32 col = isHovered ? IM_COL32(255, 240, 120, 255) : barColor;
            dl->AddRectFilled(ImVec2(bx0, by0), ImVec2(bx1, by1), col, 1.0f);
        }

        // Draw X-axis tick labels UNDER the histogram
        int numTicks = 5;
        for (int t = 0; t <= numTicks; ++t) {
            float frac = t / static_cast<float>(numTicks);
            float xVal = minX + frac * (maxX - minX);
            float gx = canvasPos.x + padL + frac * plotW;

            // Tick mark
            dl->AddLine(ImVec2(gx, canvasPos.y + padT + plotH), ImVec2(gx, canvasPos.y + padT + plotH + 4.0f), IM_COL32(120, 130, 150, 255));

            char xLbl[24];
            if (maxX >= 1000.0f) {
                if (xVal >= 1000.0f) std::snprintf(xLbl, sizeof(xLbl), "%.0fk", xVal / 1000.0f);
                else std::snprintf(xLbl, sizeof(xLbl), "%.0f", xVal);
            } else if (maxX <= 2.5f) {
                std::snprintf(xLbl, sizeof(xLbl), "%.1f", xVal);
            } else {
                std::snprintf(xLbl, sizeof(xLbl), "%.0f", xVal);
            }
            float textOffset = (t == 0) ? 0.0f : ((t == numTicks) ? 22.0f : 10.0f);
            dl->AddText(ImVec2(gx - textOffset, canvasPos.y + padT + plotH + 5.0f), IM_COL32(165, 175, 190, 255), xLbl);
        }

        // Invisible button for hover detection & tooltip
        ImGui::InvisibleButton(id, canvasSize);
        if (hoveredBin >= 0 && hoveredBin < numBins) {
            float bSpan = (maxX - minX) / numBins;
            float r0 = minX + hoveredBin * bSpan;
            float r1 = r0 + bSpan;
            float count = bins[hoveredBin];
            float pct = (totalCount > 0) ? (count / totalCount) * 100.0f : 0.0f;

            char r0Str[16], r1Str[16];
            if (maxX >= 1000.0f) {
                std::snprintf(r0Str, sizeof(r0Str), "%.0f", r0);
                std::snprintf(r1Str, sizeof(r1Str), "%.0f", r1);
            } else {
                std::snprintf(r0Str, sizeof(r0Str), "%.3f", r0);
                std::snprintf(r1Str, sizeof(r1Str), "%.3f", r1);
            }

            ImGui::SetTooltip("%s\nRange: [%s - %s] %s\nCount: %.0f shapes (%.1f%% of database)",
                title, r0Str, r1Str, unitStr, count, pct);
        }

        ImGui::EndChild();
    }

    // ========================================================================
    // TAB 2: 3D PREPROCESSING, RESAMPLING & NORMALIZATION LAB (MODE 1, 2 & 4)
    // ========================================================================
    void drawTabPreprocessingLab() {
        if (labLoadedShapeId != selectedShapeId) {
            loadLabMesh(selectedShapeId, true);
        }
        const auto& rec = shapes[selectedShapeId];

        float leftPaneW = 620.0f;
        ImGui::BeginChild("PreprocLeft", ImVec2(leftPaneW, 0), false);

        ImGui::TextColored(ImVec4(0.4f, 0.9f, 1.0f, 1.0f),
            "INTERACTIVE 3D NORMALIZATION & RESAMPLING LAB: #%d (%s)", rec.id, rec.className.c_str());
        ImGui::Text("Status: %s", labStatusText.c_str());

        drawInteractiveViewport(labMeshGL, 500.0f, "LabVP");

        // Step-by-Step Normalization Controls
        ImGui::Separator();
        ImGui::TextColored(ImVec4(1.0f, 0.85f, 0.35f, 1.0f), "STEP-BY-STEP 3D NORMALIZATION PIPELINE:");
        float btnW = (ImGui::GetContentRegionAvail().x - 10.0f) * 0.5f;

        if (ImGui::Button("0. Load Raw Original Mesh (Unprocessed)", ImVec2(btnW, 0))) {
            loadLabMesh(selectedShapeId, false);
            Point3 span = labMeshGL.maxPt - labMeshGL.minPt;
            float maxDim = std::max({ (float)span.x(), (float)span.y(), (float)span.z(), 0.5f });
            camDist = maxDim * 1.85f;
            camTarget = glm::vec3((float)labMeshGL.barycenter.x(), (float)labMeshGL.barycenter.y(), (float)labMeshGL.barycenter.z());
        }
        ImGui::SameLine();
        if (ImGui::Button("Full Normalization (Steps 1-4)", ImVec2(btnW, 0))) {
            precomputeMesh(labMesh);
            normalizeMesh(labMesh);
            uploadMeshToGL(labMesh, labMeshGL);
            camTarget = glm::vec3(0.0f);
            camDist = 1.85f;
            labStatusText = "Fully Normalized (Steps 1 to 4)";
        }

        if (ImGui::Button("1. Translate Barycenter -> (0,0,0)", ImVec2(btnW, 0))) {
            precomputeMesh(labMesh);
            translateBarycenter(labMesh);
            uploadMeshToGL(labMesh, labMeshGL);
            camTarget = glm::vec3(0.0f);
            labStatusText = "Step 1 Done: Barycenter translated to (0,0,0)";
        }
        ImGui::SameLine();
        if (ImGui::Button("2. Rotate PCA Principal Axes (Align)", ImVec2(btnW, 0))) {
            precomputeMesh(labMesh);
            alignAxes(labMesh);
            uploadMeshToGL(labMesh, labMeshGL);
            labStatusText = "Step 2 Done: PCA Eigenvectors aligned to X,Y,Z";
        }

        if (ImGui::Button("3. Flip Along Major Moments (Flip)", ImVec2(btnW, 0))) {
            precomputeMesh(labMesh);
            flipShape(labMesh);
            uploadMeshToGL(labMesh, labMeshGL);
            labStatusText = "Step 3 Done: Major moments positively flipped";
        }
        ImGui::SameLine();
        if (ImGui::Button("4. Scale Uniformly to [1x1x1]", ImVec2(btnW, 0))) {
            precomputeMesh(labMesh);
            scaleUniformly(labMesh);
            uploadMeshToGL(labMesh, labMeshGL);
            camDist = 1.85f;
            labStatusText = "Step 4 Done: Uniformly scaled to unit bounding box";
        }

        ImGui::Spacing();
        ImGui::TextColored(ImVec4(0.4f, 1.0f, 0.6f, 1.0f), "INTERACTIVE REMESHING & RESAMPLING DEMO:");
        float remeshW = (ImGui::GetContentRegionAvail().x - 30.0f) * 0.25f;

        if (ImGui::Button("Simple Subdivide (U)", ImVec2(remeshW, 0))) {
            upsampleMeshSimple(labMesh);
            precomputeMesh(labMesh);
            calculateAABB(labMesh);
            uploadMeshToGL(labMesh, labMeshGL);
            labStatusText = "1 Step Simple Subdivision executed";
        }
        ImGui::SameLine();
        if (ImGui::Button("Loop Subdivide (L)", ImVec2(remeshW, 0))) {
            upsampleMeshLoop(labMesh);
            precomputeMesh(labMesh);
            calculateAABB(labMesh);
            uploadMeshToGL(labMesh, labMeshGL);
            labStatusText = "1 Step Loop Subdivision executed";
        }
        ImGui::SameLine();
        if (ImGui::Button("Decimate (K)", ImVec2(remeshW, 0))) {
            downsampleMesh(labMesh);
            precomputeMesh(labMesh);
            calculateAABB(labMesh);
            uploadMeshToGL(labMesh, labMeshGL);
            labStatusText = "1 Step Quadric Edge Decimation executed";
        }
        ImGui::SameLine();
        if (ImGui::Button("Auto 5k-10k (R)", ImVec2(remeshW, 0))) {
            resampleMesh(labMesh);
            precomputeMesh(labMesh);
            calculateAABB(labMesh);
            uploadMeshToGL(labMesh, labMeshGL);
            labStatusText = "Auto-Resampled to target range [5,000 - 10,000] vertices";
        }

        // Live Mesh Telemetry Table
        ImGui::Spacing();
        if (ImGui::BeginTable("MeshStatsTable", 4, ImGuiTableFlags_Borders | ImGuiTableFlags_RowBg)) {
            ImGui::TableSetupColumn("Property");
            ImGui::TableSetupColumn("Original (CSV)");
            ImGui::TableSetupColumn("Normalized (CSV)");
            ImGui::TableSetupColumn("Current 3D Mesh (Live)");
            ImGui::TableHeadersRow();

            Point3 curScale = labMesh.maxPoint - labMesh.minPoint;
            ImGui::TableNextRow();
            ImGui::TableSetColumnIndex(0); ImGui::Text("Vertices");
            ImGui::TableSetColumnIndex(1); ImGui::Text("%d (%.2f sigma)", rec.origVerts, rec.zVerts);
            ImGui::TableSetColumnIndex(2); ImGui::Text("%d", rec.normVerts);
            ImGui::TableSetColumnIndex(3); ImGui::TextColored(ImVec4(0.4f, 1.0f, 0.5f, 1.0f), "%d", labMesh.vertsSize());

            ImGui::TableNextRow();
            ImGui::TableSetColumnIndex(0); ImGui::Text("Faces");
            ImGui::TableSetColumnIndex(1); ImGui::Text("%d (%.2f sigma)", rec.origFaces, rec.zFaces);
            ImGui::TableSetColumnIndex(2); ImGui::Text("%d", rec.normFaces);
            ImGui::TableSetColumnIndex(3); ImGui::TextColored(ImVec4(0.4f, 1.0f, 0.5f, 1.0f), "%d", labMesh.primsSize());

            ImGui::TableNextRow();
            ImGui::TableSetColumnIndex(0); ImGui::Text("Barycenter (X,Y,Z)");
            ImGui::TableSetColumnIndex(1); ImGui::Text("|c| = %.3f", rec.preNormBaryDist);
            ImGui::TableSetColumnIndex(2); ImGui::Text("(%.1e, %.1e, %.1e)", rec.normBaryX, rec.normBaryY, rec.normBaryZ);
            ImGui::TableSetColumnIndex(3); ImGui::Text("(%.3f, %.3f, %.3f)", labMesh.barycenter.x(), labMesh.barycenter.y(), labMesh.barycenter.z());

            ImGui::TableNextRow();
            ImGui::TableSetColumnIndex(0); ImGui::Text("AABB Extents (X,Y,Z)");
            ImGui::TableSetColumnIndex(1); ImGui::Text("Max = %.3f", rec.preNormMaxScale);
            ImGui::TableSetColumnIndex(2); ImGui::Text("(%.2f, %.2f, %.2f)", rec.normScaleX, rec.normScaleY, rec.normScaleZ);
            ImGui::TableSetColumnIndex(3); ImGui::Text("(%.3f, %.3f, %.3f)", curScale.x(), curScale.y(), curScale.z());

            ImGui::TableNextRow();
            ImGui::TableSetColumnIndex(0); ImGui::Text("Surface Area / Volume");
            ImGui::TableSetColumnIndex(1); ImGui::Text("-");
            ImGui::TableSetColumnIndex(2); ImGui::Text("A=%.3f | V=%.4f", rec.normArea, rec.normVolume);
            ImGui::TableSetColumnIndex(3); ImGui::Text("A=%.3f | V=%.4f", labMesh.area, labMesh.volume);

            ImGui::EndTable();
        }

        ImGui::EndChild(); // End PreprocLeft

        ImGui::SameLine();

        // Right Column: Starts at the very top of the tab, right alongside the 3D Viewport!
        ImGui::BeginChild("PreprocRight", ImVec2(0, 0), false);

        // Top KPI Analytics Banner (Utilizes top-right space)
        ImGui::BeginChild("PreprocSummaryBanner", ImVec2(0, 78.0f), true);
        ImGui::Columns(4, "PreprocKpiCols", false);

        // KPI 1: Dataset Coverage
        ImGui::TextDisabled("DATASET COVERAGE");
        ImGui::TextColored(ImVec4(0.4f, 0.9f, 1.0f, 1.0f), "2,483 Shapes");
        ImGui::TextDisabled("69 benchmark classes");

        // KPI 2: Vertex Resampling
        ImGui::NextColumn();
        ImGui::TextDisabled("RESAMPLING BENCHMARK");
        ImGui::TextColored(ImVec4(0.35f, 0.95f, 0.55f, 1.0f), "%.0f -> %.0f avg", avgOrigVerts, avgResampledVerts);
        ImGui::TextDisabled("Target: [5,000 - 10,000]");

        // KPI 3: Barycenter & AABB Normalization
        ImGui::NextColumn();
        ImGui::TextDisabled("NORMALIZATION ACCURACY");
        ImGui::TextColored(ImVec4(0.35f, 0.95f, 0.55f, 1.0f), "Barycenter < 1e-6");
        ImGui::TextDisabled("Unit Cube Max Dim = 1.0");

        // KPI 4: Outliers Detected
        ImGui::NextColumn();
        ImGui::TextDisabled("OUTLIERS (MODE 4)");
        ImGui::TextColored(ImVec4(1.0f, 0.75f, 0.35f, 1.0f), "%d High / %d Average", (int)outlierHighVerts.size(), (int)outlierLowVerts.size());
        ImGui::TextDisabled("Z-score > 3.0 sigma");

        ImGui::Columns(1);
        ImGui::EndChild();
        ImGui::Spacing();

        ImGui::TextColored(ImVec4(0.4f, 0.9f, 1.0f, 1.0f), "DATABASE VERIFICATION (2,483 PRECOMPUTED SHAPES)");
        ImGui::Text("Average vertex count: %.0f (pre-resampling)  -->  %.0f (post-resampling [5k-10k])",
            avgOrigVerts, avgResampledVerts);
        ImGui::Spacing();

        ImGui::Columns(2, "PreprocHistCompareCols", false);

        // Left Column: BEFORE Preprocessing
        ImGui::TextColored(ImVec4(1.0f, 0.75f, 0.35f, 1.0f), "BEFORE PREPROCESSING / RESAMPLING:");
        drawDetailedBarHistogram("HistOrigVerts", "Vertices BEFORE Resampling", "Widely distributed (outliers up to 65k+)",
            histOrigVertsBins, 0.0f, 30000.0f, "verts", IM_COL32(235, 175, 45, 230));
        drawDetailedBarHistogram("HistPreBary", "Barycenter Distance BEFORE Translation", "Distance to (0,0,0) before centering",
            histPreBaryBins, 0.0f, 2.0f, "units", IM_COL32(235, 175, 45, 230));

        // Right Column: AFTER Preprocessing (Utilizes empty top-right space)
        ImGui::NextColumn();
        ImGui::TextColored(ImVec4(0.35f, 0.95f, 0.55f, 1.0f), "AFTER PREPROCESSING / RESAMPLING:");
        drawDetailedBarHistogram("HistResampledVerts", "Vertices AFTER Resampling", "Target range [5,000 - 10,000]",
            histResampledVertsBins, 0.0f, 15000.0f, "verts", IM_COL32(65, 215, 130, 230), 5000.0f, 10000.0f);
        drawDetailedBarHistogram("HistPostScale", "Max AABB Dimension AFTER Scaling", "Scaled to unit cube (target = 1.0)",
            histPostScaleBins, 0.0f, 1.2f, "max dimension", IM_COL32(65, 215, 130, 230), -1.0f, -1.0f, 1.0f);

        ImGui::Columns(1);

        ImGui::Spacing();
        ImGui::Separator();
        ImGui::TextColored(ImVec4(1.0f, 0.75f, 0.3f, 1.0f), "OUTLIER DETECTION (MODE 4 - Z-SCORE DEVIATION FROM MEAN)");
        ImGui::TextDisabled("Click 'View 3D' to inspect an outlier directly in the 3D Lab:");

        if (ImGui::BeginTable("OutlierTable", 6, ImGuiTableFlags_Borders | ImGuiTableFlags_RowBg | ImGuiTableFlags_ScrollY, ImVec2(0, 260))) {
            ImGui::TableSetupColumn("Outlier Category");
            ImGui::TableSetupColumn("ID");
            ImGui::TableSetupColumn("Class & File");
            ImGui::TableSetupColumn("Orig Vertices");
            ImGui::TableSetupColumn("Z-Score (Sigma)");
            ImGui::TableSetupColumn("Action");
            ImGui::TableHeadersRow();

            for (int i = 0; i < std::min(15, (int)outlierHighVerts.size()); ++i) {
                int sid = outlierHighVerts[i];
                ImGui::TableNextRow();
                ImGui::TableSetColumnIndex(0); ImGui::TextColored(ImVec4(1.0f, 0.45f, 0.4f, 1.0f), "Extreme High (>3s)");
                ImGui::TableSetColumnIndex(1); ImGui::Text("#%d", sid);
                ImGui::TableSetColumnIndex(2); ImGui::Text("%s / %s", shapes[sid].className.c_str(), shapes[sid].fileName.c_str());
                ImGui::TableSetColumnIndex(3); ImGui::Text("%d v / %d f", shapes[sid].origVerts, shapes[sid].origFaces);
                ImGui::TableSetColumnIndex(4); ImGui::Text("%.2f sigma", shapes[sid].zVerts);
                ImGui::TableSetColumnIndex(5);
                ImGui::PushID(sid + 1000);
                if (ImGui::SmallButton("View 3D")) selectShape(sid);
                ImGui::PopID();
            }
            for (int i = 0; i < std::min(8, (int)outlierLowVerts.size()); ++i) {
                int sid = outlierLowVerts[i];
                ImGui::TableNextRow();
                ImGui::TableSetColumnIndex(0); ImGui::TextColored(ImVec4(0.4f, 0.95f, 0.5f, 1.0f), "Closest to Mean");
                ImGui::TableSetColumnIndex(1); ImGui::Text("#%d", sid);
                ImGui::TableSetColumnIndex(2); ImGui::Text("%s / %s", shapes[sid].className.c_str(), shapes[sid].fileName.c_str());
                ImGui::TableSetColumnIndex(3); ImGui::Text("%d v / %d f", shapes[sid].origVerts, shapes[sid].origFaces);
                ImGui::TableSetColumnIndex(4); ImGui::Text("%.4f sigma", shapes[sid].zVerts);
                ImGui::TableSetColumnIndex(5);
                ImGui::PushID(sid + 2000);
                if (ImGui::SmallButton("View 3D")) selectShape(sid);
                ImGui::PopID();
            }
            ImGui::EndTable();
        }

        ImGui::EndChild(); // End PreprocRight
    }

    // ========================================================================
    // TAB 3: FEATURE DESCRIPTORS & HISTOGRAMS EXPLORER (MODE 2 & 3)
    // ========================================================================
    void drawCustomLineHistogram(
        const char* title,
        const char* subtitle,
        int histIdx,
        int queryId,
        int compId,
        int classIdx,
        float heightPx)
    {
        ImGui::BeginChild(title, ImVec2(0, heightPx), true);
        ImGui::TextColored(ImVec4(0.4f, 0.9f, 1.0f, 1.0f), "%s", title);
        ImGui::TextDisabled("%s", subtitle);

        ImVec2 canvasPos = ImGui::GetCursorScreenPos();
        ImVec2 canvasSize = ImGui::GetContentRegionAvail();
        canvasSize.y = std::max(60.0f, canvasSize.y - 6.0f);

        ImDrawList* dl = ImGui::GetWindowDrawList();
        dl->AddRectFilled(canvasPos, ImVec2(canvasPos.x + canvasSize.x, canvasPos.y + canvasSize.y), IM_COL32(22, 25, 32, 255), 4.0f);
        dl->AddRect(canvasPos, ImVec2(canvasPos.x + canvasSize.x, canvasPos.y + canvasSize.y), IM_COL32(60, 68, 82, 255), 4.0f);

        int startCol = numSingleFeatures + histIdx * numHistogramBins;
        int bins = numHistogramBins;

        // Compute max bin height for auto-scaling Y axis
        double maxY = 0.05;
        auto scanMax = [&](int sid) {
            if (sid >= 0 && sid < (int)featStand.features.size()) {
                const auto& row = featStand.features[sid];
                for (int b = 0; b < bins && startCol + b < (int)row.size(); ++b) {
                    maxY = std::max(maxY, row[startCol + b]);
                }
            }
        };
        if (showQueryHist) scanMax(queryId);
        if (showCompareHist) scanMax(compId);

        std::vector<double> classMean(bins, 0.0);
        if (classIdx >= 0 && classIdx < (int)classes.size() && !classes[classIdx].shapeIds.empty()) {
            for (int sid : classes[classIdx].shapeIds) {
                if (showClassSpaghetti) scanMax(sid);
                if (sid >= 0 && sid < (int)featStand.features.size()) {
                    const auto& row = featStand.features[sid];
                    for (int b = 0; b < bins && startCol + b < (int)row.size(); ++b) {
                        classMean[b] += row[startCol + b];
                    }
                }
            }
            for (int b = 0; b < bins; ++b) {
                classMean[b] /= classes[classIdx].shapeIds.size();
                if (showClassMeanHist) maxY = std::max(maxY, classMean[b]);
            }
        }
        maxY *= 1.12;

        float padL = 8.0f, padR = 8.0f, padT = 8.0f, padB = 16.0f;
        float plotW = std::max(10.0f, canvasSize.x - padL - padR);
        float plotH = std::max(10.0f, canvasSize.y - padT - padB);

        auto toScreen = [&](int b, double val) -> ImVec2 {
            float x = canvasPos.x + padL + (static_cast<float>(b) / (bins - 1)) * plotW;
            float normY = static_cast<float>(std::clamp(val / maxY, 0.0, 1.0));
            float y = canvasPos.y + padT + (1.0f - normY) * plotH;
            return ImVec2(x, y);
        };

        // Grid lines
        for (int g = 1; g <= 3; ++g) {
            float gy = canvasPos.y + padT + (g / 4.0f) * plotH;
            dl->AddLine(ImVec2(canvasPos.x + padL, gy), ImVec2(canvasPos.x + padL + plotW, gy), IM_COL32(45, 50, 60, 255));
        }

        // 1. Optional spaghetti overlay for all shapes in class
        if (showClassSpaghetti && classIdx >= 0 && classIdx < (int)classes.size()) {
            for (int sid : classes[classIdx].shapeIds) {
                if (sid < 0 || sid >= (int)featStand.features.size()) continue;
                const auto& row = featStand.features[sid];
                for (int b = 0; b + 1 < bins && startCol + b + 1 < (int)row.size(); ++b) {
                    dl->AddLine(toScreen(b, row[startCol + b]), toScreen(b + 1, row[startCol + b + 1]),
                        IM_COL32(100, 220, 130, 38), 1.0f);
                }
            }
        }

        // 2. Class Mean Curve (Green)
        if (showClassMeanHist) {
            for (int b = 0; b + 1 < bins; ++b) {
                dl->AddLine(toScreen(b, classMean[b]), toScreen(b + 1, classMean[b + 1]),
                    IM_COL32(80, 230, 120, 210), 2.0f);
            }
        }

        // 3. Comparison Shape Curve (Orange)
        if (showCompareHist && compId >= 0 && compId < (int)featStand.features.size()) {
            const auto& row = featStand.features[compId];
            for (int b = 0; b + 1 < bins && startCol + b + 1 < (int)row.size(); ++b) {
                dl->AddLine(toScreen(b, row[startCol + b]), toScreen(b + 1, row[startCol + b + 1]),
                    IM_COL32(255, 155, 50, 240), 2.2f);
            }
        }

        // 4. Active Query Shape Curve (Bright Cyan)
        if (showQueryHist && queryId >= 0 && queryId < (int)featStand.features.size()) {
            const auto& row = featStand.features[queryId];
            for (int b = 0; b + 1 < bins && startCol + b + 1 < (int)row.size(); ++b) {
                ImVec2 p1 = toScreen(b, row[startCol + b]);
                ImVec2 p2 = toScreen(b + 1, row[startCol + b + 1]);
                dl->AddLine(p1, p2, IM_COL32(65, 200, 255, 255), 2.6f);
                dl->AddCircleFilled(p1, 2.5f, IM_COL32(65, 200, 255, 255));
            }
        }

        ImGui::InvisibleButton("HistCanvasBtn", canvasSize);
        if (ImGui::IsItemHovered()) {
            ImVec2 mp = ImGui::GetIO().MousePos;
            int hoveredBin = std::clamp(static_cast<int>(std::round(((mp.x - (canvasPos.x + padL)) / plotW) * (bins - 1))), 0, bins - 1);
            double qVal = (queryId >= 0 && queryId < (int)featStand.features.size()) ? featStand.features[queryId][startCol + hoveredBin] : 0.0;
            double cVal = (compId >= 0 && compId < (int)featStand.features.size()) ? featStand.features[compId][startCol + hoveredBin] : 0.0;
            ImGui::SetTooltip("%s - Bin %d / %d\nQuery #%d (%s): %.4f\nCompare #%d (%s): %.4f\nClass Mean (%s): %.4f",
                title, hoveredBin, bins,
                queryId, shapes[queryId].className.c_str(), qVal,
                compId, shapes[compId].className.c_str(), cVal,
                classes[classIdx].name.c_str(), classMean[hoveredBin]);
        }

        ImGui::EndChild();
    }

    void drawTabFeaturesAndHistograms() {
        const auto& qShape = shapes[selectedShapeId];
        const auto& cShape = shapes[std::clamp(compareShapeId, 0, (int)shapes.size() - 1)];

        ImGui::TextColored(ImVec4(0.35f, 0.85f, 1.0f, 1.0f),
            "Query Shape (Cyan): #%d %s (%s)", qShape.id, qShape.className.c_str(), qShape.fileName.c_str());
        ImGui::SameLine(0, 20);
        ImGui::TextColored(ImVec4(1.0f, 0.65f, 0.25f, 1.0f),
            "|  Comparison Shape (Orange): #%d %s (%s)", cShape.id, cShape.className.c_str(), cShape.fileName.c_str());

        ImGui::SetNextItemWidth(100.0f);
        ImGui::InputInt("Compare Shape ID", &compareShapeId);
        compareShapeId = std::clamp(compareShapeId, 0, (int)shapes.size() - 1);
        ImGui::SameLine(0, 16);
        ImGui::Checkbox("Query (Cyan)", &showQueryHist); ImGui::SameLine();
        ImGui::Checkbox("Comparison (Orange)", &showCompareHist); ImGui::SameLine();
        ImGui::Checkbox("Class Mean (Green)", &showClassMeanHist); ImGui::SameLine();
        ImGui::Checkbox("Class Spaghetti (Mode 3)", &showClassSpaghetti);

        ImGui::Separator();

        // Top section: 6 Elementary Features Comparison Table
        ImGui::TextColored(ImVec4(0.4f, 0.9f, 1.0f, 1.0f), "GROUP 0: 6 ELEMENTARY 3D SHAPE DESCRIPTORS (PRECOMPUTED)");
        const char* singleNames[6] = {
            "1. Surface Area", "2. Compactness (S^3 / 36pi V^2)", "3. Rectangularity (V / V_AABB)",
            "4. Diameter (Max Dist)", "5. Convexity (V / V_ConvexHull)", "6. Eccentricity (lambda_1 / lambda_3)"
        };

        if (ImGui::BeginTable("ElemFeatTable", 7, ImGuiTableFlags_Borders | ImGuiTableFlags_RowBg | ImGuiTableFlags_SizingStretchProp)) {
            ImGui::TableSetupColumn("Descriptor");
            ImGui::TableSetupColumn("Query Raw");
            ImGui::TableSetupColumn("Compare Raw");
            ImGui::TableSetupColumn("Query Z-Score (Stand)");
            ImGui::TableSetupColumn("Compare Z-Score (Stand)");
            ImGui::TableSetupColumn("Query Norm [0,1]");
            ImGui::TableSetupColumn("Database Mean / SD");
            ImGui::TableHeadersRow();

            for (int f = 0; f < numSingleFeatures; ++f) {
                ImGui::TableNextRow();
                ImGui::TableSetColumnIndex(0); ImGui::Text("%s", singleNames[f]);
                double qRaw = (selectedShapeId < (int)featRaw.features.size()) ? featRaw.features[selectedShapeId][f] : 0.0;
                double cRaw = (compareShapeId < (int)featRaw.features.size()) ? featRaw.features[compareShapeId][f] : 0.0;
                double qStd = (selectedShapeId < (int)featStand.features.size()) ? featStand.features[selectedShapeId][f] : 0.0;
                double cStd = (compareShapeId < (int)featStand.features.size()) ? featStand.features[compareShapeId][f] : 0.0;
                double qNrm = (selectedShapeId < (int)featNorm.features.size()) ? featNorm.features[selectedShapeId][f] : 0.0;
                double dbMean = (f < (int)stdData.means.size()) ? stdData.means[f] : 0.0;
                double dbSd = (f < (int)stdData.stddevs.size()) ? stdData.stddevs[f] : 0.0;

                ImGui::TableSetColumnIndex(1); ImGui::TextColored(ImVec4(0.35f, 0.85f, 1.0f, 1.0f), "%.4f", qRaw);
                ImGui::TableSetColumnIndex(2); ImGui::TextColored(ImVec4(1.0f, 0.65f, 0.25f, 1.0f), "%.4f", cRaw);
                ImGui::TableSetColumnIndex(3); ImGui::Text("%.3f sigma", qStd);
                ImGui::TableSetColumnIndex(4); ImGui::Text("%.3f sigma", cStd);
                ImGui::TableSetColumnIndex(5); ImGui::Text("%.4f", qNrm);
                ImGui::TableSetColumnIndex(6); ImGui::TextDisabled("mu=%.2f, sd=%.2f", dbMean, dbSd);
            }
            ImGui::EndTable();
        }

        ImGui::Spacing();
        ImGui::TextColored(ImVec4(0.4f, 0.9f, 1.0f, 1.0f),
            "GROUPS 1-5: OSADA SHAPE DISTRIBUTIONS (32 BINS - HOVER MOUSE FOR VALUES)");

        ImGui::Columns(2, "HistGridCols", false);
        drawCustomLineHistogram("A3 Distribution (Group 1)", "Angle between 3 random surface points", 0, selectedShapeId, compareShapeId, selectedClassIdx, 175.0f);
        drawCustomLineHistogram("D2 Distribution (Group 3)", "Euclidean distance between 2 random surface points", 2, selectedShapeId, compareShapeId, selectedClassIdx, 175.0f);
        drawCustomLineHistogram("D4 Distribution (Group 5)", "Cube root of tetrahedron volume (4 random points)", 4, selectedShapeId, compareShapeId, selectedClassIdx, 175.0f);

        ImGui::NextColumn();
        drawCustomLineHistogram("D1 Distribution (Group 2)", "Distance from Barycenter (0,0,0) to random point", 1, selectedShapeId, compareShapeId, selectedClassIdx, 175.0f);
        drawCustomLineHistogram("D3 Distribution (Group 4)", "Square root of triangle area (3 random surface points)", 3, selectedShapeId, compareShapeId, selectedClassIdx, 175.0f);

        // Distance breakdown card between Query and Compare shape
        ImGui::BeginChild("PairDistBox", ImVec2(0, 175.0f), true);
        ImGui::TextColored(ImVec4(0.4f, 1.0f, 0.55f, 1.0f), "PAIRWISE DISTANCE BREAKDOWN: #%d (%s) VS #%d (%s):",
            qShape.id, qShape.className.c_str(), cShape.id, cShape.className.c_str());
        if (selectedShapeId < (int)featStand.features.size() && compareShapeId < (int)featStand.features.size()) {
            auto gd = groupDistances(featStand.features[selectedShapeId], featStand.features[compareShapeId]);
            double tot = weightedDistance(featStand.features[selectedShapeId], featStand.features[compareShapeId], activeWeights);
            ImGui::Text("Weighted Total Distance: %.4f", tot);
            const char* gn[6] = { "Single (EMD)", "A3 (L2)", "D1 (L2)", "D2 (L2)", "D3 (L2)", "D4 (L2)" };
            for (int g = 0; g < 6; ++g) {
                float contrib = static_cast<float>((activeWeights[g] * gd[g]) / std::max(1e-9, tot));
                char buf[64];
                std::snprintf(buf, sizeof(buf), "%s: d=%.4f (contribution %.1f%%)", gn[g], gd[g], contrib * 100.0f);
                ImGui::ProgressBar(contrib, ImVec2(-1, 15), buf);
            }
        }
        ImGui::EndChild();
        ImGui::Columns(1);
    }

    // ========================================================================
    // TAB 4: INTERACTIVE 2D t-SNE PROJECTION & BRUSHING (MODE 7)
    // ========================================================================
    void drawTabTsneProjection() {
        ImGui::Columns(2, "TsneCols", true);
        ImGui::SetColumnWidth(0, std::max(600.0f, ImGui::GetWindowWidth() - 650.0f));

        // Top Controls
        ImGui::TextColored(ImVec4(0.4f, 0.9f, 1.0f, 1.0f),
            "2D BARNES-HUT t-SNE EMBEDDING (%d SHAPES FROM 166D FEATURE SPACE - PRECOMPUTED)", (int)tsnePoints.size());

        // Row 1: Filter Selection
        ImGui::Text("Filter:"); ImGui::SameLine();
        ImGui::RadioButton("All 69 Classes", &tsneFilterMode, 0); ImGui::SameLine();
        ImGui::RadioButton("Selected Class", &tsneFilterMode, 1); ImGui::SameLine();
        ImGui::RadioButton("Highlight Query + Top-K Neighbors", &tsneFilterMode, 2);

        // Row 2: View Controls (Point Radius & Reset View)
        ImGui::SetNextItemWidth(120.0f);
        ImGui::SliderFloat("Point Radius", &tsnePointRadius, 2.0f, 9.0f, "%.1f"); ImGui::SameLine(0, 15);
        if (ImGui::Button("Reset View")) { tsneZoom = 1.0f; tsnePan = ImVec2(0, 0); }

        ImVec2 canvasPos = ImGui::GetCursorScreenPos();
        ImVec2 canvasSize = ImGui::GetContentRegionAvail();
        canvasSize.y = std::max(200.0f, canvasSize.y - 4.0f);

        ImDrawList* dl = ImGui::GetWindowDrawList();
        dl->AddRectFilled(canvasPos, ImVec2(canvasPos.x + canvasSize.x, canvasPos.y + canvasSize.y), IM_COL32(18, 20, 26, 255), 6.0f);
        dl->AddRect(canvasPos, ImVec2(canvasPos.x + canvasSize.x, canvasPos.y + canvasSize.y), IM_COL32(65, 72, 88, 255), 6.0f);
        dl->PushClipRect(canvasPos, ImVec2(canvasPos.x + canvasSize.x, canvasPos.y + canvasSize.y), true);

        ImVec2 center(canvasPos.x + canvasSize.x * 0.5f + tsnePan.x, canvasPos.y + canvasSize.y * 0.5f + tsnePan.y);
        float scale = std::min(canvasSize.x, canvasSize.y) * 0.48f * tsneZoom;

        auto toScreen = [&](float nx, float ny) -> ImVec2 {
            return ImVec2(center.x + nx * scale, center.y - ny * scale);
        };

        // Build set of Top-K neighbor IDs
        std::unordered_map<int, int> neighborRank;
        for (size_t i = 0; i < queryResults.size(); ++i) {
            neighborRank[queryResults[i].shapeId] = static_cast<int>(i);
        }

        // Draw connecting lines from Query to Top-K neighbors
        if (selectedShapeId >= 0 && selectedShapeId < (int)tsnePoints.size()) {
            ImVec2 qPos = toScreen(tsnePoints[selectedShapeId].x, tsnePoints[selectedShapeId].y);
            for (const auto& qm : queryResults) {
                if (qm.shapeId >= 0 && qm.shapeId < (int)tsnePoints.size() && qm.shapeId != selectedShapeId) {
                    ImVec2 nPos = toScreen(tsnePoints[qm.shapeId].x, tsnePoints[qm.shapeId].y);
                    ImU32 lineCol = qm.sameClass ? IM_COL32(60, 240, 110, 140) : IM_COL32(255, 95, 80, 120);
                    dl->AddLine(qPos, nPos, lineCol, 1.6f);
                }
            }
        }

        // Draw all 2,483 points & find closest point to cursor (Brushing)
        ImVec2 mousePos = ImGui::GetIO().MousePos;
        int bestHoverId = -1;
        float bestDistSq = 144.0f; // 12 px radius threshold
        const std::string& activeClassName = classes[selectedClassIdx].name;

        for (const auto& pt : tsnePoints) {
            ImVec2 sp = toScreen(pt.x, pt.y);
            float dx = sp.x - mousePos.x;
            float dy = sp.y - mousePos.y;
            float d2 = dx * dx + dy * dy;
            if (d2 < bestDistSq) {
                bestDistSq = d2;
                bestHoverId = pt.id;
            }

            bool isEmphasized = true;
            if (tsneFilterMode == 1) {
                isEmphasized = (pt.label == activeClassName);
            } else if (tsneFilterMode == 2) {
                isEmphasized = (neighborRank.find(pt.id) != neighborRank.end());
            }

            int alpha = isEmphasized ? 235 : 32;
            float rad = isEmphasized ? tsnePointRadius : (tsnePointRadius * 0.65f);
            ImU32 col = IM_COL32((int)(pt.r * 255), (int)(pt.g * 255), (int)(pt.b * 255), alpha);
            dl->AddCircleFilled(sp, rad, col);
        }

        // Highlight Top-K neighbors with rings
        for (const auto& qm : queryResults) {
            if (qm.shapeId >= 0 && qm.shapeId < (int)tsnePoints.size()) {
                ImVec2 sp = toScreen(tsnePoints[qm.shapeId].x, tsnePoints[qm.shapeId].y);
                ImU32 ringCol = qm.sameClass ? IM_COL32(80, 255, 120, 255) : IM_COL32(255, 100, 80, 255);
                dl->AddCircle(sp, tsnePointRadius + 3.5f, ringCol, 16, 2.0f);
            }
        }

        // Highlight active query point
        if (selectedShapeId >= 0 && selectedShapeId < (int)tsnePoints.size()) {
            ImVec2 qPos = toScreen(tsnePoints[selectedShapeId].x, tsnePoints[selectedShapeId].y);
            dl->AddCircle(qPos, tsnePointRadius + 6.5f, IM_COL32(255, 255, 255, 255), 20, 2.5f);
            dl->AddCircle(qPos, tsnePointRadius + 4.0f, IM_COL32(50, 200, 255, 255), 20, 2.0f);
        }

        dl->PopClipRect();

        ImGui::InvisibleButton("TsneCanvas", canvasSize);
        if (ImGui::IsItemHovered()) {
            ImGuiIO& io = ImGui::GetIO();
            if (ImGui::IsMouseDragging(ImGuiMouseButton_Left) || ImGui::IsMouseDragging(ImGuiMouseButton_Right)) {
                tsnePan.x += io.MouseDelta.x;
                tsnePan.y += io.MouseDelta.y;
            }
            if (std::fabs(io.MouseWheel) > 1e-4f) {
                tsneZoom *= (io.MouseWheel > 0) ? 1.15f : 0.87f;
                tsneZoom = std::clamp(tsneZoom, 0.4f, 15.0f);
            }
            if (bestHoverId >= 0) {
                tsneHoveredShapeId = bestHoverId;
                ImGui::SetTooltip("t-SNE Brushing:\nShape #%d\nClass: %s\nFile: %s\n(Click to select as Query)",
                    bestHoverId, shapes[bestHoverId].className.c_str(), shapes[bestHoverId].path.c_str());
                if (ImGui::IsItemClicked(ImGuiMouseButton_Left)) {
                    selectShape(bestHoverId);
                }
            }
        }

        // Right Column: Live 3D Brushing Preview & Neighbor List
        ImGui::NextColumn();
        int previewId = (tsneHoveredShapeId >= 0) ? tsneHoveredShapeId : selectedShapeId;
        const auto& pShape = shapes[previewId];
        const auto& pMeshGL = getOrLoadNormMesh(previewId);

        ImGui::TextColored(ImVec4(1.0f, 0.9f, 0.35f, 1.0f),
            "3D BRUSHING PREVIEW: #%d (%s)", pShape.id, pShape.className.c_str());
        ImGui::TextDisabled("File: %s", pShape.path.c_str());

        drawInteractiveViewport(pMeshGL, 280.0f, "TsneVP");

        ImGui::Separator();
        ImGui::TextColored(ImVec4(0.4f, 0.9f, 1.0f, 1.0f), "QUERY & TOP-K RETRIEVED NEIGHBORS IN t-SNE SPACE:");
        ImGui::TextWrapped("Green lines in the t-SNE scatterplot connect the selected query shape (#%d, %s) to retrieved neighbors of the same class; red lines connect to different classes.",
            selectedShapeId, shapes[selectedShapeId].className.c_str());

        if (ImGui::BeginTable("TsneNeighbors", 4, ImGuiTableFlags_Borders | ImGuiTableFlags_RowBg | ImGuiTableFlags_ScrollY)) {
            ImGui::TableSetupColumn("Rank", ImGuiTableColumnFlags_WidthFixed, 40.0f);
            ImGui::TableSetupColumn("ID", ImGuiTableColumnFlags_WidthFixed, 45.0f);
            ImGui::TableSetupColumn("Class");
            ImGui::TableSetupColumn("Distance d", ImGuiTableColumnFlags_WidthFixed, 68.0f);
            ImGui::TableHeadersRow();
            for (size_t i = 0; i < queryResults.size(); ++i) {
                const auto& r = queryResults[i];
                ImGui::TableNextRow();
                ImGui::TableSetColumnIndex(0); ImGui::Text("#%d", (int)i + 1);
                ImGui::TableSetColumnIndex(1); ImGui::Text("%d", r.shapeId);
                ImGui::TableSetColumnIndex(2);
                ImVec4 col = r.sameClass ? ImVec4(0.35f, 0.95f, 0.45f, 1.0f) : ImVec4(1.0f, 0.45f, 0.4f, 1.0f);
                ImGui::TextColored(col, "%s", shapes[r.shapeId].className.c_str());
                ImGui::TableSetColumnIndex(3); ImGui::Text("%.4f", r.totalDist);
            }
            ImGui::EndTable();
        }

        ImGui::Columns(1);
    }

    // ========================================================================
    // TAB 5: SYSTEM EVALUATION & ROC CURVES DASHBOARD (MODE 6)
    // ========================================================================
    void drawTabEvaluation() {
        ImGui::Columns(2, "EvalCols", true);
        ImGui::SetColumnWidth(0, std::max(600.0f, ImGui::GetWindowWidth() * 0.46f));

        ImGui::TextColored(ImVec4(0.4f, 0.9f, 1.0f, 1.0f), "SYSTEM EVALUATION & ROC-AUC CURVES (MODE 6 - PRECOMPUTED)");
        ImGui::Checkbox("Standardized Linear (AUC 0.7820)", &showEvalStd); ImGui::SameLine();
        ImGui::Checkbox("Improved Matching (AUC 0.7905)", &showEvalImp);
        ImGui::Checkbox("ANN kd-Tree (AUC 0.8033)", &showEvalAnn); ImGui::SameLine();
        ImGui::Checkbox("Selected Class Curve", &showEvalSelectedClass);

        ImGui::Text("X-Axis Metric:"); ImGui::SameLine();
        if (ImGui::RadioButton("Specificity (TNR)", !plotStandardROC)) plotStandardROC = false;
        ImGui::SameLine();
        if (ImGui::RadioButton("1 - Specificity (FPR, Traditional ROC)", plotStandardROC)) plotStandardROC = true;

        // Draw ROC Curve Canvas
        ImVec2 canvasPos = ImGui::GetCursorScreenPos();
        ImVec2 canvasSize(ImGui::GetContentRegionAvail().x, 380.0f);
        ImDrawList* dl = ImGui::GetWindowDrawList();
        dl->AddRectFilled(canvasPos, ImVec2(canvasPos.x + canvasSize.x, canvasPos.y + canvasSize.y), IM_COL32(20, 23, 30, 255), 6.0f);
        dl->AddRect(canvasPos, ImVec2(canvasPos.x + canvasSize.x, canvasPos.y + canvasSize.y), IM_COL32(65, 72, 88, 255), 6.0f);

        float padL = 42.0f, padR = 18.0f, padT = 18.0f, padB = 34.0f;
        float plotW = std::max(20.0f, canvasSize.x - padL - padR);
        float plotH = std::max(20.0f, canvasSize.y - padT - padB);

        auto toPt = [&](float sens, float spec) -> ImVec2 {
            float xVal = plotStandardROC ? (1.0f - spec) : spec;
            float x = canvasPos.x + padL + std::clamp(xVal, 0.0f, 1.0f) * plotW;
            float y = canvasPos.y + padT + (1.0f - std::clamp(sens, 0.0f, 1.0f)) * plotH;
            return ImVec2(x, y);
        };

        // Grid & Diagonal Random Guess Line
        for (int g = 0; g <= 4; ++g) {
            float f = g / 4.0f;
            float gx = canvasPos.x + padL + f * plotW;
            float gy = canvasPos.y + padT + (1.0f - f) * plotH;
            dl->AddLine(ImVec2(gx, canvasPos.y + padT), ImVec2(gx, canvasPos.y + padT + plotH), IM_COL32(42, 47, 58, 255));
            dl->AddLine(ImVec2(canvasPos.x + padL, gy), ImVec2(canvasPos.x + padL + plotW, gy), IM_COL32(42, 47, 58, 255));
            char lbl[16];
            std::snprintf(lbl, sizeof(lbl), "%.2f", f);
            dl->AddText(ImVec2(canvasPos.x + 6, gy - 7), IM_COL32(160, 170, 185, 255), lbl);
            dl->AddText(ImVec2(gx - 12, canvasPos.y + padT + plotH + 6), IM_COL32(160, 170, 185, 255), lbl);
        }

        // Diagonal random baseline (AUC = 0.50)
        if (plotStandardROC) {
            dl->AddLine(toPt(0.0f, 1.0f), toPt(1.0f, 0.0f), IM_COL32(120, 120, 130, 140), 1.5f);
        } else {
            dl->AddLine(toPt(0.0f, 1.0f), toPt(1.0f, 0.0f), IM_COL32(120, 120, 130, 140), 1.5f);
        }

        auto drawCurve = [&](const std::vector<std::pair<float, float>>& pts, ImU32 col, float thick) {
            for (size_t i = 0; i + 1 < pts.size(); ++i) {
                dl->AddLine(toPt(pts[i].first, pts[i].second), toPt(pts[i + 1].first, pts[i + 1].second), col, thick);
            }
        };

        if (showEvalStd) drawCurve(evalStd.points, IM_COL32(70, 195, 255, 255), 2.5f);
        if (showEvalImp) drawCurve(evalImp.points, IM_COL32(85, 240, 130, 255), 2.5f);
        if (showEvalAnn) drawCurve(evalAnn.points, IM_COL32(255, 165, 55, 255), 2.5f);

        const std::string& selClassName = classes[selectedClassIdx].name;
        if (showEvalSelectedClass && evalStd.classPoints.count(selClassName)) {
            drawCurve(evalStd.classPoints[selClassName], IM_COL32(255, 90, 220, 255), 3.0f);
        }

        ImGui::Dummy(canvasSize);

        // Summary Benchmark Metrics Box
        ImGui::Spacing();
        ImGui::BeginChild("EvalSummaryBox", ImVec2(0, 0), true);
        ImGui::TextColored(ImVec4(0.4f, 1.0f, 0.55f, 1.0f), "SYSTEM BENCHMARK SUMMARY (2,483 MODELS, 69 CLASSES):");
        ImGui::Bullet(); ImGui::TextWrapped("Global ROC-AUC (Standardized Linear): 0.7820 | After Improvements: 0.7905 | ANN kd-tree: 0.8033");
        ImGui::Bullet(); ImGui::TextWrapped("Precision@1 (excl. self): 55.05%% (vs. 1.45%% for random guess over 69 classes)");
        ImGui::Bullet(); ImGui::TextWrapped("Top Performing Classes: Bicycle (AUC 0.9763, 100%% P@1), ComputerKeyboard (AUC 0.9592), Glasses (AUC 0.9495)");
        ImGui::Bullet(); ImGui::TextWrapped("Selected Class '%s' (%d shapes): AUC Standard = %.4f | AUC Improved = %.4f | AUC ANN = %.4f",
            selClassName.c_str(), classes[selectedClassIdx].count,
            classes[selectedClassIdx].aucStd, classes[selectedClassIdx].aucImp, classes[selectedClassIdx].aucAnn);
        ImGui::EndChild();

        // Right Column: Sortable Per-Class AUC Table
        ImGui::NextColumn();
        ImGui::TextColored(ImVec4(0.4f, 0.9f, 1.0f, 1.0f), "PER-CLASS EVALUATION BENCHMARK (CLICK ROW TO VIEW CLASS ROC CURVE)");

        // Sort classes by AUC descending for easy presentation
        std::vector<int> sortedClassIndices(classes.size());
        for (size_t i = 0; i < classes.size(); ++i) sortedClassIndices[i] = static_cast<int>(i);
        std::sort(sortedClassIndices.begin(), sortedClassIndices.end(), [&](int a, int b) {
            return classes[a].aucStd > classes[b].aucStd;
        });

        if (ImGui::BeginTable("PerClassEvalTable", 7, ImGuiTableFlags_Borders | ImGuiTableFlags_RowBg | ImGuiTableFlags_ScrollY)) {
            ImGui::TableSetupColumn("Rank", ImGuiTableColumnFlags_WidthFixed, 38.0f);
            ImGui::TableSetupColumn("Class");
            ImGui::TableSetupColumn("Count", ImGuiTableColumnFlags_WidthFixed, 48.0f);
            ImGui::TableSetupColumn("AUC Linear", ImGuiTableColumnFlags_WidthFixed, 72.0f);
            ImGui::TableSetupColumn("AUC Impr.", ImGuiTableColumnFlags_WidthFixed, 72.0f);
            ImGui::TableSetupColumn("AUC ANN", ImGuiTableColumnFlags_WidthFixed, 72.0f);
            ImGui::TableSetupColumn("Demo Action", ImGuiTableColumnFlags_WidthFixed, 85.0f);
            ImGui::TableHeadersRow();

            for (size_t r = 0; r < sortedClassIndices.size(); ++r) {
                int cIdx = sortedClassIndices[r];
                const auto& ci = classes[cIdx];
                ImGui::TableNextRow();
                ImGui::TableSetColumnIndex(0); ImGui::Text("#%d", (int)r + 1);
                ImGui::TableSetColumnIndex(1);
                bool isSel = (selectedClassIdx == cIdx);
                if (ImGui::Selectable(ci.name.c_str(), isSel)) {
                    selectedClassIdx = cIdx;
                }
                ImGui::TableSetColumnIndex(2); ImGui::Text("%d", ci.count);
                ImGui::TableSetColumnIndex(3);
                ImVec4 aucCol = (ci.aucStd >= 0.85f) ? ImVec4(0.35f, 0.95f, 0.45f, 1.0f)
                              : (ci.aucStd < 0.65f ? ImVec4(1.0f, 0.5f, 0.4f, 1.0f) : ImVec4(0.9f, 0.9f, 0.9f, 1.0f));
                ImGui::TextColored(aucCol, "%.4f", ci.aucStd);
                ImGui::TableSetColumnIndex(4); ImGui::Text("%.4f", ci.aucImp);
                ImGui::TableSetColumnIndex(5); ImGui::Text("%.4f", ci.aucAnn);
                ImGui::TableSetColumnIndex(6);
                ImGui::PushID(cIdx + 4000);
                if (ImGui::SmallButton("Query Demo")) {
                    selectedClassIdx = cIdx;
                    if (!ci.shapeIds.empty()) selectShape(ci.shapeIds[0]);
                    requestedTab = 0;
                    activeTab = 0;
                }
                ImGui::PopID();
            }
            ImGui::EndTable();
        }

        ImGui::Columns(1);
    }

public:
    void renderFrame(float deltaTime) {
        if (autoRotateTurntable) {
            camYaw += turntableSpeed * deltaTime;
        }

        ImGuiViewport* viewport = ImGui::GetMainViewport();
        ImGui::SetNextWindowPos(viewport->Pos);
        ImGui::SetNextWindowSize(viewport->Size);
        ImGuiWindowFlags rootFlags = ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoResize |
                                     ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoCollapse |
                                     ImGuiWindowFlags_NoBringToFrontOnFocus;

        ImGui::Begin("MultimediaRetrievalRoot", nullptr, rootFlags);

        // Top Presentation Banner
        ImGui::TextColored(ImVec4(0.35f, 0.85f, 1.0f, 1.0f),
            "3D MULTIMEDIA RETRIEVAL SYSTEM - INTERACTIVE PRESENTATION & DEMO SUITE");
        if (ImGui::GetWindowWidth() > 1050.0f) {
            ImGui::SameLine(ImGui::GetWindowWidth() - 470.0f);
        } else {
            ImGui::Spacing();
        }
        ImGui::TextDisabled("Precomputed DB: %d shapes, %d classes (Loaded in %.0f ms) | %.0f FPS",
            (int)shapes.size(), (int)classes.size(), startupLoadTimeMs, ImGui::GetIO().Framerate);
        ImGui::Separator();

        // Left Sidebar
        drawSidebar();
        ImGui::SameLine();

        // Main Tabbed Workspace
        ImGui::BeginChild("MainWorkspace", ImVec2(0, 0), false);

        // Handle programmatic tab requests or keyboard shortcuts (1-5)
        if (requestedTab >= 0 && requestedTab < 5) {
            activeTab = requestedTab;
            requestedTab = -1;
        }
        if (!ImGui::GetIO().WantTextInput) {
            if (ImGui::IsKeyPressed(ImGuiKey_1)) activeTab = 0;
            if (ImGui::IsKeyPressed(ImGuiKey_2)) activeTab = 1;
            if (ImGui::IsKeyPressed(ImGuiKey_3)) activeTab = 2;
            if (ImGui::IsKeyPressed(ImGuiKey_4)) activeTab = 3;
            if (ImGui::IsKeyPressed(ImGuiKey_5)) activeTab = 4;
        }

        // Modern Navigation Bar with active tab highlight
        const char* tabNames[5] = {
            " 1. 3D Query Studio (Step 4 & 5) ",
            " 2. 3D Preprocessing & Outliers (Step 1 & 2) ",
            " 3. Features & Histograms (Step 3) ",
            " 4. 2D t-SNE Projection (Step 5) ",
            " 5. System Evaluation & ROC Curves (Step 6) "
        };

        ImGui::PushStyleVar(ImGuiStyleVar_FrameRounding, 4.0f);
        ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, ImVec2(4.0f, 0.0f));
        for (int i = 0; i < 5; ++i) {
            if (i > 0) ImGui::SameLine();
            bool isSelected = (activeTab == i);
            if (isSelected) {
                ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.20f, 0.50f, 0.85f, 1.0f));
                ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(0.26f, 0.58f, 0.95f, 1.0f));
                ImGui::PushStyleColor(ImGuiCol_ButtonActive, ImVec4(0.18f, 0.44f, 0.78f, 1.0f));
                ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(1.0f, 1.0f, 1.0f, 1.0f));
            } else {
                ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.13f, 0.15f, 0.20f, 1.0f));
                ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(0.20f, 0.25f, 0.35f, 1.0f));
                ImGui::PushStyleColor(ImGuiCol_ButtonActive, ImVec4(0.16f, 0.20f, 0.28f, 1.0f));
                ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.70f, 0.76f, 0.85f, 1.0f));
            }

            if (ImGui::Button(tabNames[i], ImVec2(0, 32.0f))) {
                activeTab = i;
            }
            ImGui::PopStyleColor(4);
        }
        ImGui::PopStyleVar(2);
        ImGui::Spacing();
        ImGui::Separator();
        ImGui::Spacing();

        // Render Active Tab Content
        switch (activeTab) {
        case 0: drawTabQueryStudio(); break;
        case 1: drawTabPreprocessingLab(); break;
        case 2: drawTabFeaturesAndHistograms(); break;
        case 3: drawTabTsneProjection(); break;
        case 4: drawTabEvaluation(); break;
        default: activeTab = 0; drawTabQueryStudio(); break;
        }

        ImGui::EndChild();

        ImGui::End();
    }
};

inline void applyModernDarkTheme() {
    ImGui::StyleColorsDark();
    ImGuiStyle& style = ImGui::GetStyle();
    style.WindowRounding = 5.0f;
    style.ChildRounding = 5.0f;
    style.FrameRounding = 4.0f;
    style.GrabRounding = 3.0f;
    style.TabRounding = 4.0f;
    style.FramePadding = ImVec2(6.0f, 4.0f);
    style.ItemSpacing = ImVec2(8.0f, 6.0f);

    ImVec4* colors = style.Colors;
    colors[ImGuiCol_WindowBg]       = ImVec4(0.08f, 0.09f, 0.12f, 1.00f);
    colors[ImGuiCol_ChildBg]        = ImVec4(0.11f, 0.12f, 0.16f, 1.00f);
    colors[ImGuiCol_FrameBg]        = ImVec4(0.16f, 0.18f, 0.24f, 1.00f);
    colors[ImGuiCol_FrameBgHovered] = ImVec4(0.22f, 0.26f, 0.34f, 1.00f);
    colors[ImGuiCol_FrameBgActive]  = ImVec4(0.26f, 0.32f, 0.42f, 1.00f);
    colors[ImGuiCol_Header]         = ImVec4(0.18f, 0.36f, 0.56f, 0.75f);
    colors[ImGuiCol_HeaderHovered]  = ImVec4(0.24f, 0.46f, 0.70f, 0.85f);
    colors[ImGuiCol_HeaderActive]   = ImVec4(0.28f, 0.54f, 0.82f, 1.00f);
    colors[ImGuiCol_Button]         = ImVec4(0.18f, 0.38f, 0.62f, 0.85f);
    colors[ImGuiCol_ButtonHovered]  = ImVec4(0.24f, 0.48f, 0.76f, 1.00f);
    colors[ImGuiCol_ButtonActive]   = ImVec4(0.15f, 0.32f, 0.52f, 1.00f);
    colors[ImGuiCol_Tab]            = ImVec4(0.14f, 0.22f, 0.34f, 0.90f);
    colors[ImGuiCol_TabHovered]     = ImVec4(0.26f, 0.48f, 0.74f, 0.95f);
    colors[ImGuiCol_TabActive]      = ImVec4(0.22f, 0.44f, 0.70f, 1.00f);
}

inline void runDemoUI(const std::vector<std::string>& pathList, const std::vector<int>& objCounts) {
    if (!glfwInit()) {
        std::cerr << "Error: Failed to initialize GLFW for Demo UI.\n";
        return;
    }

    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    glfwWindowHint(GLFW_MAXIMIZED, GLFW_TRUE);

    GLFWwindow* window = glfwCreateWindow(1600, 920, "3D Multimedia Retrieval - Demo & Presentation Suite", NULL, NULL);
    if (!window) {
        std::cerr << "Error: Failed to create GLFW window.\n";
        glfwTerminate();
        return;
    }

    glfwMakeContextCurrent(window);
    glfwSwapInterval(1); // Enable vsync for smooth 60 FPS presentation

    if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress)) {
        std::cerr << "Error: Failed to initialize GLAD.\n";
        glfwDestroyWindow(window);
        glfwTerminate();
        return;
    }

    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO();
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;
    applyModernDarkTheme();

    ImGui_ImplGlfw_InitForOpenGL(window, true);
    ImGui_ImplOpenGL3_Init("#version 330");

    MultimediaDemoApp app;
    app.init(pathList, objCounts);

    float lastTime = static_cast<float>(glfwGetTime());

    while (!glfwWindowShouldClose(window)) {
        glfwPollEvents();

        float now = static_cast<float>(glfwGetTime());
        float dt = now - lastTime;
        lastTime = now;

        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();

        app.renderFrame(dt);

        ImGui::Render();
        int displayW, displayH;
        glfwGetFramebufferSize(window, &displayW, &displayH);
        glBindFramebuffer(GL_FRAMEBUFFER, 0);
        glViewport(0, 0, displayW, displayH);
        glClearColor(0.08f, 0.09f, 0.12f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT);
        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

        glfwSwapBuffers(window);
    }

    app.cleanup();
    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();

    glfwDestroyWindow(window);
    glfwTerminate();
}

} // namespace DemoUI

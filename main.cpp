#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <filesystem>
#include <iostream>
#include <vector>
#include <algorithm>
#include <limits>
#include <omp.h>
#include <atomic>
#include "parseobj.h"
#include "object.h"
#include "camera.h"
#include "fileSelector.h"
#include "resampling.h"

#define GLM_ENABLE_EXPERIMENTAL
#include <glm/gtx/pca.hpp>
#include <glm/glm.hpp>


namespace fs = std::filesystem;
int application_mode;

// Render functions

const char* vertexShaderSource = "#version 330 core\n"
"layout (location = 0) in vec3 aPos;\n"
"uniform mat4 uMVP;\n"
"uniform mat4 uModelView;\n"
"out vec3 FragPosEye;\n"
"void main()\n"
"{\n"
"   FragPosEye = vec3(uModelView * vec4(aPos, 1.0));\n"
"   gl_Position = uMVP * vec4(aPos, 1.0);\n"
"}\0";

const char* fragmentShaderSource = "#version 330 core\n"
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
"}\n\0";


void renderObject(fs::path selectedFile)
{
    Parser p;
    Mesh obj = p.parseOFF(selectedFile.string());
    GLsizei vertexCount = static_cast<GLsizei>(obj.trivertsgl.size() / 3);

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
    std::cout << "colorLoc = " << colorLoc << std::endl;

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

        glUseProgram(shaderProgram);

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

void exportToCSV(const std::vector<int>& faces, const std::vector<int>& verts, const std::vector<int>& types, const std::vector<Point3>& barycentres, const std::vector<Vec3>& scales, const std::string& filename)
{
    std::ofstream file(filename + ".csv");

    if (!file.is_open())
        return;

    file << "Shape id; Number of faces; Number of vertices; Face type; Barycentre X; Barycentre Y; Barycenter Z; Scale X; Scale Y; Scale Z\n";

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

        file << "\n";
    }
}

// Database analyse functions

std::vector<Mesh> parseDatabase(fs::path path)
{
    int shape_limit = 0;
    int class_limit = 0;

    std::cout << "Type in the max number of classes to parse (0-69):\n";
    std::cin >> class_limit;

    std::cout << "Type in the max number of shapes per class to parse (0-100):\n";
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

        Point3 cumulative_point(0, 0, 0);
        for (const auto& point : m.vertices)
        {
            cumulative_point += point;
        }
        if (!m.vertices.empty())
        {
            cumulative_point /= static_cast<double>(m.vertices.size());
        }
        barycentres[i] = cumulative_point;

        scales[i] = { m.maxPoint - m.minPoint };
    }

    // Print summary to console cleanly
    for (int i = 0; i < n; ++i)
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
    }

    std::cout << "Export to CSV file? y/n\n";
    char c;
    std::cin >> c;

    if (c == 'y')
    {
        std::cout << "Enter CSV file name:\n";
        std::string filename;
        std::cin >> filename;
        exportToCSV(faces, verts, types, barycentres, scales, filename);
    }
    else if (c != 'y' && c != 'n')
        std::cout << "Invalid input. Try again.\n";
}

// Resample functions

std::vector<Mesh> resampleMeshes(std::vector<Mesh> meshes, int minVert = minVertCount, int maxVert = maxVertCount)
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
    return meshes;
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
    if (input.vertices.empty()) return;

    Point3 barycenter(0, 0, 0);
    for (const auto& v : input.vertices) {
        barycenter += v;
    }
    barycenter /= static_cast<double>(input.vertices.size());

    // translate the unique vertices
    for (auto& v : input.vertices) {
        v -= barycenter;
    }

    // Update old AABB into new one
    calculateAABB(input);

    input.updateGLBuffers();
}

void scaleUniformly(Mesh& input)
{
    if (input.vertices.empty()) return;

    Point3 minpoint = input.minPoint;
    Point3 maxpoint = input.maxPoint;

    float largest_dim = std::max({maxpoint.e[0] - minpoint.e[0], maxpoint.e[1] - minpoint.e[1], maxpoint.e[2] - minpoint.e[2]});
    if (largest_dim == 0.0f) return;

    double factor = 1.0 / largest_dim;
    for (auto& v : input.vertices)
    {
        v *= factor;
    }

    // Update old AABB into new one
    calculateAABB(input);

    input.updateGLBuffers();
}

void alignAxes(Mesh& input)
{
    // TODO: to be done with step 3.1

    // Step 1: Compute Covariance Matrix (3x3)
    // Can be calculated by first putting all vertices in a Nx3 matrix V (so each row of the matrix is a xyz of a vertex in the shape)
    // Then we can get C = (1 / N) * (V^t * V), where N is the number of vertices
    
    std::vector<glm::dvec3> V;
    V.reserve(input.vertices.size());
    for (const auto& v : input.vertices)
    {
        V.emplace_back(v.x(), v.y(), v.z());
    }
    
    glm::dvec3 C(0.0, 0.0, 0.0);
    

    // Step 2: Perform Eigen decomposition
    // We have to solve det(C - l * I) = 0, where I is the identity matrix and l is lambda --> solve for lambda
    // Since C is a 3x3 matrix, we get -l^3 + c2 * lambda^2 + c1 * lambda + c0 = 0 to get the eigenvalues (l1, l2, l3)
    // To get the eigenvectors, we plug the eigenvalues back into the original equations: (C - li * I) * vi = 0,
    // where li is the ith eigenvalue, and vi is the corresponding eigenvector.
    // We can use Gaussian elimination for this. After applying that, we have to normalize the result.
    // Largest eigenvalue means that its corresponding eigenvector represents the longest dimension of the shape --> same principal for the other values/vectors

    glm::dmat3 covMat = glm::computeCovarianceMatrix(V.data(), V.size(), C);
    
    glm::dvec3 evals;
    glm::dmat3 evecs; 
    int count = glm::findEigenvaluesSymReal(covMat, evals, evecs);
    if (count != 3) {
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
    
    // Rotate the vertices 
    for (auto& v : input.vertices) {
        glm::dvec3 p(v.x(), v.y(), v.z());
        glm::dvec3 pRot = glm::transpose(R) * p; // oftewel dot-product met e1, e2, e3
        v = Point3(pRot.x, pRot.y, pRot.z);
    }
    
    // Update old AABB into new one
    calculateAABB(input);
    if (application_mode == 1)
    {
        input.updateGLBuffers();
    }
}

void flipShape(Mesh& input)
{
    // TODO: to be done with step 3.1

    // The idea is to have the sides of the shapes that have more mass always be on the same side of some axis
    // We can do this by first calculating the centers of all triangles
    // Then, for x, y and z, we calculate fi = sum_t sign(C t,i)(C t,i)^2
    // where i = {x, y, z}, and (C t, i) is the ith coordinate of center of triangle t
    // Basically, all we do, is for each axis, calculate how much its weights towards either the positive or negative side
    // We can then make a 3x3 matrix F, where the diagonals are sign(fx), sign(fy) and sign(fz)
    // We multiply each set of vertex coordinates with F to apply the flipping

    // Update old AABB into new one
    calculateAABB(input);
}

void normalizeMesh(Mesh& input)
{
    translateBarycenter(input);
    //alignAxes(input);
    //flipShape(input);
    scaleUniformly(input);
}

std::vector<Mesh> normalizeMeshes(std::vector<Mesh> meshes)
{
    int total = static_cast<int>(meshes.size());
    std::cout << "Normalizing " << total << " meshes in parallel...\n";

    #pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < total; ++i)
    {
        normalizeMesh(meshes[i]);
    }

    std::cout << "Finished normalizing meshes.\n";
    return meshes;
}

// Main functions

int main()
{
    std::cout << "Multimedia Retrieval tool\n";
    std::cout << "Select mode to begin: \n";
    std::cout << "1. Rendering mode\n";
    std::cout << "2. Analyse mode\n";
    
    std::cin >> application_mode;

    // Rendering mode
    if (application_mode == 1)
    {
        fs::path selectedFile = selectModelFile("ObjectFiles");
        if (selectedFile.empty())
        {
            std::cout << "Error: File is empty.\n";
            return -1;
        }

        std::cout << "Loading file: " << selectedFile.string() << "\n";
        renderObject(selectedFile);
    }

    // Analyse mode
    else if (application_mode == 2)
    {
        // Analyse initial database
        fs::path objectFiles = "ObjectFiles";
        std::vector<Mesh> original_meshes = parseDatabase(objectFiles);
        std::vector<Mesh> resampled_meshes = resampleMeshes(original_meshes);
        analyseDatabase(resampled_meshes);

        // Normalization --> remeshing, translation, (rotation), (flipping), scaling
        std::vector<Mesh> normalized_meshes = normalizeMeshes(resampled_meshes);

        // Check if normalization tool works correctly
        analyseDatabase(normalized_meshes);
    }

    return 0;
}

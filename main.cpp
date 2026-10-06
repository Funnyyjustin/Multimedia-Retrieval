#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <filesystem>
#include <iostream>
#include <vector>
#include <algorithm>
#include <limits>
#include "parseobj.h"
#include "object.h"
#include "camera.h"
#include "fileSelector.h"
#include "resampling.h"

namespace fs = std::filesystem;

// Render functions

const char* vertexShaderSource = "#version 330 core\n"
"layout (location = 0) in vec3 aPos;\n"
"uniform mat4 uMVP;\n"
"void main()\n"
"{\n"
"   gl_Position = uMVP * vec4(aPos, 1.0);\n"
"}\0";
const char* fragmentShaderSource = "#version 330 core\n"
"uniform vec4 uColor;\n"
"out vec4 FragColor;\n"
"void main()\n"
"{\n"
"   FragColor = uColor;\n"
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

        int width, height;
        glfwGetFramebufferSize(window, &width, &height);
        float aspect = (height > 0) ? (float)width / (float)height : 1.0f;
        glm::mat4 projection = glm::perspective(glm::radians(45.0f), aspect, 0.1f, 100.0f);
        glm::mat4 mvp = projection * view * model;

        // Write to shader
        glUniformMatrix4fv(mvpLoc, 1, GL_FALSE, glm::value_ptr(mvp));

        // Draw models
        glBindVertexArray(VAO);

        // Draw face
        if (showFace)
        {
            glUniform4f(colorLoc, 1.0f, 0.5f, 0.2f, 1.0f);
            glDrawArrays(GL_TRIANGLES, 0, vertexCount);
        }

        // Draw edges
        if (showEdge)
        {
            glUniform4f(colorLoc, 0.5f, 0.25f, 0.1f, 1.0f);
            glLineWidth(2.0f);
            glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
            glDrawArrays(GL_TRIANGLES, 0, vertexCount);
            glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
        }

        // Draw node
        if (showNode)
        {
            glUniform4f(colorLoc, 1.0f, 1.0f, 1.0f, 1.0f);
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

void exportToCSV(const std::vector<int>& faces, const std::vector<int>& verts, const std::vector<int>& types, std::string filename)
{
    std::ofstream file(filename + ".csv");

    if (!file.is_open())
        return;

    file << "Shape id; Number of faces; Number of vertices; Face type\n";

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

        file << "\n";
    }
}

// Database analyse functions

std::vector<Mesh> parseDatabase(fs::path path)
{
    Parser p;
    int index = 0;

    int class_index = 0;

    int shape_limit = 0;
    int class_limit = 0;

    std::cout << "Type in the max number of classes to parse (0-69):\n";
    std::cin >> class_limit;

    std::cout << "Type in the max number of shapes per class to parse (0-100):\n";
    std::cin >> shape_limit;

    std::vector<std::vector<int>> faces_vector;
    std::vector<std::vector<int>> verts_vector;
    std::vector<std::vector<int>> types_vector;

    std::vector<std::vector<Mesh>> meshes_vector;

    std::cout << "Parsing database..\n";

    // 69 classes in total in the database
    for (const auto& folder : fs::directory_iterator(path))
    {
        if (class_index >= class_limit)
            continue;

        // Only check directories
        if (!folder.is_directory())
            continue;

        int shape_index = 0;

        std::vector<int> faces;
        std::vector<int> verts;
        std::vector<int> types;

        std::vector<Mesh> meshes;

        // Check every file in each subfolder
        // This should only be the files of the shape in that folder / class
        for (const auto& file : fs::directory_iterator(folder.path()))
        {
            if (shape_index >= shape_limit)
                continue;

            std::cout << "Parsing file " << index << ".\n";

            fs::path selectedFile = file.path();
            Mesh m = p.parseOFF(selectedFile.string());
            m.setLabel(folder.path().filename().string());
            meshes.push_back(m);

            if (selectedFile.empty())
            {
                std::cout << "File is empty.\n";
                continue;
            }

            faces.push_back(m.primsSize());
            verts.push_back(m.vertsSize());

            if (m.quadsOnly) types.push_back(0);
            else if (m.trisOnly) types.push_back(1);
            else types.push_back(2);

            shape_index++;
            index++;
        }

        faces_vector.push_back(faces);
        verts_vector.push_back(verts);
        types_vector.push_back(types);

        meshes_vector.push_back(meshes);

        class_index++;
    }

    std::vector<Mesh> meshes = flatten(meshes_vector);

    std::cout << "Finished parsing database.\n";

    return meshes;
}

void analyseDatabase(std::vector<Mesh> meshes)
{
    std::cout << "Analysing database..\n";

    std::vector<int> faces;
    std::vector<int> verts;
    std::vector<int> types;

    // 69 classes in total in the database
    for (Mesh m : meshes)
    {
        faces.push_back(m.primsSize());
        verts.push_back(m.vertsSize());

        string type = "Mix of tris and quads";
        if (m.quadsOnly)
        {
            type = "Quads only";
            types.push_back(0);
        }
        else if (m.trisOnly)
        {
            type = "Tris only";
            types.push_back(1);
        }
        else types.push_back(2);

        std::cout << "Class: " << m.getLabel() << "\n";
        std::cout << "Number of faces: " << m.primsSize() << "\n";
        std::cout << "Number of vertices: " << m.vertsSize() << "\n";
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
        exportToCSV(faces, verts, types, filename);
    }
    else if (c != 'y' && c != 'n')
        std::cout << "Invalid input. Try again.\n";
}

// Resample functions

std::vector<Mesh> resampleMeshes(std::vector<Mesh> meshes)
{
    std::vector<Mesh> result;

    for (Mesh m : meshes)
    {
        resampleMesh(m);
        result.push_back(m);
    }

    return result;
}

// Normalisation functions

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

    input.updateGLBuffers();
}

void scaleUniformly(Mesh& input)
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

    float largest_dim = std::max({x_max - x_min, y_max - y_min, z_max - z_min});
    if (largest_dim == 0.0f) return;

    double factor = 1.0 / largest_dim;
    for (auto& v : input.vertices)
    {
        v *= factor;
    }

    input.updateGLBuffers();
}

void alignAxes(Mesh& input)
{
    // TODO: to be done with step 3.1
}

void flipShape(Mesh& input)
{
    // TODO: to be done with step 3.1
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
    std::vector<Mesh> result;

    for (Mesh m : meshes)
    {
        normalizeMesh(m);
        result.push_back(m);
    }

    return result;
}

// Main functions

int main()
{
    std::cout << "Multimedia Retrieval tool\n";
    std::cout << "Select mode to begin: \n";
    std::cout << "1. Rendering mode\n";
    std::cout << "2. Analyse mode\n";
    
    int num;
    std::cin >> num;

    // Rendering mode
    if (num == 1)
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
    else if (num == 2)
    {
        // Analyse initial database
        fs::path objectFiles = "ObjectFiles";
        std::vector<Mesh> original_meshes = parseDatabase(objectFiles);
        analyseDatabase(original_meshes);

        // Resample outliers
        //std::vector<Mesh> resampled_meshes = resampleMeshes(original_meshes);

        // Analyse updated meshes
        //analyseDatabase(resampled_meshes);

        // Normalization --> remeshing, translation, (rotation), (flipping), scaling
        //std::vector<Mesh> normalized_meshes = normalizeMeshes(resampled_meshes);

        // Check if normalization tool works correctly
        //analyseDatabase(normalized_meshes);
    }

    return 0;
}

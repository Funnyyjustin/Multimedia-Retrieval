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

namespace fs = std::filesystem;

// Shaders required for the 3D shapes

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

// Helper functions

std::vector<int> flatten(std::vector<std::vector<int>>& vec)
{
    std::vector<int> result;

    for (const auto& v : vec) {
        result.insert(result.end(), v.begin(), v.end());
    }

    return result;
}

void exportToCSV(const std::vector<int>& faces, const std::vector<int>& verts, const std::vector<int>& types)
{
    std::ofstream file("histograms.csv");

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

void analyseDatabase(fs::path path)
{
    int index = 0;
    int class_index = 0;

    int class_limit;
    int shape_limit;
    std::cout << "Type in the max number of classes to analyze:\n";
    std::cin >> class_limit;

    std::cout << "Type in the max number of shapes per class to analyze:\n";
    std::cin >> shape_limit;

    std::vector<std::vector<int>> faces_vector;
    std::vector<std::vector<int>> verts_vector;
    std::vector<std::vector<int>> types_vector;

    // 69 classes in total.. nice ;)
    for (const auto& folder : fs::directory_iterator(objectFiles))
    {
        if (class_index >= class_limit)
            continue;

        int shape_index = 0;

        // Only check directories
        if (!folder.is_directory())
            continue;

        std::vector<int> faces;
        std::vector<int> verts;
        std::vector<int> types;

        // Check every file in each subfolder
        // This should only be the files of the shape in that folder / class
        for (const auto& file : fs::directory_iterator(folder.path()))
        {
            // Only check the first 'limit' shapes of each class
            if (shape_index >= shape_limit)
                continue;

            fs::path selectedFile = file.path();
            Mesh m = p.parseOFF(selectedFile.string());

            if (selectedFile.empty())
            {
                return -1;
            }

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

            std::cout << "Shape id: " << index << "\n";
            std::cout << "Shape filename: " << file.path().filename() << "\n";
            std::cout << "Class: " << folder.path().filename() << "\n";
            std::cout << "Number of faces: " << m.primsSize() << "\n";
            std::cout << "Number of vertices: " << m.vertsSize() << "\n";
            std::cout << "Type of faces: " << type << "\n";
            std::cout << "AABB: \n";
            std::cout << "  min: " << m.minPoint.e[0] << " " << m.minPoint.e[1] << " " << m.minPoint.e[2] << "\n";
            std::cout << "  max: " << m.maxPoint.e[0] << " " << m.maxPoint.e[1] << " " << m.maxPoint.e[2] << "\n";
            std::cout << "\n";
            shape_index++;
            index++;
        }

        faces_vector.push_back(faces);
        verts_vector.push_back(verts);
        types_vector.push_back(types);

        class_index++;
    }

    std::vector<int> faces = flatten(faces_vector);
    std::vector<int> verts = flatten(verts_vector);
    std::vector<int> types = flatten(types_vector);

    std::cout << "Export to CSV file? y/n\n";
    char c;
    std::cin >> c;

    if (c == 'y')
        exportToCSV(faces, verts, types);
    else if (c != 'y' && c != 'n')
        std::cout << "Invalid input. Try again.\n";
}

// Main functions

int main()
{
    Parser p;

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
            return -1;
        }

        std::cout << "Loading file: " << selectedFile.string() << "\n";

        Mesh obj = p.parseOFF(selectedFile.string());
        float vertexCount = obj.trivertsgl.size() / 3;

        //std::cout << obj.size() << std::endl;
        //obj.show();

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
            return -1;
        }
        glfwMakeContextCurrent(window);

        // Initialise GLAD
        if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress))
        {
            std::cout << "Failed to initialize GLAD" << std::endl;
            return -1;
        }

        glViewport(0, 0, 800, 600);
        glEnable(GL_DEPTH_TEST);

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

            // Process keyboard input
            processCameraKeyboard(window, deltaTime);

            glClearColor(0.1f, 0.1f, 0.1f, 1.0f);
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

            glUseProgram(shaderProgram);
            glUseProgram(fragmentShader);

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

    // Analyse mode
    else if (num == 2)
    {
        // Analyse initial database
        fs::path objectFiles = "ObjectFiles";
        analyseDatabase(objectFiles);

        // Resample outliers --> TODO (step 2.3)

        // Analyse updated database --> should uncomment the following two lines when step 2.3 has been implemented
        //fs::path updatedFiles = "UpdatedFiles";
        //analyseDatabase(updatedFiles);

        // Normalization tool --> TODO (step 2.5)

        // Check if normalization tool works correctly --> should uncomment the following two lines when step 2.5 has been implemented
        //fs::path translatedFiles = "TranslatedFiles";
        //analyseDatabase(translatedFiles);
    }

    return 0;
}

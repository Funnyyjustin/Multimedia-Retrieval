#pragma once
#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

// Camera parameters
float cameraDistance = 2.0f; // Afstand tot het object (zoom)
float cameraYaw = 0.0f;      // Horizontale rotatie (in radialen)
float cameraPitch = 0.0f;    // Verticale rotatie (in radialen)
glm::vec3 cameraTarget = glm::vec3(0.0f, 0.0f, 0.0f); // Waar de camera naar kijkt

// Muis status
bool isOrbiting = false; // Linkermuisknop = roteren
bool isPanning  = false; // Rechtermuisknop = pannen (links/rechts/op/neer)
double lastMouseX = 0.0;
double lastMouseY = 0.0;

// Hulpfunctie om huidige camera vectors te berekenen
inline void getCameraVectors(glm::vec3& outPos, glm::vec3& outRight, glm::vec3& outUp)
{
    outPos.x = cameraTarget.x + cameraDistance * cos(cameraPitch) * sin(cameraYaw);
    outPos.y = cameraTarget.y + cameraDistance * sin(cameraPitch);
    outPos.z = cameraTarget.z + cameraDistance * cos(cameraPitch) * cos(cameraYaw);

    glm::vec3 forward = glm::normalize(cameraTarget - outPos);
    outRight = glm::normalize(glm::cross(forward, glm::vec3(0.0f, 1.0f, 0.0f)));
    outUp    = glm::normalize(glm::cross(outRight, forward));
}

// Callback: muisklikken
inline void mouse_button_callback(GLFWwindow* window, int button, int action, int mods)
{
    if (button == GLFW_MOUSE_BUTTON_LEFT)
    {
        isOrbiting = (action == GLFW_PRESS);
        if (isOrbiting) glfwGetCursorPos(window, &lastMouseX, &lastMouseY);
    }
    else if (button == GLFW_MOUSE_BUTTON_RIGHT)
    {
        isPanning = (action == GLFW_PRESS);
        if (isPanning) glfwGetCursorPos(window, &lastMouseX, &lastMouseY);
    }
}

bool showEdge = true;
bool showNode = false;
bool showFace = true;

void key_callback(GLFWwindow* window, int key, int scancode, int action, int mods)
{
    // action == GLFW_PRESS vuurt exact één keer af bij het indrukken
    if (action == GLFW_PRESS)
    {
        if (key == GLFW_KEY_1) showEdge = !showEdge;
        if (key == GLFW_KEY_2) showNode = !showNode;
        if (key == GLFW_KEY_3) showFace = !showFace;
    }
}

// Callback: muisbeweging
inline void cursor_position_callback(GLFWwindow* window, double xpos, double ypos)
{
    float deltaX = static_cast<float>(xpos - lastMouseX);
    float deltaY = static_cast<float>(ypos - lastMouseY);
    lastMouseX = xpos;
    lastMouseY = ypos;

    // 1. Orbit (Draaien met linkermuisknop)
    if (isOrbiting)
    {
        float sensitivity = 0.005f;
        cameraYaw   += deltaX * sensitivity;
        cameraPitch += deltaY * sensitivity;

        const float maxPitch = glm::radians(89.0f);
        if (cameraPitch > maxPitch) cameraPitch = maxPitch;
        if (cameraPitch < -maxPitch) cameraPitch = -maxPitch;
    }

    // 2. Pan (Verschuiven met rechtermuisknop)
    if (isPanning)
    {
        glm::vec3 camPos, camRight, camUp;
        getCameraVectors(camPos, camRight, camUp);

        // Pas pan-snelheid aan op basis van de zoomafstand
        float panSpeed = 0.0015f * cameraDistance;
        cameraTarget -= camRight * (deltaX * panSpeed);
        cameraTarget += camUp    * (deltaY * panSpeed);
    }
}

// Callback: scrollen (zoomen)
inline void scroll_callback(GLFWwindow* window, double xoffset, double yoffset)
{
    float zoomSpeed = 0.2f;
    cameraDistance -= static_cast<float>(yoffset) * zoomSpeed;

    if (cameraDistance < 0.05f) cameraDistance = 0.05f;
    if (cameraDistance > 50.0f) cameraDistance = 50.0f;
}

// Toetsenbord input (WASD / pijltjestoetsen) voor bewegen
inline void processCameraKeyboard(GLFWwindow* window, float deltaTime)
{
    float speed = 1.5f * deltaTime; // Snelheid van bewegen

    glm::vec3 camPos, camRight, camUp;
    getCameraVectors(camPos, camRight, camUp);

    // Links / Rechts (A / D of Left / Right)
    if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS || glfwGetKey(window, GLFW_KEY_LEFT) == GLFW_PRESS)
        cameraTarget -= camRight * speed;
    if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS || glfwGetKey(window, GLFW_KEY_RIGHT) == GLFW_PRESS)
        cameraTarget += camRight * speed;

    // Omhoog / Omlaag (W / S of Up / Down)
    if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS || glfwGetKey(window, GLFW_KEY_UP) == GLFW_PRESS)
        cameraTarget += camUp * speed;
    if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS || glfwGetKey(window, GLFW_KEY_DOWN) == GLFW_PRESS)
        cameraTarget -= camUp * speed;
}

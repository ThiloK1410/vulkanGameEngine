#include "camera.h"

#include <cmath>
#include <glm/gtc/matrix_transform.hpp>

glm::mat4 Camera::viewMatrix() const { return glm::lookAt(position, position + direction, up); }

glm::mat4 Camera::projectionMatrix(float aspectRatio) const {
  glm::mat4 proj = glm::perspective(glm::radians(fov), aspectRatio, nearPlane, farPlane);
  proj[1][1] *= -1; // Vulkan Y-axis is flipped vs OpenGL
  return proj;
}

void Camera::orbitAround(glm::vec3 target, float radius, float angle) {
  position = target + glm::vec3{std::cos(angle) * radius, 0.f, std::sin(angle) * radius};
  direction = glm::normalize(target - position);
}

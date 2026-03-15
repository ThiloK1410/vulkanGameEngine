#pragma once

#include <glm/mat4x4.hpp>
#include <glm/vec3.hpp>

struct Camera {
  glm::vec3 position{0.f, 0.f, 3.f};
  glm::vec3 direction{0.f, 0.f, -1.f};
  glm::vec3 up{0.f, 1.f, 0.f};

  float fov{70.f};
  float nearPlane{0.1f};
  float farPlane{200.f};

  glm::mat4 viewMatrix() const;
  glm::mat4 projectionMatrix(float aspectRatio) const;

  // orbit around a target point at a given radius, using an angle in radians
  void orbitAround(glm::vec3 target, float radius, float angle);
};

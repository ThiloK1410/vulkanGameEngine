// vulkan_guide.h : Include file for standard system include files,
// or project specific include files.

#pragma once

#include <vulkan/vulkan.h>

#include <glm/mat4x4.hpp>
#include <glm/vec3.hpp>
#include <glm/vec4.hpp>
#include <vk_mem_alloc.h>

struct PushConstants {
  glm::mat4 mvp;
  glm::vec4 color;
};

struct Vertex {
  glm::vec3 position;
};

struct AllocatedBuffer {
  VkBuffer buffer;
  VmaAllocation allocation;
};

struct Mesh {
  AllocatedBuffer vertexBuffer;
  AllocatedBuffer indexBuffer;
  uint32_t indexCount;
};

// vulkan_guide.h : Include file for standard system include files,
// or project specific include files.
#pragma once

#include <array>
#include <deque>
#include <functional>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <vector>

#include <fmt/core.h>
#include <glm/mat4x4.hpp>
#include <glm/vec4.hpp>
#include <vk_mem_alloc.h>
#include <vulkan/vk_enum_string_helper.h>
#include <vulkan/vulkan.h>

#define VK_CHECK(x)                                                                                \
  do {                                                                                             \
    VkResult err = x;                                                                              \
    if (err) {                                                                                     \
      fmt::println("Detected Vulkan error: {}", string_VkResult(err));                             \
      abort();                                                                                     \
    }                                                                                              \
  } while (0)

void transition_image(VkCommandBuffer cmd, VkImage image,
                      VkImageLayout oldLayout, VkImageLayout newLayout,
                      VkPipelineStageFlags2 srcStage, VkAccessFlags2 srcAccess,
                      VkPipelineStageFlags2 dstStage, VkAccessFlags2 dstAccess);

void buffer_barrier(VkCommandBuffer cmd, VkBuffer buffer, VkDeviceSize size,
                    VkPipelineStageFlags2 srcStage, VkAccessFlags2 srcAccess,
                    VkPipelineStageFlags2 dstStage, VkAccessFlags2 dstAccess);

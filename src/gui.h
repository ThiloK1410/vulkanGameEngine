#pragma once

#include <vulkan/vulkan.h>

struct SDL_Window;
union SDL_Event;
struct TerrainParams;

class DebugGui {
public:
    void init(VkInstance instance, VkPhysicalDevice gpu, VkDevice device,
              uint32_t queueFamily, VkQueue queue,
              VkFormat colorFormat, VkFormat depthFormat, uint32_t imageCount,
              SDL_Window* window);
    void cleanup();
    void process_event(SDL_Event& e);
    void begin_frame();
    void render(VkCommandBuffer cmd, TerrainParams& params);
};
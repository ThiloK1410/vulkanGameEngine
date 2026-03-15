// vulkan_guide.h : Include file for standard system include files,
// or project specific include files.

#pragma once

#include <vector> // std::vector for swapchain image arrays

#include <vk_mem_alloc.h>

#include "camera.h"
#include "vk_types.h"

struct Pipelines {
  VkPipeline triangle;
  VkPipelineLayout triangleLayout;

  void cleanup(VkDevice device);
};

class VulkanEngine {
public:
  bool _isInitialized{false};
  int _frameNumber{0};
  bool stop_rendering{false};

  VkExtent2D _windowExtent{1700, 900};

  struct SDL_Window *_window{nullptr};

  // from <vulkan/vulkan.h> — core Vulkan handles
  VkInstance _instance;                      // connection between your app and the Vulkan library
  VkDebugUtilsMessengerEXT _debug_messenger; // callback handle for validation layer messages
  VkPhysicalDevice _chosenGPU;               // the physical GPU we selected
  VkDevice _device;                          // logical device — our interface to the GPU
  VkSurfaceKHR _surface;                     // the window surface Vulkan renders to

  VkQueue _graphicsQueue;                    // queue we submit draw commands to
  uint32_t _graphicsQueueFamily;             // index of the queue family

  VkCommandPool _commandPool;               // allocator for command buffers
  VkCommandBuffer _mainCommandBuffer;       // the command buffer we record into

  VkFence _renderFence;                          // CPU waits on this until GPU finishes rendering
  VkSemaphore _swapchainSemaphore;               // signals when swapchain image is acquired
  std::vector<VkSemaphore> _renderSemaphores;    // one per swapchain image, present waits on it

  VkSwapchainKHR _swapchain;                          // manages the images we present to the surface
  VkFormat _swapchainImageFormat;                      // pixel format of swapchain images (e.g. BGRA8)
  std::vector<VkImage> _swapchainImages;               // the actual images owned by the swapchain
  std::vector<VkImageView> _swapchainImageViews;       // "views" into the images (how shaders access them)
  VkExtent2D _swapchainExtent;                         // resolution of the swapchain images

  VmaAllocator _allocator;

  Camera _camera;
  Mesh _cubeMesh;

  Pipelines _pipelines;

  static VulkanEngine &Get();

  // initializes everything in the engine
  void init();

  // shuts down the engine
  void cleanup();

  // draw loop
  void draw();

  // run main loop
  void run();

private:
  void init_vulkan();
  void init_commands();
  void init_sync_structures();
  void init_swapchain();
  void init_pipelines();
  void init_meshes();
  void create_swapchain(uint32_t width, uint32_t height);
  void destroy_swapchain();
  AllocatedBuffer create_buffer(VkDeviceSize size, VkBufferUsageFlags usage, VmaMemoryUsage memoryUsage);
};

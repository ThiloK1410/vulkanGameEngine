

#include "vk_engine.h"

#include <chrono>
#include <cstring>
#include <thread>

#include <SDL3/SDL.h>
#include <SDL3/SDL_vulkan.h>
#include <VkBootstrap.h>

#include <fmt/core.h>
#include <glm/mat4x4.hpp>

#include "vk_initializers.h"
#include "vk_pipelines.h"
#include "vk_types.h"

#ifdef NDEBUG
constexpr bool bUseValidationLayers = false;
#else
constexpr bool bUseValidationLayers = true;
#endif

VulkanEngine *loadedEngine = nullptr;

VulkanEngine &VulkanEngine::Get() {
  assert(loadedEngine != nullptr);
  return *loadedEngine;
}

void VulkanEngine::init() {
  loadedEngine = this;
  assert(loadedEngine);
  // We initialize SDL and create a window with it.
  if (!SDL_Init(SDL_INIT_VIDEO)) {
    SDL_Log("SDL_Init failed: %s", SDL_GetError());
    return;
  }

  SDL_WindowFlags window_flags = (SDL_WindowFlags)(SDL_WINDOW_VULKAN);

  _window =
      SDL_CreateWindow("Vulkan Engine", _windowExtent.width, _windowExtent.height, window_flags);
  if (!_window) {
    SDL_Log("SDL_CreateWindow failed: %s", SDL_GetError());
    return;
  }

  init_vulkan();
  init_swapchain();
  init_commands();
  init_sync_structures();
  init_pipelines();
  init_meshes();

  // everything went fine
  _isInitialized = true;
}
void VulkanEngine::init_vulkan() {
  // 1. Create Vulkan instance
  vkb::InstanceBuilder builder;
  auto inst_ret = builder.set_app_name("Vulkan Engine")
                      .request_validation_layers(bUseValidationLayers)
                      .use_default_debug_messenger()
                      .require_api_version(1, 3, 0)
                      .build();

  if (!inst_ret) {
    throw std::runtime_error(
        fmt::format("Failed to create Vulkan instance: {}", inst_ret.error().message()));
  }

  vkb::Instance vkb_inst = inst_ret.value();
  _instance = vkb_inst.instance;
  _debug_messenger = vkb_inst.debug_messenger;

  // 2. Create window surface
  if (!SDL_Vulkan_CreateSurface(_window, _instance, nullptr, &_surface)) {
    throw std::runtime_error(fmt::format("Failed to create Vulkan surface: {}", SDL_GetError()));
  }

  // 3. Select physical device
  //   Enable Vulkan 1.3 features we need:
  //   - dynamicRendering: use vkCmdBeginRendering instead of render passes
  //   - synchronization2: improved pipeline barrier API (vkCmdPipelineBarrier2)
  VkPhysicalDeviceVulkan13Features features13{
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES,
      .synchronization2 = true,
      .dynamicRendering = true,
  };

  vkb::PhysicalDeviceSelector selector{vkb_inst};
  auto phys_ret = selector.set_minimum_version(1, 3)
                      .set_surface(_surface)
                      .set_required_features_13(features13)
                      .select();

  if (!phys_ret) {
    throw std::runtime_error(
        fmt::format("Failed to select a suitable GPU: {}", phys_ret.error().message()));
  }

  vkb::PhysicalDevice physicalDevice = phys_ret.value();
  _chosenGPU = physicalDevice.physical_device;

  // 4. Create logical device
  vkb::DeviceBuilder deviceBuilder{physicalDevice};
  auto dev_ret = deviceBuilder.build();

  if (!dev_ret) {
    throw std::runtime_error(
        fmt::format("Failed to create logical device: {}", dev_ret.error().message()));
  }

  vkb::Device vkbDevice = dev_ret.value();
  _device = vkbDevice.device;

  // 5. Get graphics queue
  auto queue_ret = vkbDevice.get_queue(vkb::QueueType::graphics);
  if (!queue_ret) {
    throw std::runtime_error(
        fmt::format("Failed to get graphics queue: {}", queue_ret.error().message()));
  }
  _graphicsQueue = queue_ret.value();
  _graphicsQueueFamily = vkbDevice.get_queue_index(vkb::QueueType::graphics).value();

  // 6. Create VMA allocator
  VmaAllocatorCreateInfo allocatorInfo{
      .physicalDevice = _chosenGPU,
      .device = _device,
      .instance = _instance,
  };
  vmaCreateAllocator(&allocatorInfo, &_allocator);

  fmt::print("Vulkan initialized — GPU: {}\n", physicalDevice.name);
}

void VulkanEngine::init_swapchain() {
  create_swapchain(_windowExtent.width, _windowExtent.height);
}

void VulkanEngine::create_swapchain(uint32_t width, uint32_t height) {
  vkb::SwapchainBuilder swapchainBuilder{_chosenGPU, _device, _surface};

  _swapchainImageFormat = VK_FORMAT_B8G8R8A8_UNORM;

  auto swap_ret = swapchainBuilder
                      // .set_desired_format — pixel format for swapchain images
                      //   BGRA8 UNORM is the most widely supported format
                      //   SRGB colorspace for correct gamma
                      .set_desired_format(VkSurfaceFormatKHR{
                          .format = _swapchainImageFormat,
                          .colorSpace = VK_COLOR_SPACE_SRGB_NONLINEAR_KHR,
                      })
                      // .set_desired_present_mode — FIFO is vsync, guaranteed supported
                      .set_desired_present_mode(VK_PRESENT_MODE_FIFO_KHR)
                      // .set_desired_extent — resolution of the swapchain images
                      .set_desired_extent(width, height)
                      // .add_image_usage_flags — we want to write directly into these images
                      //   from compute shaders or transfer operations, not just render passes
                      .add_image_usage_flags(VK_IMAGE_USAGE_TRANSFER_DST_BIT)
                      .build();

  if (!swap_ret) {
    throw std::runtime_error(
        fmt::format("Failed to create swapchain: {}", swap_ret.error().message()));
  }

  vkb::Swapchain vkbSwapchain = swap_ret.value();
  _swapchain = vkbSwapchain.swapchain;
  _swapchainExtent = vkbSwapchain.extent;
  _swapchainImages = vkbSwapchain.get_images().value();
  _swapchainImageViews = vkbSwapchain.get_image_views().value();
}

void VulkanEngine::destroy_swapchain() {
  vkDestroySwapchainKHR(_device, _swapchain, nullptr);

  for (auto& imageView : _swapchainImageViews) {
    vkDestroyImageView(_device, imageView, nullptr);
  }
}

void VulkanEngine::init_commands() {
  // Create command pool
  //   .flags = RESET_COMMAND_BUFFER_BIT — allows individual command buffers to be reset and
  //            re-recorded, rather than having to reset the entire pool
  //   .queueFamilyIndex — command buffers from this pool can only be submitted to queues
  //            of this family
  VkCommandPoolCreateInfo commandPoolInfo{
      .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
      .flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
      .queueFamilyIndex = _graphicsQueueFamily,
  };

  VK_CHECK(vkCreateCommandPool(_device, &commandPoolInfo, nullptr, &_commandPool));

  // Allocate a single command buffer from the pool
  //   .commandPool — which pool to allocate from
  //   .level = PRIMARY — can be submitted directly to a queue (SECONDARY buffers are called
  //            from primary ones, used for multi-threaded recording)
  //   .commandBufferCount — how many buffers to allocate
  VkCommandBufferAllocateInfo cmdAllocInfo{
      .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
      .commandPool = _commandPool,
      .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
      .commandBufferCount = 1,
  };

  VK_CHECK(vkAllocateCommandBuffers(_device, &cmdAllocInfo, &_mainCommandBuffer));
}

void VulkanEngine::init_sync_structures() {
  // .flags = SIGNALED — fence starts signaled so the first frame doesn't wait forever
  VkFenceCreateInfo fenceCreateInfo{
      .sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO,
      .flags = VK_FENCE_CREATE_SIGNALED_BIT,
  };

  VK_CHECK(vkCreateFence(_device, &fenceCreateInfo, nullptr, &_renderFence));

  // Semaphores need no flags
  VkSemaphoreCreateInfo semaphoreCreateInfo{
      .sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO,
  };

  VK_CHECK(vkCreateSemaphore(_device, &semaphoreCreateInfo, nullptr, &_swapchainSemaphore));

  // One render semaphore per swapchain image to avoid reuse conflicts
  _renderSemaphores.resize(_swapchainImages.size());
  for (auto& sem : _renderSemaphores) {
    VK_CHECK(vkCreateSemaphore(_device, &semaphoreCreateInfo, nullptr, &sem));
  }
}

AllocatedBuffer VulkanEngine::create_buffer(VkDeviceSize size, VkBufferUsageFlags usage,
                                             VmaMemoryUsage memoryUsage) {
  VkBufferCreateInfo bufferInfo{
      .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
      .size = size,
      .usage = usage,
  };

  VmaAllocationCreateInfo allocInfo{
      .usage = memoryUsage,
  };

  AllocatedBuffer buffer;
  VK_CHECK(vmaCreateBuffer(_allocator, &bufferInfo, &allocInfo, &buffer.buffer, &buffer.allocation,
                           nullptr));
  return buffer;
}

void VulkanEngine::init_meshes() {
  // 8 corners of a unit cube
  Vertex vertices[] = {
      {{-0.5f, -0.5f, -0.5f}}, // 0: left  bottom back
      {{ 0.5f, -0.5f, -0.5f}}, // 1: right bottom back
      {{ 0.5f,  0.5f, -0.5f}}, // 2: right top    back
      {{-0.5f,  0.5f, -0.5f}}, // 3: left  top    back
      {{-0.5f, -0.5f,  0.5f}}, // 4: left  bottom front
      {{ 0.5f, -0.5f,  0.5f}}, // 5: right bottom front
      {{ 0.5f,  0.5f,  0.5f}}, // 6: right top    front
      {{-0.5f,  0.5f,  0.5f}}, // 7: left  top    front
  };

  // 12 edges, 2 indices each = 24 indices
  uint16_t indices[] = {
      // back face edges
      0, 1, 1, 2, 2, 3, 3, 0,
      // front face edges
      4, 5, 5, 6, 6, 7, 7, 4,
      // connecting edges (back to front)
      0, 4, 1, 5, 2, 6, 3, 7,
  };

  // create and upload vertex buffer
  _cubeMesh.vertexBuffer =
      create_buffer(sizeof(vertices), VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, VMA_MEMORY_USAGE_CPU_TO_GPU);

  void* data;
  vmaMapMemory(_allocator, _cubeMesh.vertexBuffer.allocation, &data);
  memcpy(data, vertices, sizeof(vertices));
  vmaUnmapMemory(_allocator, _cubeMesh.vertexBuffer.allocation);

  // create and upload index buffer
  _cubeMesh.indexBuffer =
      create_buffer(sizeof(indices), VK_BUFFER_USAGE_INDEX_BUFFER_BIT, VMA_MEMORY_USAGE_CPU_TO_GPU);

  vmaMapMemory(_allocator, _cubeMesh.indexBuffer.allocation, &data);
  memcpy(data, indices, sizeof(indices));
  vmaUnmapMemory(_allocator, _cubeMesh.indexBuffer.allocation);

  _cubeMesh.indexCount = sizeof(indices) / sizeof(indices[0]);

  fmt::print("Cube mesh created — {} vertices, {} indices\n", sizeof(vertices) / sizeof(vertices[0]),
             _cubeMesh.indexCount);
}

void Pipelines::cleanup(VkDevice device) {
  vkDestroyPipeline(device, triangle, nullptr);
  vkDestroyPipelineLayout(device, triangleLayout, nullptr);
}

void VulkanEngine::init_pipelines() {
  // push constants: mat4 MVP (vertex) + vec4 color (fragment)
  VkPushConstantRange pushConstantRange{
      .stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
      .offset = 0,
      .size = sizeof(PushConstants),
  };

  VkPipelineLayoutCreateInfo layoutInfo{
      .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
      .pushConstantRangeCount = 1,
      .pPushConstantRanges = &pushConstantRange,
  };
  VK_CHECK(vkCreatePipelineLayout(_device, &layoutInfo, nullptr, &_pipelines.triangleLayout));

  // load shaders (resolve relative to the executable's directory)
  std::string basePath = SDL_GetBasePath();

  VkShaderModule vertShader;
  if (!load_shader_module((basePath + "shaders/cube.vert.spv").c_str(), _device, &vertShader)) {
    throw std::runtime_error("Failed to load cube vertex shader");
  }
  VkShaderModule fragShader;
  if (!load_shader_module((basePath + "shaders/cube.frag.spv").c_str(), _device, &fragShader)) {
    throw std::runtime_error("Failed to load cube fragment shader");
  }

  // build the pipeline
  PipelineBuilder builder;
  builder._pipelineLayout = _pipelines.triangleLayout;
  builder.set_shaders(vertShader, fragShader);
  builder.set_input_topology(VK_PRIMITIVE_TOPOLOGY_LINE_LIST);
  builder.set_polygon_mode(VK_POLYGON_MODE_FILL);
  builder.set_cull_mode(VK_CULL_MODE_NONE, VK_FRONT_FACE_CLOCKWISE);
  builder.set_vertex_input(sizeof(Vertex), {
      {.location = 0, .binding = 0, .format = VK_FORMAT_R32G32B32_SFLOAT, .offset = 0},
  });
  builder.set_multisampling_none();
  builder.disable_blending();
  builder.disable_depthtest();
  builder.set_color_attachment_format(_swapchainImageFormat);

  _pipelines.triangle = builder.build(_device);

  // shader modules can be destroyed after pipeline creation
  vkDestroyShaderModule(_device, vertShader, nullptr);
  vkDestroyShaderModule(_device, fragShader, nullptr);

  fmt::print("Cube pipeline created\n");
}

void VulkanEngine::cleanup() {
  if (_isInitialized) {
    vkDeviceWaitIdle(_device);

    _pipelines.cleanup(_device);

    vmaDestroyBuffer(_allocator, _cubeMesh.vertexBuffer.buffer, _cubeMesh.vertexBuffer.allocation);
    vmaDestroyBuffer(_allocator, _cubeMesh.indexBuffer.buffer, _cubeMesh.indexBuffer.allocation);
    vmaDestroyAllocator(_allocator);

    vkDestroyFence(_device, _renderFence, nullptr);
    vkDestroySemaphore(_device, _swapchainSemaphore, nullptr);
    for (auto& sem : _renderSemaphores) {
      vkDestroySemaphore(_device, sem, nullptr);
    }
    vkDestroyCommandPool(_device, _commandPool, nullptr);
    destroy_swapchain();
    vkDestroySurfaceKHR(_instance, _surface, nullptr);
    vkDestroyDevice(_device, nullptr);
    vkb::destroy_debug_utils_messenger(_instance, _debug_messenger);
    vkDestroyInstance(_instance, nullptr);

    SDL_DestroyWindow(_window);
  }
}

void VulkanEngine::draw() {
  // Wait for GPU to finish previous frame (1s timeout)
  VK_CHECK(vkWaitForFences(_device, 1, &_renderFence, true, 1000000000));
  VK_CHECK(vkResetFences(_device, 1, &_renderFence));

  // Get the next swapchain image
  uint32_t swapchainImageIndex;
  VK_CHECK(vkAcquireNextImageKHR(_device, _swapchain, 1000000000, _swapchainSemaphore, nullptr,
                                 &swapchainImageIndex));

  // Reset and begin recording
  VK_CHECK(vkResetCommandBuffer(_mainCommandBuffer, 0));

  VkCommandBufferBeginInfo cmdBeginInfo{
      .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
      // ONE_TIME_SUBMIT — we re-record every frame, lets the driver optimize for that
      .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
  };

  VK_CHECK(vkBeginCommandBuffer(_mainCommandBuffer, &cmdBeginInfo));

  // Clear color that cycles over time
  float flash = std::abs(std::sin(_frameNumber / 120.f));

  // Transition image layout: UNDEFINED → COLOR_ATTACHMENT_OPTIMAL
  VkImageMemoryBarrier2 clearBarrier{
      .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2,
      .srcStageMask = VK_PIPELINE_STAGE_2_NONE,
      .srcAccessMask = VK_ACCESS_2_NONE,
      .dstStageMask = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
      .dstAccessMask = VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT,
      .oldLayout = VK_IMAGE_LAYOUT_UNDEFINED,
      .newLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
      .image = _swapchainImages[swapchainImageIndex],
      .subresourceRange = {
          .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
          .levelCount = 1,
          .layerCount = 1,
      },
  };

  VkDependencyInfo clearDep{
      .sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
      .imageMemoryBarrierCount = 1,
      .pImageMemoryBarriers = &clearBarrier,
  };
  vkCmdPipelineBarrier2(_mainCommandBuffer, &clearDep);

  // Begin dynamic rendering — clears the image and provides a target for draw commands
  VkRenderingAttachmentInfo colorAttachment{
      .sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO,
      .imageView = _swapchainImageViews[swapchainImageIndex],
      .imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
      .loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR,
      .storeOp = VK_ATTACHMENT_STORE_OP_STORE,
      .clearValue = {.color = {{0.0f, 0.0f, flash, 1.0f}}},
  };

  VkRenderingInfo renderInfo{
      .sType = VK_STRUCTURE_TYPE_RENDERING_INFO,
      .renderArea = {.extent = _swapchainExtent},
      .layerCount = 1,
      .colorAttachmentCount = 1,
      .pColorAttachments = &colorAttachment,
  };

  vkCmdBeginRendering(_mainCommandBuffer, &renderInfo);

  // set viewport and scissor to cover the full swapchain image
  VkViewport viewport{
      .x = 0,
      .y = 0,
      .width = (float)_swapchainExtent.width,
      .height = (float)_swapchainExtent.height,
      .minDepth = 0.f,
      .maxDepth = 1.f,
  };
  vkCmdSetViewport(_mainCommandBuffer, 0, 1, &viewport);

  VkRect2D scissor{
      .offset = {0, 0},
      .extent = _swapchainExtent,
  };
  vkCmdSetScissor(_mainCommandBuffer, 0, 1, &scissor);

  vkCmdBindPipeline(_mainCommandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, _pipelines.triangle);

  // orbit camera around the origin using frame number
  float angle = _frameNumber / 120.f;
  _camera.orbitAround(glm::vec3{0.f}, 3.f, angle);

  float aspectRatio = (float)_swapchainExtent.width / (float)_swapchainExtent.height;
  glm::mat4 mvp = _camera.projectionMatrix(aspectRatio) * _camera.viewMatrix();

  PushConstants push{
      .mvp = mvp,
      .color = {1.0f, 1.0f, 1.0f, 1.0f},
  };
  vkCmdPushConstants(_mainCommandBuffer, _pipelines.triangleLayout,
                     VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(push),
                     &push);

  // bind vertex and index buffers
  VkDeviceSize offset = 0;
  vkCmdBindVertexBuffers(_mainCommandBuffer, 0, 1, &_cubeMesh.vertexBuffer.buffer, &offset);
  vkCmdBindIndexBuffer(_mainCommandBuffer, _cubeMesh.indexBuffer.buffer, 0, VK_INDEX_TYPE_UINT16);

  vkCmdDrawIndexed(_mainCommandBuffer, _cubeMesh.indexCount, 1, 0, 0, 0);

  vkCmdEndRendering(_mainCommandBuffer);

  // Transition image layout: COLOR_ATTACHMENT_OPTIMAL → PRESENT_SRC
  VkImageMemoryBarrier2 presentBarrier{
      .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2,
      .srcStageMask = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
      .srcAccessMask = VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT,
      .dstStageMask = VK_PIPELINE_STAGE_2_NONE,
      .dstAccessMask = VK_ACCESS_2_NONE,
      .oldLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
      .newLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
      .image = _swapchainImages[swapchainImageIndex],
      .subresourceRange = {
          .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
          .levelCount = 1,
          .layerCount = 1,
      },
  };

  VkDependencyInfo presentDep{
      .sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
      .imageMemoryBarrierCount = 1,
      .pImageMemoryBarriers = &presentBarrier,
  };
  vkCmdPipelineBarrier2(_mainCommandBuffer, &presentDep);

  VK_CHECK(vkEndCommandBuffer(_mainCommandBuffer));

  // Submit to the graphics queue
  VkPipelineStageFlags waitStage = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;

  VkSubmitInfo submitInfo{
      .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
      // Wait on swapchain semaphore before executing
      .waitSemaphoreCount = 1,
      .pWaitSemaphores = &_swapchainSemaphore,
      .pWaitDstStageMask = &waitStage,
      // The command buffer to execute
      .commandBufferCount = 1,
      .pCommandBuffers = &_mainCommandBuffer,
      // Signal render semaphore when done
      .signalSemaphoreCount = 1,
      .pSignalSemaphores = &_renderSemaphores[swapchainImageIndex],
  };

  // Submit and signal the render fence when GPU finishes
  VK_CHECK(vkQueueSubmit(_graphicsQueue, 1, &submitInfo, _renderFence));

  // Present the image — waits on this image's render semaphore
  VkPresentInfoKHR presentInfo{
      .sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR,
      .waitSemaphoreCount = 1,
      .pWaitSemaphores = &_renderSemaphores[swapchainImageIndex],
      .swapchainCount = 1,
      .pSwapchains = &_swapchain,
      .pImageIndices = &swapchainImageIndex,
  };

  VK_CHECK(vkQueuePresentKHR(_graphicsQueue, &presentInfo));

  _frameNumber++;
}

void VulkanEngine::run() {
  SDL_Event e;
  bool bQuit = false;

  // main loop
  while (!bQuit) {
    // Handle events on queue
    while (SDL_PollEvent(&e)) {
      // close the window when user alt-f4s or clicks the X button
      if (e.type == SDL_EVENT_QUIT)
        bQuit = true;

      if (e.type == SDL_EVENT_WINDOW_MINIMIZED) {
        stop_rendering = true;
      }
      if (e.type == SDL_EVENT_WINDOW_RESTORED) {
        stop_rendering = false;
      }
    }

    draw();
  }
}

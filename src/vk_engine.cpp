

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

VulkanEngine* loadedEngine = nullptr;

VulkanEngine& VulkanEngine::Get() {
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

    SDL_WindowFlags window_flags = (SDL_WindowFlags)(SDL_WINDOW_VULKAN | SDL_WINDOW_RESIZABLE);

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
    init_imgui();
    _terrain.init(_device, _allocator, _graphicsQueueFamily);

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
        throw std::runtime_error(
            fmt::format("Failed to create Vulkan surface: {}", SDL_GetError()));
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

void VulkanEngine::init_swapchain() { create_swapchain(_windowExtent.width, _windowExtent.height); }

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

    // create persistent draw image for accumulation effects (e.g. motion trails)
    VkImageCreateInfo drawImageInfo{
        .sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
        .imageType = VK_IMAGE_TYPE_2D,
        .format = _swapchainImageFormat,
        .extent = {_swapchainExtent.width, _swapchainExtent.height, 1},
        .mipLevels = 1,
        .arrayLayers = 1,
        .samples = VK_SAMPLE_COUNT_1_BIT,
        .tiling = VK_IMAGE_TILING_OPTIMAL,
        .usage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT,
    };

    VmaAllocationCreateInfo drawImageAllocInfo{
        .usage = VMA_MEMORY_USAGE_GPU_ONLY,
    };

    VK_CHECK(vmaCreateImage(_allocator, &drawImageInfo, &drawImageAllocInfo, &_drawImage,
                            &_drawImageAllocation, nullptr));

    VkImageViewCreateInfo drawImageViewInfo{
        .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
        .image = _drawImage,
        .viewType = VK_IMAGE_VIEW_TYPE_2D,
        .format = _swapchainImageFormat,
        .subresourceRange =
            {
                .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
                .levelCount = 1,
                .layerCount = 1,
            },
    };

    VK_CHECK(vkCreateImageView(_device, &drawImageViewInfo, nullptr, &_drawImageView));
    _drawImageReady = false;

    // create depth image matching draw image dimensions
    VkImageCreateInfo depthImageInfo{
        .sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
        .imageType = VK_IMAGE_TYPE_2D,
        .format = _depthFormat,
        .extent = {_swapchainExtent.width, _swapchainExtent.height, 1},
        .mipLevels = 1,
        .arrayLayers = 1,
        .samples = VK_SAMPLE_COUNT_1_BIT,
        .tiling = VK_IMAGE_TILING_OPTIMAL,
        .usage = VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT,
    };

    VK_CHECK(vmaCreateImage(_allocator, &depthImageInfo, &drawImageAllocInfo, &_depthImage,
                            &_depthImageAllocation, nullptr));

    VkImageViewCreateInfo depthImageViewInfo{
        .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
        .image = _depthImage,
        .viewType = VK_IMAGE_VIEW_TYPE_2D,
        .format = _depthFormat,
        .subresourceRange =
            {
                .aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT,
                .levelCount = 1,
                .layerCount = 1,
            },
    };

    VK_CHECK(vkCreateImageView(_device, &depthImageViewInfo, nullptr, &_depthImageView));
}

void VulkanEngine::destroy_swapchain() {
    vkDestroyImageView(_device, _depthImageView, nullptr);
    vmaDestroyImage(_allocator, _depthImage, _depthImageAllocation);

    vkDestroyImageView(_device, _drawImageView, nullptr);
    vmaDestroyImage(_allocator, _drawImage, _drawImageAllocation);
    _drawImageReady = false;

    vkDestroySwapchainKHR(_device, _swapchain, nullptr);

    for (auto& imageView : _swapchainImageViews) {
        vkDestroyImageView(_device, imageView, nullptr);
    }
}

void VulkanEngine::resize_swapchain() {
    vkDeviceWaitIdle(_device);

    // destroy old per-image semaphores
    for (auto& sem : _renderSemaphores) {
        vkDestroySemaphore(_device, sem, nullptr);
    }

    destroy_swapchain();

    // query current window size
    int w, h;
    SDL_GetWindowSize(_window, &w, &h);
    _windowExtent.width = w;
    _windowExtent.height = h;

    create_swapchain(_windowExtent.width, _windowExtent.height);

    // recreate per-image semaphores (image count may have changed)
    VkSemaphoreCreateInfo semaphoreCreateInfo{
        .sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO,
    };
    _renderSemaphores.resize(_swapchainImages.size());
    for (auto& sem : _renderSemaphores) {
        VK_CHECK(vkCreateSemaphore(_device, &semaphoreCreateInfo, nullptr, &sem));
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
    VK_CHECK(vmaCreateBuffer(_allocator, &bufferInfo, &allocInfo, &buffer.buffer,
                             &buffer.allocation, nullptr));
    return buffer;
}

void VulkanEngine::init_meshes() {
    // 8 corners of a unit cube
    Vertex vertices[] = {
        {{-0.5f, -0.5f, -0.5f}}, // 0: left  bottom back
        {{0.5f, -0.5f, -0.5f}},  // 1: right bottom back
        {{0.5f, 0.5f, -0.5f}},   // 2: right top    back
        {{-0.5f, 0.5f, -0.5f}},  // 3: left  top    back
        {{-0.5f, -0.5f, 0.5f}},  // 4: left  bottom front
        {{0.5f, -0.5f, 0.5f}},   // 5: right bottom front
        {{0.5f, 0.5f, 0.5f}},    // 6: right top    front
        {{-0.5f, 0.5f, 0.5f}},   // 7: left  top    front
    };

    // 12 edges, 2 indices each = 24 indices
    uint16_t indices[] = {
        // back face edges
        0,
        1,
        1,
        2,
        2,
        3,
        3,
        0,
        // front face edges
        4,
        5,
        5,
        6,
        6,
        7,
        7,
        4,
        // connecting edges (back to front)
        0,
        4,
        1,
        5,
        2,
        6,
        3,
        7,
    };

    // create and upload vertex buffer
    _cubeMesh.vertexBuffer = create_buffer(sizeof(vertices), VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
                                           VMA_MEMORY_USAGE_CPU_TO_GPU);

    void* data;
    vmaMapMemory(_allocator, _cubeMesh.vertexBuffer.allocation, &data);
    memcpy(data, vertices, sizeof(vertices));
    vmaUnmapMemory(_allocator, _cubeMesh.vertexBuffer.allocation);

    // create and upload index buffer
    _cubeMesh.indexBuffer = create_buffer(sizeof(indices), VK_BUFFER_USAGE_INDEX_BUFFER_BIT,
                                          VMA_MEMORY_USAGE_CPU_TO_GPU);

    vmaMapMemory(_allocator, _cubeMesh.indexBuffer.allocation, &data);
    memcpy(data, indices, sizeof(indices));
    vmaUnmapMemory(_allocator, _cubeMesh.indexBuffer.allocation);

    _cubeMesh.indexCount = sizeof(indices) / sizeof(indices[0]);

    fmt::print("Cube mesh created — {} vertices, {} indices\n",
               sizeof(vertices) / sizeof(vertices[0]), _cubeMesh.indexCount);
}

void Pipelines::cleanup(VkDevice device) {
    vkDestroyPipeline(device, terrain, nullptr);
    vkDestroyPipeline(device, fade, nullptr);
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
    builder.set_vertex_input(
        sizeof(Vertex),
        {
            {.location = 0, .binding = 0, .format = VK_FORMAT_R32G32B32_SFLOAT, .offset = 0},
        });
    builder.set_multisampling_none();
    builder.disable_blending();
    builder.enable_depthtest();
    builder.set_color_attachment_format(_swapchainImageFormat);
    builder.set_depth_format(_depthFormat);

    _pipelines.triangle = builder.build(_device);

    vkDestroyShaderModule(_device, vertShader, nullptr);
    vkDestroyShaderModule(_device, fragShader, nullptr);

    // fade pipeline — fullscreen triangle with alpha blending for motion trails
    VkShaderModule fadeVert;
    if (!load_shader_module((basePath + "shaders/fade.vert.spv").c_str(), _device, &fadeVert)) {
        throw std::runtime_error("Failed to load fade vertex shader");
    }
    VkShaderModule fadeFrag;
    if (!load_shader_module((basePath + "shaders/fade.frag.spv").c_str(), _device, &fadeFrag)) {
        throw std::runtime_error("Failed to load fade fragment shader");
    }

    PipelineBuilder fadeBuilder;
    fadeBuilder._pipelineLayout = _pipelines.triangleLayout;
    fadeBuilder.set_shaders(fadeVert, fadeFrag);
    fadeBuilder.set_input_topology(VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST);
    fadeBuilder.set_polygon_mode(VK_POLYGON_MODE_FILL);
    fadeBuilder.set_cull_mode(VK_CULL_MODE_NONE, VK_FRONT_FACE_CLOCKWISE);
    fadeBuilder.set_multisampling_none();
    fadeBuilder.enable_blending_alpha();
    fadeBuilder.disable_depthtest();
    fadeBuilder.set_color_attachment_format(_swapchainImageFormat);
    fadeBuilder.set_depth_format(_depthFormat);

    _pipelines.fade = fadeBuilder.build(_device);

    vkDestroyShaderModule(_device, fadeVert, nullptr);
    vkDestroyShaderModule(_device, fadeFrag, nullptr);

    // terrain pipeline — filled triangles with depth test and diffuse lighting
    VkShaderModule terrainVert;
    if (!load_shader_module((basePath + "shaders/terrain.vert.spv").c_str(), _device, &terrainVert)) {
        throw std::runtime_error("Failed to load terrain vertex shader");
    }
    VkShaderModule terrainFrag;
    if (!load_shader_module((basePath + "shaders/terrain.frag.spv").c_str(), _device, &terrainFrag)) {
        throw std::runtime_error("Failed to load terrain fragment shader");
    }

    PipelineBuilder terrainBuilder;
    terrainBuilder._pipelineLayout = _pipelines.triangleLayout;
    terrainBuilder.set_shaders(terrainVert, terrainFrag);
    terrainBuilder.set_input_topology(VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST);
    terrainBuilder.set_polygon_mode(VK_POLYGON_MODE_FILL);
    terrainBuilder.set_cull_mode(VK_CULL_MODE_BACK_BIT, VK_FRONT_FACE_COUNTER_CLOCKWISE);
    terrainBuilder.set_vertex_input(sizeof(TerrainVertex), {
        {.location = 0, .binding = 0, .format = VK_FORMAT_R32G32B32A32_SFLOAT, .offset = 0},
        {.location = 1, .binding = 0, .format = VK_FORMAT_R32G32B32A32_SFLOAT, .offset = 16},
    });
    terrainBuilder.set_multisampling_none();
    terrainBuilder.disable_blending();
    terrainBuilder.enable_depthtest();
    terrainBuilder.set_color_attachment_format(_swapchainImageFormat);
    terrainBuilder.set_depth_format(_depthFormat);

    _pipelines.terrain = terrainBuilder.build(_device);

    vkDestroyShaderModule(_device, terrainVert, nullptr);
    vkDestroyShaderModule(_device, terrainFrag, nullptr);

    fmt::print("Pipelines created\n");
}

void VulkanEngine::init_imgui() {
    _debugGui.init(_instance, _chosenGPU, _device, _graphicsQueueFamily,
                   _graphicsQueue, _swapchainImageFormat, _depthFormat,
                   (uint32_t)_swapchainImages.size(), _window);
}

void VulkanEngine::cleanup() {
    if (_isInitialized) {
        vkDeviceWaitIdle(_device);

        _debugGui.cleanup();
        _terrain.cleanup(_device, _allocator);
        _pipelines.cleanup(_device);

        vkDestroyFence(_device, _renderFence, nullptr);
        vkDestroySemaphore(_device, _swapchainSemaphore, nullptr);
        for (auto& sem : _renderSemaphores) {
            vkDestroySemaphore(_device, sem, nullptr);
        }
        vkDestroyCommandPool(_device, _commandPool, nullptr);
        destroy_swapchain();

        vmaDestroyBuffer(_allocator, _cubeMesh.vertexBuffer.buffer,
                         _cubeMesh.vertexBuffer.allocation);
        vmaDestroyBuffer(_allocator, _cubeMesh.indexBuffer.buffer,
                         _cubeMesh.indexBuffer.allocation);
        vmaDestroyAllocator(_allocator);
        vkDestroySurfaceKHR(_instance, _surface, nullptr);
        vkDestroyDevice(_device, nullptr);
        vkb::destroy_debug_utils_messenger(_instance, _debug_messenger);
        vkDestroyInstance(_instance, nullptr);

        SDL_DestroyWindow(_window);
    }
}

void VulkanEngine::record_scene(VkCommandBuffer cmd) {
    // First frame after creation: UNDEFINED + CLEAR (contents are garbage)
    // Subsequent frames: TRANSFER_SRC + LOAD (preserve previous frame for trails)
    transition_image(
        cmd, _drawImage,
        _drawImageReady ? VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL : VK_IMAGE_LAYOUT_UNDEFINED,
        VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, VK_PIPELINE_STAGE_2_TRANSFER_BIT,
        VK_ACCESS_2_TRANSFER_READ_BIT, VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
        VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT | VK_ACCESS_2_COLOR_ATTACHMENT_READ_BIT);

    transition_image(cmd, _depthImage,
        VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL,
        VK_PIPELINE_STAGE_2_NONE, VK_ACCESS_2_NONE,
        VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT, VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT);

    VkRenderingAttachmentInfo colorAttachment{
        .sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO,
        .imageView = _drawImageView,
        .imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
        .loadOp = _drawImageReady ? VK_ATTACHMENT_LOAD_OP_LOAD : VK_ATTACHMENT_LOAD_OP_CLEAR,
        .storeOp = VK_ATTACHMENT_STORE_OP_STORE,
        .clearValue = {.color = {{0.0f, 0.0f, 0.0f, 1.0f}}},
    };

    VkRenderingAttachmentInfo depthAttachment{
        .sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO,
        .imageView = _depthImageView,
        .imageLayout = VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL,
        .loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR,
        .storeOp = VK_ATTACHMENT_STORE_OP_DONT_CARE,
        .clearValue = {.depthStencil = {1.0f, 0}},
    };

    VkRenderingInfo renderInfo{
        .sType = VK_STRUCTURE_TYPE_RENDERING_INFO,
        .renderArea = {.extent = _swapchainExtent},
        .layerCount = 1,
        .colorAttachmentCount = 1,
        .pColorAttachments = &colorAttachment,
        .pDepthAttachment = &depthAttachment,
    };

    vkCmdBeginRendering(cmd, &renderInfo);

    VkViewport viewport{
        .x = 0,
        .y = 0,
        .width = (float)_swapchainExtent.width,
        .height = (float)_swapchainExtent.height,
        .minDepth = 0.f,
        .maxDepth = 1.f,
    };
    vkCmdSetViewport(cmd, 0, 1, &viewport);

    VkRect2D scissor{
        .offset = {0, 0},
        .extent = _swapchainExtent,
    };
    vkCmdSetScissor(cmd, 0, 1, &scissor);

    // 1) Fade: draw fullscreen triangle with low-alpha black to dim previous contents
    if (_drawImageReady) {
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, _pipelines.fade);

        PushConstants fadePush{
            .mvp = glm::mat4(1.0f),
            .color = {0.0f, 0.0f, 0.0f, 0.1f},
        };
        vkCmdPushConstants(cmd, _pipelines.triangleLayout,
                           VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0,
                           sizeof(fadePush), &fadePush);

        vkCmdDraw(cmd, 3, 1, 0, 0);
    }

    // 2) Draw the cube
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, _pipelines.triangle);

    float angle = _frameNumber / 120.f;
    _camera.orbitAround(glm::vec3{0.f}, 3.f, angle);

    float aspectRatio = (float)_swapchainExtent.width / (float)_swapchainExtent.height;
    glm::mat4 mvp = _camera.projectionMatrix(aspectRatio) * _camera.viewMatrix();

    PushConstants push{
        .mvp = mvp,
        .color = {1.0f, 1.0f, 1.0f, 1.0f},
    };
    vkCmdPushConstants(cmd, _pipelines.triangleLayout,
                       VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(push),
                       &push);

    VkDeviceSize offset = 0;
    vkCmdBindVertexBuffers(cmd, 0, 1, &_cubeMesh.vertexBuffer.buffer, &offset);
    vkCmdBindIndexBuffer(cmd, _cubeMesh.indexBuffer.buffer, 0, VK_INDEX_TYPE_UINT16);

    vkCmdDrawIndexed(cmd, _cubeMesh.indexCount, 1, 0, 0, 0);

    // draw terrain
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, _pipelines.terrain);

    PushConstants terrainPush{
        .mvp = mvp,
        .color = {0.6f, 0.8f, 0.4f, 1.0f},
    };
    vkCmdPushConstants(cmd, _pipelines.triangleLayout,
                       VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0,
                       sizeof(terrainPush), &terrainPush);
    _terrain.draw(cmd);

    _debugGui.begin_frame();
    _debugGui.render(cmd, _terrainParams);

    vkCmdEndRendering(cmd);

    _drawImageReady = true;
}

void VulkanEngine::copy_to_swapchain(VkCommandBuffer cmd, uint32_t swapchainImageIndex) {
    transition_image(cmd, _drawImage, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                     VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                     VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
                     VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT, VK_PIPELINE_STAGE_2_TRANSFER_BIT,
                     VK_ACCESS_2_TRANSFER_READ_BIT);

    transition_image(cmd, _swapchainImages[swapchainImageIndex], VK_IMAGE_LAYOUT_UNDEFINED,
                     VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_PIPELINE_STAGE_2_NONE,
                     VK_ACCESS_2_NONE, VK_PIPELINE_STAGE_2_TRANSFER_BIT,
                     VK_ACCESS_2_TRANSFER_WRITE_BIT);

    VkImageCopy2 copyRegion{
        .sType = VK_STRUCTURE_TYPE_IMAGE_COPY_2,
        .srcSubresource = {.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT, .layerCount = 1},
        .dstSubresource = {.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT, .layerCount = 1},
        .extent = {_swapchainExtent.width, _swapchainExtent.height, 1},
    };

    VkCopyImageInfo2 copyInfo{
        .sType = VK_STRUCTURE_TYPE_COPY_IMAGE_INFO_2,
        .srcImage = _drawImage,
        .srcImageLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
        .dstImage = _swapchainImages[swapchainImageIndex],
        .dstImageLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
        .regionCount = 1,
        .pRegions = &copyRegion,
    };

    vkCmdCopyImage2(cmd, &copyInfo);

    transition_image(cmd, _swapchainImages[swapchainImageIndex],
                     VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
                     VK_PIPELINE_STAGE_2_TRANSFER_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT,
                     VK_PIPELINE_STAGE_2_NONE, VK_ACCESS_2_NONE);
}

void VulkanEngine::draw() {
    VK_CHECK(vkWaitForFences(_device, 1, &_renderFence, true, 1000000000));
    VK_CHECK(vkResetFences(_device, 1, &_renderFence));

    uint32_t swapchainImageIndex;
    VkResult acquireResult = vkAcquireNextImageKHR(
        _device, _swapchain, 1000000000, _swapchainSemaphore, nullptr, &swapchainImageIndex);
    // if swap chain is out of date (e.g. window has resized) we recreate full swap chain
    if (acquireResult == VK_ERROR_OUT_OF_DATE_KHR) {
        resize_swapchain();
        return;
    }
    if (acquireResult != VK_SUCCESS && acquireResult != VK_SUBOPTIMAL_KHR) {
        VK_CHECK(acquireResult);
    }

    VK_CHECK(vkResetCommandBuffer(_mainCommandBuffer, 0));

    VkCommandBufferBeginInfo cmdBeginInfo{
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
        .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
    };
    VK_CHECK(vkBeginCommandBuffer(_mainCommandBuffer, &cmdBeginInfo));

    _terrain.dispatch(_mainCommandBuffer, _terrainParams);
    record_scene(_mainCommandBuffer);
    copy_to_swapchain(_mainCommandBuffer, swapchainImageIndex);

    VK_CHECK(vkEndCommandBuffer(_mainCommandBuffer));

    // Submit and present
    VkPipelineStageFlags waitStage = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;

    VkSubmitInfo submitInfo{
        .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
        .waitSemaphoreCount = 1,
        .pWaitSemaphores = &_swapchainSemaphore,
        .pWaitDstStageMask = &waitStage,
        .commandBufferCount = 1,
        .pCommandBuffers = &_mainCommandBuffer,
        .signalSemaphoreCount = 1,
        .pSignalSemaphores = &_renderSemaphores[swapchainImageIndex],
    };

    VK_CHECK(vkQueueSubmit(_graphicsQueue, 1, &submitInfo, _renderFence));

    VkPresentInfoKHR presentInfo{
        .sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR,
        .waitSemaphoreCount = 1,
        .pWaitSemaphores = &_renderSemaphores[swapchainImageIndex],
        .swapchainCount = 1,
        .pSwapchains = &_swapchain,
        .pImageIndices = &swapchainImageIndex,
    };

    VkResult presentResult = vkQueuePresentKHR(_graphicsQueue, &presentInfo);
    if (presentResult == VK_ERROR_OUT_OF_DATE_KHR || presentResult == VK_SUBOPTIMAL_KHR) {
        resize_swapchain();
    } else {
        VK_CHECK(presentResult);
    }

    _frameNumber++;
}

void VulkanEngine::run() {
    SDL_Event e;
    bool bQuit = false;

    // main loop
    while (!bQuit) {
        // Handle events on queue
        while (SDL_PollEvent(&e)) {
            _debugGui.process_event(e);

            if (e.type == SDL_EVENT_QUIT)
                bQuit = true;

            if (e.type == SDL_EVENT_WINDOW_MINIMIZED) {
                stop_rendering = true;
            }
            if (e.type == SDL_EVENT_WINDOW_RESTORED) {
                stop_rendering = false;
            }
            if (e.type == SDL_EVENT_WINDOW_RESIZED) {
                resize_swapchain();
            }
        }

        if (stop_rendering) {
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
            continue;
        }

        draw();
    }
}

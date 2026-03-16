#include "gui.h"

#ifdef ENABLE_DEBUG_GUI

#include <imgui.h>
#include <backends/imgui_impl_sdl3.h>
#include <backends/imgui_impl_vulkan.h>

#include <SDL3/SDL.h>

#include "terrain.h"
#include "vk_initializers.h"

void DebugGui::init(VkInstance instance, VkPhysicalDevice gpu, VkDevice device,
                    uint32_t queueFamily, VkQueue queue,
                    VkFormat colorFormat, VkFormat depthFormat, uint32_t imageCount,
                    SDL_Window* window) {
    ImGui::CreateContext();
    ImGui_ImplSDL3_InitForVulkan(window);

    ImGui_ImplVulkan_InitInfo initInfo{};
    initInfo.ApiVersion = VK_API_VERSION_1_3;
    initInfo.Instance = instance;
    initInfo.PhysicalDevice = gpu;
    initInfo.Device = device;
    initInfo.QueueFamily = queueFamily;
    initInfo.Queue = queue;
    initInfo.DescriptorPoolSize = 8;
    initInfo.MinImageCount = 2;
    initInfo.ImageCount = imageCount;
    initInfo.UseDynamicRendering = true;

    VkPipelineRenderingCreateInfo renderingInfo{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO,
        .colorAttachmentCount = 1,
        .pColorAttachmentFormats = &colorFormat,
        .depthAttachmentFormat = depthFormat,
    };
    initInfo.PipelineInfoMain.PipelineRenderingCreateInfo = renderingInfo;

    ImGui_ImplVulkan_Init(&initInfo);
}

void DebugGui::cleanup() {
    ImGui_ImplVulkan_Shutdown();
    ImGui_ImplSDL3_Shutdown();
    ImGui::DestroyContext();
}

void DebugGui::process_event(SDL_Event& e) {
    ImGui_ImplSDL3_ProcessEvent(&e);
}

void DebugGui::begin_frame() {
    ImGui_ImplVulkan_NewFrame();
    ImGui_ImplSDL3_NewFrame();
    ImGui::NewFrame();
}

void DebugGui::render(VkCommandBuffer cmd, TerrainParams& params) {
    ImGui::Begin("Terrain");
    ImGui::SliderFloat("Alpha", &params.alpha, 0.0f, 6.28f);
    ImGui::SliderFloat("Iso Level", &params.isoLevel, -1.0f, 1.0f);
    ImGui::SliderFloat("Frequency", &params.frequency, 0.1f, 10.0f);
    ImGui::SliderFloat("Amplitude", &params.amplitude, 0.1f, 5.0f);
    ImGui::SliderFloat("Voxel Scale", &params.voxelScale, 0.01f, 0.2f);
    ImGui::End();

    ImGui::Render();
    ImGui_ImplVulkan_RenderDrawData(ImGui::GetDrawData(), cmd);
}

#else

void DebugGui::init(VkInstance, VkPhysicalDevice, VkDevice,
                    uint32_t, VkQueue, VkFormat, VkFormat, uint32_t, SDL_Window*) {}
void DebugGui::cleanup() {}
void DebugGui::process_event(SDL_Event&) {}
void DebugGui::begin_frame() {}
void DebugGui::render(VkCommandBuffer, TerrainParams&) {}

#endif
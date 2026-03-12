cmake_minimum_required(VERSION 3.14)
set(CMAKE_POLICY_VERSION_MINIMUM 3.14)
project(Vulkan)
include(FetchContent)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# include CPM (package manager) and download if necessary
set(CPM_DOWNLOAD_VERSION 0.42.1)
set(CPM_DOWNLOAD_LOCATION
    "${CMAKE_BINARY_DIR}/cmake/CPM_${CPM_DOWNLOAD_VERSION}.cmake")

if(NOT (EXISTS ${CPM_DOWNLOAD_LOCATION}))
  message(STATUS "Downloading CPM.cmake")
  file(
    DOWNLOAD
    https://github.com/TheLartians/CPM.cmake/releases/download/v${CPM_DOWNLOAD_VERSION}/CPM.cmake
    ${CPM_DOWNLOAD_LOCATION})
endif()

include(${CPM_DOWNLOAD_LOCATION})

# fetch vulkan headers + utilities at a matching pinned version (before other
# vulkan-dependent packages so CPM satisfies their find_package(Vulkan) calls)
set(VULKAN_PIN "v1.4.304")
cpmaddpackage("gh:KhronosGroup/Vulkan-Headers#${VULKAN_PIN}")
cpmaddpackage("gh:KhronosGroup/Vulkan-Utility-Libraries#${VULKAN_PIN}")

# find vulkan loader (ABI-stable, safe as system dependency)
find_library(VULKAN_LOADER NAMES vulkan vulkan-1
  HINTS $ENV{VULKAN_SDK}/lib)
if(NOT VULKAN_LOADER)
  message(FATAL_ERROR "Vulkan loader not found")
endif()

# get dependencies via CPMs short syntax
cpmaddpackage("gh:libsdl-org/SDL#release-3.4.2")
cpmaddpackage("gh:g-truc/glm#1.0.3")
cpmaddpackage("gh:tinyobjloader/tinyobjloader@1.0.6")
cpmaddpackage("gh:charles-lunarg/vk-bootstrap@1.4.344")
cpmaddpackage("gh:GPUOpen-LibrariesAndSDKs/VulkanMemoryAllocator@3.3.0")
cpmaddpackage("gh:fmtlib/fmt@#12.1.0")

# link imgui manually because of custom backends
cpmaddpackage(URI gh:ocornut/imgui@1.92.6 DOWNLOAD_ONLY)
add_library(
  imgui STATIC
  ${imgui_SOURCE_DIR}/imgui.cpp
  ${imgui_SOURCE_DIR}/imgui_draw.cpp
  ${imgui_SOURCE_DIR}/imgui_tables.cpp
  ${imgui_SOURCE_DIR}/imgui_widgets.cpp
  ${imgui_SOURCE_DIR}/backends/imgui_impl_sdl3.cpp
  ${imgui_SOURCE_DIR}/backends/imgui_impl_opengl3.cpp)
target_include_directories(imgui PUBLIC ${imgui_SOURCE_DIR})
target_link_libraries(imgui PUBLIC SDL3::SDL3)

# manual download of single stb header
file(
  DOWNLOAD
  https://raw.githubusercontent.com/nothings/stb/refs/heads/master/stb_image.h
  ${CMAKE_BINARY_DIR}/stb/stb_image.h)

add_executable(
  ${CMAKE_PROJECT_NAME}
  ${PROJECT_SOURCE_DIR}/src/main.cpp ${PROJECT_SOURCE_DIR}/src/vk_engine.cpp
  ${PROJECT_SOURCE_DIR}/src/vk_initializers.cpp)
target_include_directories(${CMAKE_PROJECT_NAME}
                           PRIVATE ${CMAKE_BINARY_DIR}/stb)
target_link_libraries(
  ${CMAKE_PROJECT_NAME}
  Vulkan::Headers
  Vulkan::UtilityHeaders
  ${VULKAN_LOADER}
  SDL3::SDL3
  glm
  imgui
  GPUOpen::VulkanMemoryAllocator
  vk-bootstrap::vk-bootstrap
  fmt)

cmake_minimum_required(VERSION 3.14)
set(CMAKE_POLICY_VERSION_MINIMUM 3.14)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

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
set(VULKAN_VERSION "1.4.304")
cpmaddpackage("gh:KhronosGroup/Vulkan-Headers#v${VULKAN_VERSION}")
cpmaddpackage("gh:KhronosGroup/Vulkan-Utility-Libraries#v${VULKAN_VERSION}")

# find vulkan loader (ABI-stable, safe as system dependency)
find_library(
  VULKAN_LOADER
  NAMES vulkan vulkan-1
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

# shader compiler (builds glslang from source)
set(ENABLE_GLSLANG_BINARIES ON CACHE BOOL "" FORCE)
set(ENABLE_SPVREMAPPER OFF CACHE BOOL "" FORCE)
set(ENABLE_CTEST OFF CACHE BOOL "" FORCE)
set(ENABLE_OPT OFF CACHE BOOL "" FORCE)
cpmaddpackage("gh:KhronosGroup/glslang#vulkan-sdk-${VULKAN_VERSION}.0")

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

# auto-discover all shader files and compile to SPIR-V
file(
  GLOB_RECURSE SHADER_FILES
  CONFIGURE_DEPENDS
  ${CMAKE_SOURCE_DIR}/shaders/*.vert
  ${CMAKE_SOURCE_DIR}/shaders/*.frag
  ${CMAKE_SOURCE_DIR}/shaders/*.comp)

foreach(SHADER ${SHADER_FILES})
  # mirror directory structure: shaders/subdir/foo.vert → build/shaders/subdir/foo.vert.spv
  file(RELATIVE_PATH REL_PATH ${CMAKE_SOURCE_DIR}/shaders ${SHADER})
  set(SPV_OUTPUT ${CMAKE_BINARY_DIR}/shaders/${REL_PATH}.spv)
  get_filename_component(SPV_DIR ${SPV_OUTPUT} DIRECTORY)
  file(MAKE_DIRECTORY ${SPV_DIR})

  add_custom_command(
    OUTPUT ${SPV_OUTPUT}
    COMMAND glslang-standalone -V ${SHADER} -o ${SPV_OUTPUT}
    DEPENDS ${SHADER}
    COMMENT "Compiling shader ${REL_PATH}")
  list(APPEND SPV_FILES ${SPV_OUTPUT})
endforeach()

add_custom_target(shaders DEPENDS ${SPV_FILES})

add_executable(
  ${CMAKE_PROJECT_NAME}
  ${PROJECT_SOURCE_DIR}/src/main.cpp ${PROJECT_SOURCE_DIR}/src/vk_engine.cpp
  ${PROJECT_SOURCE_DIR}/src/vk_initializers.cpp
  ${PROJECT_SOURCE_DIR}/src/vk_pipelines.cpp
  ${PROJECT_SOURCE_DIR}/src/vk_mem_alloc_impl.cpp
  ${PROJECT_SOURCE_DIR}/src/camera.cpp)
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
add_dependencies(${CMAKE_PROJECT_NAME} shaders)

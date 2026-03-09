

#include "vk_engine.h"

#include <SDL3/SDL.h>
#include <SDL3/SDL_vulkan.h>

#include "vk_initializers.h"
#include "vk_types.h"

void VulkanEngine::init() {
  // We initialize SDL and create a window with it.
  if (!SDL_Init(SDL_INIT_VIDEO)) {
    SDL_Log("SDL_Init failed: %s", SDL_GetError());
    return;
  }

  SDL_WindowFlags window_flags = (SDL_WindowFlags)(SDL_WINDOW_VULKAN);

  _window = SDL_CreateWindow("Vulkan Engine", _windowExtent.width,
                             _windowExtent.height, window_flags);
  if (!_window) {
    SDL_Log("SDL_CreateWindow failed: %s", SDL_GetError());
    return;
  }

  // everything went fine
  _isInitialized = true;
}
void VulkanEngine::cleanup() {
  if (_isInitialized) {

    SDL_DestroyWindow(_window);
  }
}

void VulkanEngine::draw() {
  // nothing yet
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
    }

    draw();
  }
}

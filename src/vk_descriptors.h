#pragma once

#include <span>
#include <vector>
#include <vulkan/vulkan.h>

class DescriptorLayoutBuilder {
public:
    DescriptorLayoutBuilder& add_binding(uint32_t binding, VkDescriptorType type,
                                         VkShaderStageFlags stages, uint32_t count = 1);
    VkDescriptorSetLayout build(VkDevice device);

private:
    std::vector<VkDescriptorSetLayoutBinding> _bindings;
};

class DescriptorAllocator {
public:
    void init(VkDevice device, uint32_t maxSets, std::span<VkDescriptorPoolSize> poolSizes);
    VkDescriptorSet allocate(VkDescriptorSetLayout layout);
    void destroy();

private:
    VkDevice _device{VK_NULL_HANDLE};
    VkDescriptorPool _pool{VK_NULL_HANDLE};
};

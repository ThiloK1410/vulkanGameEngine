#include "vk_descriptors.h"
#include "vk_initializers.h"

DescriptorLayoutBuilder& DescriptorLayoutBuilder::add_binding(uint32_t binding,
                                                               VkDescriptorType type,
                                                               VkShaderStageFlags stages,
                                                               uint32_t count) {
    _bindings.push_back({
        .binding = binding,
        .descriptorType = type,
        .descriptorCount = count,
        .stageFlags = stages,
    });
    return *this;
}

VkDescriptorSetLayout DescriptorLayoutBuilder::build(VkDevice device) {
    VkDescriptorSetLayoutCreateInfo info{
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
        .bindingCount = (uint32_t)_bindings.size(),
        .pBindings = _bindings.data(),
    };

    VkDescriptorSetLayout layout;
    VK_CHECK(vkCreateDescriptorSetLayout(device, &info, nullptr, &layout));
    return layout;
}

void DescriptorAllocator::init(VkDevice device, uint32_t maxSets,
                                std::span<VkDescriptorPoolSize> poolSizes) {
    _device = device;

    VkDescriptorPoolCreateInfo poolInfo{
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
        .maxSets = maxSets,
        .poolSizeCount = (uint32_t)poolSizes.size(),
        .pPoolSizes = poolSizes.data(),
    };

    VK_CHECK(vkCreateDescriptorPool(_device, &poolInfo, nullptr, &_pool));
}

VkDescriptorSet DescriptorAllocator::allocate(VkDescriptorSetLayout layout) {
    VkDescriptorSetAllocateInfo allocInfo{
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
        .descriptorPool = _pool,
        .descriptorSetCount = 1,
        .pSetLayouts = &layout,
    };

    VkDescriptorSet set;
    VK_CHECK(vkAllocateDescriptorSets(_device, &allocInfo, &set));
    return set;
}

void DescriptorAllocator::destroy() {
    if (_pool != VK_NULL_HANDLE) {
        vkDestroyDescriptorPool(_device, _pool, nullptr);
        _pool = VK_NULL_HANDLE;
    }
}
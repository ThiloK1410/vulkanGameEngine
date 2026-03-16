#pragma once

#include <vulkan/vulkan.h>
#include <vk_mem_alloc.h>

#include "vk_descriptors.h"
#include "vk_types.h"

struct TerrainParams {
    uint32_t gridSize = 32;
    float voxelScale = 1.0f / 32.0f;
    float frequency = 4.0f;
    float amplitude = 1.0f;
    float alpha = 0.0f;
    float isoLevel = 0.0f;
};

class Terrain {
public:
    void init(VkDevice device, VmaAllocator allocator, uint32_t queueFamily);
    void cleanup(VkDevice device, VmaAllocator allocator);

    // record compute commands into cmd
    void dispatch(VkCommandBuffer cmd, const TerrainParams& params);

    // bind buffers and issue indirect draw
    void draw(VkCommandBuffer cmd);

private:
    // compute pipelines
    VkPipeline _densityPipeline{VK_NULL_HANDLE};
    VkPipeline _mcPipeline{VK_NULL_HANDLE};
    VkPipelineLayout _computeLayout{VK_NULL_HANDLE};

    // descriptors
    VkDescriptorSetLayout _descriptorLayout{VK_NULL_HANDLE};
    VkDescriptorSet _descriptorSet{VK_NULL_HANDLE};
    DescriptorAllocator _descriptorAllocator;

    // GPU buffers
    AllocatedBuffer _densityBuffer;
    AllocatedBuffer _edgeTableBuffer;
    AllocatedBuffer _triTableBuffer;
    AllocatedBuffer _vertexBuffer;
    AllocatedBuffer _indexBuffer;
    AllocatedBuffer _indirectBuffer;

    uint32_t _gridSize{0};
    uint32_t _maxVertices{0};

    void create_buffers(VkDevice device, VmaAllocator allocator, uint32_t gridSize);
    void destroy_buffers(VkDevice device, VmaAllocator allocator);
    void upload_lookup_tables(VkDevice device, VmaAllocator allocator);
    void update_descriptor_set(VkDevice device);
};
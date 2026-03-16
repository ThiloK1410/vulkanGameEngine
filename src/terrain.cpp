#include "terrain.h"

#include <cstring>

#include <SDL3/SDL.h>
#include <fmt/core.h>

#include "mc_tables.h"
#include "vk_initializers.h"
#include "vk_pipelines.h"

// push constant layout shared by both compute shaders
struct ComputeParams {
    uint32_t gridSize;
    float voxelScale;
    float frequency;
    float amplitude;
    float alpha;
    float isoLevel;
};

// --- buffer management ---

void Terrain::create_buffers(VkDevice device, VmaAllocator allocator, uint32_t gridSize) {
    _gridSize = gridSize;
    uint32_t cells = gridSize - 1;
    _maxVertices = cells * cells * cells * 15; // 5 triangles * 3 verts per cell

    auto make_buffer = [&](VkDeviceSize size, VkBufferUsageFlags usage) -> AllocatedBuffer {
        VkBufferCreateInfo bufInfo{
            .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
            .size = size,
            .usage = usage,
        };
        VmaAllocationCreateInfo allocInfo{.usage = VMA_MEMORY_USAGE_GPU_ONLY};
        AllocatedBuffer buf;
        VK_CHECK(vmaCreateBuffer(allocator, &bufInfo, &allocInfo, &buf.buffer, &buf.allocation, nullptr));
        return buf;
    };

    uint32_t densityCount = gridSize * gridSize * gridSize;

    _densityBuffer = make_buffer(
        densityCount * sizeof(float),
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);

    _vertexBuffer = make_buffer(
        _maxVertices * sizeof(TerrainVertex),
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT);

    _indexBuffer = make_buffer(
        _maxVertices * sizeof(uint32_t),
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_INDEX_BUFFER_BIT);

    // indirect draw command buffer — also used as atomic counter by compute shader
    _indirectBuffer = make_buffer(
        sizeof(VkDrawIndexedIndirectCommand),
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT |
        VK_BUFFER_USAGE_TRANSFER_DST_BIT);
}

void Terrain::destroy_buffers(VkDevice device, VmaAllocator allocator) {
    vmaDestroyBuffer(allocator, _densityBuffer.buffer, _densityBuffer.allocation);
    vmaDestroyBuffer(allocator, _vertexBuffer.buffer, _vertexBuffer.allocation);
    vmaDestroyBuffer(allocator, _indexBuffer.buffer, _indexBuffer.allocation);
    vmaDestroyBuffer(allocator, _indirectBuffer.buffer, _indirectBuffer.allocation);
}

void Terrain::upload_lookup_tables(VkDevice device, VmaAllocator allocator) {
    auto upload = [&](const void* data, VkDeviceSize size, VkBufferUsageFlags usage) -> AllocatedBuffer {
        VkBufferCreateInfo bufInfo{
            .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
            .size = size,
            .usage = usage,
        };
        VmaAllocationCreateInfo allocInfo{.usage = VMA_MEMORY_USAGE_CPU_TO_GPU};
        AllocatedBuffer buf;
        VK_CHECK(vmaCreateBuffer(allocator, &bufInfo, &allocInfo, &buf.buffer, &buf.allocation, nullptr));

        void* mapped;
        vmaMapMemory(allocator, buf.allocation, &mapped);
        memcpy(mapped, data, size);
        vmaUnmapMemory(allocator, buf.allocation);

        return buf;
    };

    _edgeTableBuffer = upload(MC_EDGE_TABLE, sizeof(MC_EDGE_TABLE), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    _triTableBuffer = upload(MC_TRI_TABLE, sizeof(MC_TRI_TABLE), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
}

// --- descriptor set ---

void Terrain::update_descriptor_set(VkDevice device) {
    VkDescriptorBufferInfo bufferInfos[] = {
        {_densityBuffer.buffer,   0, VK_WHOLE_SIZE},  // binding 0
        {_edgeTableBuffer.buffer, 0, VK_WHOLE_SIZE},  // binding 1
        {_triTableBuffer.buffer,  0, VK_WHOLE_SIZE},  // binding 2
        {_vertexBuffer.buffer,    0, VK_WHOLE_SIZE},  // binding 3
        {_indexBuffer.buffer,     0, VK_WHOLE_SIZE},  // binding 4
        {_indirectBuffer.buffer,  0, VK_WHOLE_SIZE},  // binding 5
    };

    VkWriteDescriptorSet writes[6];
    for (uint32_t i = 0; i < 6; i++) {
        writes[i] = {
            .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            .dstSet = _descriptorSet,
            .dstBinding = i,
            .descriptorCount = 1,
            .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            .pBufferInfo = &bufferInfos[i],
        };
    }

    vkUpdateDescriptorSets(device, 6, writes, 0, nullptr);
}

// --- init / cleanup ---

void Terrain::init(VkDevice device, VmaAllocator allocator, uint32_t queueFamily) {
    // descriptor layout: 6 storage buffers, all used by compute
    _descriptorLayout = DescriptorLayoutBuilder()
        .add_binding(0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_COMPUTE_BIT)
        .add_binding(1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_COMPUTE_BIT)
        .add_binding(2, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_COMPUTE_BIT)
        .add_binding(3, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_COMPUTE_BIT)
        .add_binding(4, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_COMPUTE_BIT)
        .add_binding(5, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_COMPUTE_BIT)
        .build(device);

    VkDescriptorPoolSize poolSize{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 6};
    _descriptorAllocator.init(device, 1, {&poolSize, 1});
    _descriptorSet = _descriptorAllocator.allocate(_descriptorLayout);

    // compute pipeline layout: descriptor set + push constants
    VkPushConstantRange pushRange{
        .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
        .offset = 0,
        .size = sizeof(ComputeParams),
    };

    VkPipelineLayoutCreateInfo layoutInfo{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
        .setLayoutCount = 1,
        .pSetLayouts = &_descriptorLayout,
        .pushConstantRangeCount = 1,
        .pPushConstantRanges = &pushRange,
    };
    VK_CHECK(vkCreatePipelineLayout(device, &layoutInfo, nullptr, &_computeLayout));

    // load compute shaders
    std::string basePath = SDL_GetBasePath();

    auto create_compute_pipeline = [&](const char* shaderPath) -> VkPipeline {
        VkShaderModule module;
        if (!load_shader_module((basePath + shaderPath).c_str(), device, &module)) {
            throw std::runtime_error(fmt::format("Failed to load shader: {}", shaderPath));
        }

        VkComputePipelineCreateInfo pipelineInfo{
            .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
            .stage = {
                .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                .stage = VK_SHADER_STAGE_COMPUTE_BIT,
                .module = module,
                .pName = "main",
            },
            .layout = _computeLayout,
        };

        VkPipeline pipeline;
        VK_CHECK(vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &pipeline));
        vkDestroyShaderModule(device, module, nullptr);
        return pipeline;
    };

    _densityPipeline = create_compute_pipeline("shaders/density.comp.spv");
    _mcPipeline = create_compute_pipeline("shaders/marching_cubes.comp.spv");

    // create buffers and upload lookup tables
    upload_lookup_tables(device, allocator);
    create_buffers(device, allocator, 32);
    update_descriptor_set(device);

    fmt::print("Terrain initialized — grid {}³\n", _gridSize);
}

void Terrain::cleanup(VkDevice device, VmaAllocator allocator) {
    destroy_buffers(device, allocator);

    vmaDestroyBuffer(allocator, _edgeTableBuffer.buffer, _edgeTableBuffer.allocation);
    vmaDestroyBuffer(allocator, _triTableBuffer.buffer, _triTableBuffer.allocation);

    _descriptorAllocator.destroy();
    vkDestroyDescriptorSetLayout(device, _descriptorLayout, nullptr);
    vkDestroyPipeline(device, _densityPipeline, nullptr);
    vkDestroyPipeline(device, _mcPipeline, nullptr);
    vkDestroyPipelineLayout(device, _computeLayout, nullptr);
}

// --- per-frame dispatch ---

void Terrain::dispatch(VkCommandBuffer cmd, const TerrainParams& params) {
    ComputeParams push{
        .gridSize = params.gridSize,
        .voxelScale = params.voxelScale,
        .frequency = params.frequency,
        .amplitude = params.amplitude,
        .alpha = params.alpha,
        .isoLevel = params.isoLevel,
    };

    // reset indirect draw command: indexCount=0, instanceCount=1, rest=0
    VkDrawIndexedIndirectCommand resetCmd{0, 1, 0, 0, 0};
    vkCmdUpdateBuffer(cmd, _indirectBuffer.buffer, 0, sizeof(resetCmd), &resetCmd);

    buffer_barrier(cmd, _indirectBuffer.buffer, sizeof(VkDrawIndexedIndirectCommand),
        VK_PIPELINE_STAGE_2_TRANSFER_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT,
        VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
        VK_ACCESS_2_SHADER_READ_BIT | VK_ACCESS_2_SHADER_WRITE_BIT);

    uint32_t groups = (params.gridSize + 3) / 4;

    // pass 1: density field
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, _densityPipeline);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, _computeLayout,
                            0, 1, &_descriptorSet, 0, nullptr);
    vkCmdPushConstants(cmd, _computeLayout, VK_SHADER_STAGE_COMPUTE_BIT,
                       0, sizeof(push), &push);
    vkCmdDispatch(cmd, groups, groups, groups);

    // barrier: density write -> density read
    buffer_barrier(cmd, _densityBuffer.buffer,
        params.gridSize * params.gridSize * params.gridSize * sizeof(float),
        VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_WRITE_BIT,
        VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_READ_BIT);

    // pass 2: marching cubes
    uint32_t mcGroups = (params.gridSize - 1 + 3) / 4;
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, _mcPipeline);
    vkCmdPushConstants(cmd, _computeLayout, VK_SHADER_STAGE_COMPUTE_BIT,
                       0, sizeof(push), &push);
    vkCmdDispatch(cmd, mcGroups, mcGroups, mcGroups);

    // barrier: compute writes -> vertex input + indirect draw
    buffer_barrier(cmd, _vertexBuffer.buffer, _maxVertices * sizeof(TerrainVertex),
        VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_WRITE_BIT,
        VK_PIPELINE_STAGE_2_VERTEX_INPUT_BIT, VK_ACCESS_2_VERTEX_ATTRIBUTE_READ_BIT);

    buffer_barrier(cmd, _indexBuffer.buffer, _maxVertices * sizeof(uint32_t),
        VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_WRITE_BIT,
        VK_PIPELINE_STAGE_2_INDEX_INPUT_BIT, VK_ACCESS_2_INDEX_READ_BIT);

    buffer_barrier(cmd, _indirectBuffer.buffer, sizeof(VkDrawIndexedIndirectCommand),
        VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_WRITE_BIT,
        VK_PIPELINE_STAGE_2_DRAW_INDIRECT_BIT, VK_ACCESS_2_INDIRECT_COMMAND_READ_BIT);
}

void Terrain::draw(VkCommandBuffer cmd) {
    VkDeviceSize offset = 0;
    vkCmdBindVertexBuffers(cmd, 0, 1, &_vertexBuffer.buffer, &offset);
    vkCmdBindIndexBuffer(cmd, _indexBuffer.buffer, 0, VK_INDEX_TYPE_UINT32);
    vkCmdDrawIndexedIndirect(cmd, _indirectBuffer.buffer, 0, 1, 0);
}
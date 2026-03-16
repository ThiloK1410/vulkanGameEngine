#include "vk_pipelines.h"

#include <fstream>

#include <fmt/core.h>

bool load_shader_module(const char* filePath, VkDevice device, VkShaderModule* outShaderModule) {
  std::ifstream file(filePath, std::ios::ate | std::ios::binary);

  if (!file.is_open()) {
    fmt::println("Failed to open shader file: {}", filePath);
    return false;
  }

  // read the whole file into a uint32_t buffer (SPIR-V is 4-byte aligned)
  size_t fileSize = (size_t)file.tellg();
  std::vector<uint32_t> buffer(fileSize / sizeof(uint32_t));
  file.seekg(0);
  file.read((char*)buffer.data(), fileSize);
  file.close();

  VkShaderModuleCreateInfo createInfo{
      .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
      .codeSize = fileSize,
      .pCode = buffer.data(),
  };

  VkShaderModule shaderModule;
  if (vkCreateShaderModule(device, &createInfo, nullptr, &shaderModule) != VK_SUCCESS) {
    fmt::println("Failed to create shader module: {}", filePath);
    return false;
  }

  *outShaderModule = shaderModule;
  return true;
}

PipelineBuilder::PipelineBuilder() {
  // zero-initialize everything
  _inputAssembly = {.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO};
  _rasterizer = {.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO};
  _colorBlendAttachment = {};
  _multisampling = {.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO};
  _pipelineLayout = {};
  _depthStencil = {.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO};
  _renderInfo = {.sType = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO};
  _colorAttachmentFormat = VK_FORMAT_UNDEFINED;
}

void PipelineBuilder::set_shaders(VkShaderModule vertexShader, VkShaderModule fragmentShader) {
  _shaderStages.clear();
  _shaderStages.push_back({
      .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
      .stage = VK_SHADER_STAGE_VERTEX_BIT,
      .module = vertexShader,
      .pName = "main",
  });
  _shaderStages.push_back({
      .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
      .stage = VK_SHADER_STAGE_FRAGMENT_BIT,
      .module = fragmentShader,
      .pName = "main",
  });
}

void PipelineBuilder::set_input_topology(VkPrimitiveTopology topology) {
  _inputAssembly.topology = topology;
  // primitive restart is for strip topologies — not needed for triangle lists
  _inputAssembly.primitiveRestartEnable = VK_FALSE;
}

void PipelineBuilder::set_polygon_mode(VkPolygonMode mode) {
  _rasterizer.polygonMode = mode;
  _rasterizer.lineWidth = 1.f;
}

void PipelineBuilder::set_cull_mode(VkCullModeFlags cullMode, VkFrontFace frontFace) {
  _rasterizer.cullMode = cullMode;
  _rasterizer.frontFace = frontFace;
}

void PipelineBuilder::set_multisampling_none() {
  _multisampling.sampleShadingEnable = VK_FALSE;
  _multisampling.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
  _multisampling.minSampleShading = 1.0f;
  _multisampling.pSampleMask = nullptr;
  _multisampling.alphaToCoverageEnable = VK_FALSE;
  _multisampling.alphaToOneEnable = VK_FALSE;
}

void PipelineBuilder::set_color_attachment_format(VkFormat format) {
  _colorAttachmentFormat = format;
  _renderInfo.colorAttachmentCount = 1;
  _renderInfo.pColorAttachmentFormats = &_colorAttachmentFormat;
}

void PipelineBuilder::set_vertex_input(uint32_t stride,
                                       std::vector<VkVertexInputAttributeDescription> attributes) {
  _vertexBindings.push_back({
      .binding = 0,
      .stride = stride,
      .inputRate = VK_VERTEX_INPUT_RATE_VERTEX,
  });
  _vertexAttributes = std::move(attributes);
}

void PipelineBuilder::disable_blending() {
  _colorBlendAttachment.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
                                         VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
  _colorBlendAttachment.blendEnable = VK_FALSE;
}

void PipelineBuilder::enable_blending_alpha() {
  _colorBlendAttachment.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
                                         VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
  _colorBlendAttachment.blendEnable = VK_TRUE;
  _colorBlendAttachment.srcColorBlendFactor = VK_BLEND_FACTOR_SRC_ALPHA;
  _colorBlendAttachment.dstColorBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
  _colorBlendAttachment.colorBlendOp = VK_BLEND_OP_ADD;
  _colorBlendAttachment.srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
  _colorBlendAttachment.dstAlphaBlendFactor = VK_BLEND_FACTOR_ZERO;
  _colorBlendAttachment.alphaBlendOp = VK_BLEND_OP_ADD;
}

void PipelineBuilder::set_depth_format(VkFormat format) {
  _renderInfo.depthAttachmentFormat = format;
}

void PipelineBuilder::enable_depthtest(VkCompareOp op) {
  _depthStencil.depthTestEnable = VK_TRUE;
  _depthStencil.depthWriteEnable = VK_TRUE;
  _depthStencil.depthCompareOp = op;
  _depthStencil.depthBoundsTestEnable = VK_FALSE;
  _depthStencil.stencilTestEnable = VK_FALSE;
}

void PipelineBuilder::disable_depthtest() {
  _depthStencil.depthTestEnable = VK_FALSE;
  _depthStencil.depthWriteEnable = VK_FALSE;
  _depthStencil.depthCompareOp = VK_COMPARE_OP_NEVER;
  _depthStencil.depthBoundsTestEnable = VK_FALSE;
  _depthStencil.stencilTestEnable = VK_FALSE;
}

VkPipeline PipelineBuilder::build(VkDevice device) {
  // viewport and scissor are dynamic — set per frame, not baked into the pipeline
  VkPipelineViewportStateCreateInfo viewportState{
      .sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,
      .viewportCount = 1,
      .scissorCount = 1,
  };

  // vertex input — describes how to read vertex data from bound buffers
  VkPipelineVertexInputStateCreateInfo vertexInputInfo{
      .sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
      .vertexBindingDescriptionCount = (uint32_t)_vertexBindings.size(),
      .pVertexBindingDescriptions = _vertexBindings.data(),
      .vertexAttributeDescriptionCount = (uint32_t)_vertexAttributes.size(),
      .pVertexAttributeDescriptions = _vertexAttributes.data(),
  };

  // color blending — one attachment, applied to all color writes
  VkPipelineColorBlendStateCreateInfo colorBlending{
      .sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
      .logicOpEnable = VK_FALSE,
      .logicOp = VK_LOGIC_OP_COPY,
      .attachmentCount = 1,
      .pAttachments = &_colorBlendAttachment,
  };

  // dynamic state — viewport and scissor are set at draw time
  VkDynamicState dynamicStates[] = {VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR};
  VkPipelineDynamicStateCreateInfo dynamicInfo{
      .sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO,
      .dynamicStateCount = 2,
      .pDynamicStates = dynamicStates,
  };

  VkGraphicsPipelineCreateInfo pipelineInfo{
      .sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
      // chain the dynamic rendering info instead of using a VkRenderPass
      .pNext = &_renderInfo,
      .stageCount = (uint32_t)_shaderStages.size(),
      .pStages = _shaderStages.data(),
      .pVertexInputState = &vertexInputInfo,
      .pInputAssemblyState = &_inputAssembly,
      .pViewportState = &viewportState,
      .pRasterizationState = &_rasterizer,
      .pMultisampleState = &_multisampling,
      .pDepthStencilState = &_depthStencil,
      .pColorBlendState = &colorBlending,
      .pDynamicState = &dynamicInfo,
      .layout = _pipelineLayout,
      .renderPass = VK_NULL_HANDLE, // not used with dynamic rendering
  };

  VkPipeline pipeline;
  if (vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &pipeline) !=
      VK_SUCCESS) {
    fmt::println("Failed to create graphics pipeline");
    return VK_NULL_HANDLE;
  }

  return pipeline;
}

#pragma once

#include <vector>
#include <vulkan/vulkan.h>

// Loads a SPIR-V file and creates a VkShaderModule
bool load_shader_module(const char* filePath, VkDevice device, VkShaderModule* outShaderModule);

// Builder that collects pipeline state and produces a VkPipeline
class PipelineBuilder {
public:
  std::vector<VkPipelineShaderStageCreateInfo> _shaderStages;
  VkPipelineInputAssemblyStateCreateInfo _inputAssembly;
  VkPipelineRasterizationStateCreateInfo _rasterizer;
  VkPipelineColorBlendAttachmentState _colorBlendAttachment;
  VkPipelineMultisampleStateCreateInfo _multisampling;
  VkPipelineLayout _pipelineLayout;
  VkPipelineDepthStencilStateCreateInfo _depthStencil;
  VkPipelineRenderingCreateInfo _renderInfo;
  VkFormat _colorAttachmentFormat;

  std::vector<VkVertexInputBindingDescription> _vertexBindings;
  std::vector<VkVertexInputAttributeDescription> _vertexAttributes;

  PipelineBuilder();

  void set_shaders(VkShaderModule vertexShader, VkShaderModule fragmentShader);
  void set_input_topology(VkPrimitiveTopology topology);
  void set_polygon_mode(VkPolygonMode mode);
  void set_cull_mode(VkCullModeFlags cullMode, VkFrontFace frontFace);
  void set_multisampling_none();
  void set_color_attachment_format(VkFormat format);
  void set_vertex_input(uint32_t stride, std::vector<VkVertexInputAttributeDescription> attributes);
  void set_depth_format(VkFormat format);
  void disable_blending();
  void enable_blending_alpha();
  void enable_depthtest(VkCompareOp op = VK_COMPARE_OP_LESS);
  void disable_depthtest();

  VkPipeline build(VkDevice device);
};

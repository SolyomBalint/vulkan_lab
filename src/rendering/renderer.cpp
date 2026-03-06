#include "rendering/renderer.h"
#include "shader_manager.h"
#include "rendering/common_rendering.h"
#include "window_manager.h"
#include "vulkan_layer.h"
#include "gui/gui_manager.h"
#include "scene/game_object.h"
#include "scene/lights.h"
#include "scene/game_scene.h"
#include "asset_managment/asset_manager.h"
#include "camera/perspective_camera.h"
#include "utility/utility.hpp"
#include "rendering/renderer_dependency_provider.h"

void Renderer::initialize()
{
    query.create();
    createResources();
    createPipelineLayout();
    if (theVulkanLayer.dynamicRenderingAvailable) {
        createPostProcessPipeline();
    }
}

void Renderer::destroy()
{
    query.destroy();
    theVulkanLayer.device.destroy(pipelineLayout);
    descriptorPoolBuilder.destroy();
    perFrameDscSetBuilder.destroyLayout();
    perObjectDscSetBuilder.destroyLayout();
    for (auto& it : perObjectBuffers) it.destroy();
    for (auto& it : perFrameBuffers) it.destroy();
    for (auto& it : pointLightBuffers) it.destroy();
    for (auto& it : directionalLightBuffers) it.destroy();
    if (postProcessPipeline.pipeline) {
        postProcessPipeline.destroy();
    }
    rendererCache.destroy();
    free(aUniformData);
    if (!theVulkanLayer.dynamicRenderingAvailable) {
        theVulkanLayer.device.destroy(renderPass);
    }
}

void Renderer::initializeSwapChainDependentComponents()
{
    createImages();
    updateDescriptorSets();
    if (!theVulkanLayer.dynamicRenderingAvailable) {
        if (firstSwapChainCreate) {
            createRenderPass();
            createPostProcessPipeline();
            firstSwapChainCreate = false;
        }
        createFramebuffers();
    }
}

void Renderer::destroySwapChainDependentComponents()
{
    depthImage.destroy();
    postProcessImage.destroy();
    if (postProcessSampler) {
        theVulkanLayer.device.destroy(postProcessSampler);
        postProcessSampler = nullptr;
    }
    if (!theVulkanLayer.dynamicRenderingAvailable) {
        for (auto& it : framebuffer) {
            theVulkanLayer.device.destroy(it);
        }
    }
}

void Renderer::render(const RenderContext& ctx)
{
    auto& vl = theVulkanLayer;
    auto& wm = theWindowManager;

    // timestamp query
    query.query();
    theGUIManager.addStatistic("Renderer", std::make_tuple("Frame time: ", (query.results[1] - query.results[0]) * vl.timestampPeriod / 1e6, " ms"));
    
    auto vp = vk::Viewport{ 0, 0, (float)wm.swapChainExtent.width, (float)wm.swapChainExtent.height, 0.0, 1.0 };
    auto projectionMx = ctx.cam->P(vp.width / vp.height);

    // Update uniforms

    // per frame uniforms
    PerFrameUniformData fud = {}; // prepare uniform data on CPU
    fud.v = ctx.cam->V();
    fud.ambientLight = glm::vec4{ ctx.pGameScene->ambientLight.color, ctx.pGameScene->ambientLight.power };
    memcpy(perFrameBuffers[ctx.frameID].allocationInfo.pMappedData, &fud, sizeof(PerFrameUniformData)); // upload to GPU through mapped pointer
    perFrameBuffers[ctx.frameID].flush(); // ensure visibility

    // point lights
    {
        PointLightsUBO plUbo = {}; // this will be uploaded to the uniform buffer
        uint32_t idx = 0;
        for (auto& pl : ctx.pGameScene->pointLights) { // iterate over all pointlights and set values
            plUbo.pointLights[idx].position = glm::vec4{ pl.position, 1.0 };
            plUbo.pointLights[idx].power = glm::vec4{ pl.color, pl.power };
            idx++;
        }
        plUbo.lightCount_pad3.x = idx; // number of lights
        memcpy(pointLightBuffers[ctx.frameID].allocationInfo.pMappedData, &plUbo, sizeof(plUbo)); // copy
        pointLightBuffers[ctx.frameID].flush(); // ensure visibility
    }

    {
        DirectionalLightsUBO dlUbo = {};
        uint32_t idx = 0;
        for (auto& dl : ctx.pGameScene->directionalLights) {
            if (idx >= MAX_DIRECTIONALLIGHT_COUNT) break;
            glm::vec3 dir = dl.direction;
            if (glm::dot(dir, dir) <= 0.000001f) {
                dir = glm::vec3{ 0.0f, -1.0f, 0.0f };
            }
            dlUbo.directionalLights[idx].direction = glm::vec4{ glm::normalize(dir), 0.0f };
            dlUbo.directionalLights[idx].power = glm::vec4{ dl.color, dl.power };
            idx++;
        }
        dlUbo.lightCount_pad3.x = idx;
        memcpy(directionalLightBuffers[ctx.frameID].allocationInfo.pMappedData, &dlUbo, sizeof(dlUbo));
        directionalLightBuffers[ctx.frameID].flush();
    }

    // per object uniforms
    {
        uint32_t idx = 0;
        char* uniformChar = (char*)aUniformData;
        for (auto& it : ctx.pGameScene->gameObjects) {
            for (int i = 0; i < it.mesh->meshHandlers.size(); ++i) {
                PerObjectUniformData ud = {};
                ud.mv = fud.v * it.transform.M() * it.mesh->getTransform(i);
                ud.mvp = projectionMx * ud.mv;
                ud.mvInvT = glm::transpose(glm::inverse(ud.mv));
                ud.materialIndex = it.mesh->getMaterialIndex(i);
                memcpy(&uniformChar[idx * dynamicAlignment], &ud, sizeof(PerObjectUniformData));
                idx++;
            }
        }

        if (idx != 0) {
            memcpy(perObjectBuffers[ctx.frameID].allocationInfo.pMappedData, uniformChar, dynamicAlignment * idx);
            perObjectBuffers[ctx.frameID].flush();
        }
    }

    query.beginFrame(ctx.cmd); // reset query
    query.write(ctx.cmd, vk::PipelineStageFlagBits::eTopOfPipe); // write timestamp

    // prepare render pass
    vk::RenderingAttachmentInfo colorInfo = { // color attachment (render target)
        .imageView = vl.dynamicRenderingAvailable ? postProcessImage.view : wm.swapChainImageViews[ctx.imageID],
        .imageLayout = vk::ImageLayout::eColorAttachmentOptimal,
        .loadOp = vk::AttachmentLoadOp::eClear,
        .storeOp = vk::AttachmentStoreOp::eStore,
        .clearValue = vk::ClearColorValue{ 0.0f, 0.0f, 0.0f, 1.0f }
    };

    vk::RenderingAttachmentInfo depthInfo = { // depth attachment
        .imageView = depthImage.view,
        .imageLayout = vk::ImageLayout::eDepthStencilAttachmentOptimal,
        .loadOp = vk::AttachmentLoadOp::eClear,
        .storeOp = vk::AttachmentStoreOp::eDontCare,
        .clearValue = vk::ClearDepthStencilValue{ 1.0f, 0 }
    };

    vk::RenderingInfo renderingInfo = {
        .renderArea = {{0, 0}, wm.swapChainExtent},
        .layerCount = 1,
        .colorAttachmentCount = 1,
        .pColorAttachments = &colorInfo,
        .pDepthAttachment = &depthInfo
    };

    // start render pass
    if (vl.dynamicRenderingAvailable) {
        ctx.cmd.beginRendering(renderingInfo);
    } else {
        startRenderpass(ctx);
    }

    // main render pass
    ctx.cmd.bindDescriptorSets(
        vk::PipelineBindPoint::eGraphics, pipelineLayout,
        0, { perFrameDescriptorSets[ctx.frameID], theAssetManager.materialDescriptorSet }, nullptr
    );

    for (uint32_t idx = 0; auto& it : ctx.pGameScene->gameObjects) {
        for (int i = 0; i < it.mesh->meshHandlers.size(); ++i) {
            auto& cache = rendererCache.initializeElement(it.mesh->getDependencyIndex(i), [&](auto& cache) {
                createPipeline(theRendererDependencyProvider[it.mesh->getDependencyIndex(i)], cache);
            });

            ctx.cmd.bindPipeline(vk::PipelineBindPoint::eGraphics, cache.pipeline);
            ctx.cmd.setViewport(0, vp);
            ctx.cmd.setScissor(0, renderingInfo.renderArea);

            uint32_t dynamicOffset = idx++ * static_cast<uint32_t>(dynamicAlignment);
            ctx.cmd.bindDescriptorSets(
                vk::PipelineBindPoint::eGraphics, pipelineLayout,
                2, perObjectDescriptorSets[ctx.frameID], dynamicOffset
            );

            auto vertexBuffers = it.mesh->getVertexBindingInfo(i);
            auto [bufs, offsets, count] = extractVertexBufferBindingInfo(vertexBuffers, cache.bindings);
            ctx.cmd.bindVertexBuffers(0, count, bufs.data(), offsets.data());

            if (it.mesh->getIndexBindingInfo(i).buffer.valid()) {
                auto info = it.mesh->getIndexBindingInfo(i);
                ctx.cmd.bindIndexBuffer(info.buffer->buffer.buffer, info.offset, info.type);
                ctx.cmd.drawIndexed(it.mesh->getIndexCount(i), 1, 0, 0, 0);
            } else {
                ctx.cmd.draw(it.mesh->getIndexCount(i), 1, 0, 0);
            }
        }
    }

    if (vl.dynamicRenderingAvailable) {
        ctx.cmd.endRendering();
    } else {
        ctx.cmd.endRenderPass();
    }    

    if (vl.dynamicRenderingAvailable) {
        vk::ImageMemoryBarrier2 toShaderRead = {};
        toShaderRead
            .setSrcStageMask(vk::PipelineStageFlagBits2::eColorAttachmentOutput)
            .setSrcAccessMask(vk::AccessFlagBits2::eColorAttachmentWrite)
            .setDstStageMask(vk::PipelineStageFlagBits2::eFragmentShader)
            .setDstAccessMask(vk::AccessFlagBits2::eShaderSampledRead)
            .setOldLayout(vk::ImageLayout::eColorAttachmentOptimal)
            .setNewLayout(vk::ImageLayout::eShaderReadOnlyOptimal)
            .setImage(postProcessImage.image)
            .setSubresourceRange(vk::ImageSubresourceRange{
                vk::ImageAspectFlagBits::eColor,
                0,
                1,
                0,
                1
            });
        vk::DependencyInfo depToShaderRead = {};
        depToShaderRead.setImageMemoryBarriers(toShaderRead);
        ctx.cmd.pipelineBarrier2(depToShaderRead);

        vk::RenderingAttachmentInfo postProcessColorInfo = {
            .imageView = wm.swapChainImageViews[ctx.imageID],
            .imageLayout = vk::ImageLayout::eColorAttachmentOptimal,
            .loadOp = vk::AttachmentLoadOp::eClear,
            .storeOp = vk::AttachmentStoreOp::eStore,
            .clearValue = vk::ClearColorValue{ 0.0f, 0.0f, 0.0f, 1.0f }
        };

        vk::RenderingInfo postProcessRenderingInfo = {
            .renderArea = {{0, 0}, wm.swapChainExtent},
            .layerCount = 1,
            .colorAttachmentCount = 1,
            .pColorAttachments = &postProcessColorInfo,
        };

        ctx.cmd.beginRendering(postProcessRenderingInfo);
        ctx.cmd.bindPipeline(vk::PipelineBindPoint::eGraphics, postProcessPipeline.pipeline);
        ctx.cmd.setViewport(0, vp);
        ctx.cmd.setScissor(0, postProcessRenderingInfo.renderArea);
        ctx.cmd.bindDescriptorSets(
            vk::PipelineBindPoint::eGraphics, pipelineLayout,
            0, perFrameDescriptorSets[ctx.frameID], nullptr
        );
        ctx.cmd.draw(3, 1, 0, 0);
        ctx.cmd.endRendering();

        vk::ImageMemoryBarrier2 backToColorAttachment = {};
        backToColorAttachment
            .setSrcStageMask(vk::PipelineStageFlagBits2::eFragmentShader)
            .setSrcAccessMask(vk::AccessFlagBits2::eShaderSampledRead)
            .setDstStageMask(vk::PipelineStageFlagBits2::eColorAttachmentOutput)
            .setDstAccessMask(vk::AccessFlagBits2::eColorAttachmentWrite)
            .setOldLayout(vk::ImageLayout::eShaderReadOnlyOptimal)
            .setNewLayout(vk::ImageLayout::eColorAttachmentOptimal)
            .setImage(postProcessImage.image)
            .setSubresourceRange(vk::ImageSubresourceRange{
                vk::ImageAspectFlagBits::eColor,
                0,
                1,
                0,
                1
            });
        vk::DependencyInfo depBackToColorAttachment = {};
        depBackToColorAttachment.setImageMemoryBarriers(backToColorAttachment);
        ctx.cmd.pipelineBarrier2(depBackToColorAttachment);
    }

    query.write(ctx.cmd, vk::PipelineStageFlagBits::eBottomOfPipe);

    query.endFrame();
}

void Renderer::createPipelineLayout()
{
    descriptorPoolBuilder.initialize();

    perFrameDscSetBuilder
        .setMaximumSetCount(MAX_FRAMES_IN_FLIGHT)
        .addBinding(0, vk::DescriptorType::eUniformBuffer, // perFrameUniformData
            vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eFragment)
        .addBinding(1, vk::DescriptorType::eUniformBuffer, // pointLightUniformData
            vk::ShaderStageFlagBits::eFragment)
        .addBinding(2, vk::DescriptorType::eUniformBuffer, // directionalLightUniformData
            vk::ShaderStageFlagBits::eFragment)
        .addBinding(3, vk::DescriptorType::eCombinedImageSampler, // post process render target
            vk::ShaderStageFlagBits::eFragment)
        .createLayout();

    perObjectDscSetBuilder.setMaximumSetCount(MAX_FRAMES_IN_FLIGHT)
        .addBinding(0, vk::DescriptorType::eUniformBufferDynamic, // per object uniform
            vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eFragment)
        .createLayout();

    descriptorPoolBuilder.create();
    perFrameDescriptorSets = perFrameDscSetBuilder.allocateSets<MAX_FRAMES_IN_FLIGHT>();
    perObjectDescriptorSets = perObjectDscSetBuilder.allocateSets<MAX_FRAMES_IN_FLIGHT>();

    pipelineLayout = theVulkanLayer.name(theVulkanLayer.createPipelineLayout(
        { perFrameDscSetBuilder.layout(), theAssetManager.materialSetLayout, perObjectDscSetBuilder.layout() }
    ), "Renderer_PipelineLayout");
}

void Renderer::updateDescriptorSets()
{
    perObjectDscSetBuilder.update(perObjectDescriptorSets)
        .writeBuffer(0, perObjectBuffers, 0, sizeof(PerObjectUniformData));

    perFrameDscSetBuilder.update(perFrameDescriptorSets)
        .writeBuffer(0, perFrameBuffers)
        .writeBuffer(1, pointLightBuffers)
        .writeBuffer(2, directionalLightBuffers);

    std::array<vk::DescriptorImageInfo, MAX_FRAMES_IN_FLIGHT> postProcessInfos;
    std::array<vk::WriteDescriptorSet, MAX_FRAMES_IN_FLIGHT> postProcessWrites;
    for (uint32_t i = 0; i < MAX_FRAMES_IN_FLIGHT; ++i) {
        postProcessInfos[i] = {
            .sampler = postProcessSampler,
            .imageView = postProcessImage.view,
            .imageLayout = vk::ImageLayout::eShaderReadOnlyOptimal
        };
        postProcessWrites[i] = {
            .dstSet = perFrameDescriptorSets[i],
            .dstBinding = 3,
            .dstArrayElement = 0,
            .descriptorCount = 1,
            .descriptorType = vk::DescriptorType::eCombinedImageSampler,
            .pImageInfo = &postProcessInfos[i]
        };
    }
    theVulkanLayer.device.updateDescriptorSets(postProcessWrites, nullptr);
}

void Renderer::createPostProcessPipeline()
{
    if (theVulkanLayer.dynamicRenderingAvailable) {
        theShaderManager.addAndExecuteShaderDependency(postProcessPipeline, { "post_process.frag" },
            [this](auto shaders) {
                auto& wm = theWindowManager;

                vk::PipelineRenderingCreateInfo renderingInfo = {
                    .colorAttachmentCount = 1,
                    .pColorAttachmentFormats = &wm.swapChainImageFormat
                };

                return theVulkanLayer.name(thePipelineBuildHelpers.createFullscreenPipeline(
                    pipelineLayout,
                    renderingInfo,
                    shaders[0]
                ), "Renderer_PostProcessPipeline");
            }
        );
    } else {
        theShaderManager.addAndExecuteShaderDependency(postProcessPipeline, { "post_process.frag" },
            [this](auto shaders) {
                return theVulkanLayer.name(thePipelineBuildHelpers.createFullscreenPipeline(
                    pipelineLayout,
                    renderPass,
                    0,
                    shaders[0]
                ), "Renderer_PostProcessPipeline");
            }
        );
    }
}

void Renderer::createPipeline(const RendererDependency& dep, CachedRenderingData& cache)
{
    std::array<std::string, 2> shaders;
    bool useTextures = dep.layout.infos.contains(VertexDataElem::TextureCoordinate)
        && dep.layout.infos.contains(VertexDataElem::Tangent)
        && theVulkanLayer.descriptorIndexingAvailable;

    if (useTextures) {
        shaders[0] = "forward.vert";
        shaders[1] = "forward.frag";
    } else {
        shaders[0] = "forward_simple.vert";
        shaders[1] = "forward_simple.frag";
    }

    theShaderManager.addAndExecuteShaderDependency(cache.pipeline, { shaders[0], shaders[1] },
        [this, &cache, dep = dep, useTextures = useTextures]
            (auto shaders) {
        auto& vl = theVulkanLayer;
        auto& wm = theWindowManager;

        int binding = 0;
        VertexInputBindingBuilder vertexInputBuilder;
        vertexInputBuilder.
            addVertexAttribute(0, vk::Format::eR32G32B32Sfloat, dep.layout.infos.find(VertexDataElem::Position)->second.stride).
            addVertexAttribute(1, vk::Format::eR32G32B32Sfloat, dep.layout.infos.find(VertexDataElem::Normal)->second.stride);
        cache.bindings[binding++] = VertexDataElem::Position;
        cache.bindings[binding++] = VertexDataElem::Normal;

        if (useTextures) {
            vertexInputBuilder.
                addVertexAttribute(2, vk::Format::eR32G32Sfloat, dep.layout.infos.find(VertexDataElem::TextureCoordinate)->second.stride)
                .addVertexAttribute(3, vk::Format::eR32G32B32A32Sfloat, dep.layout.infos.find(VertexDataElem::Tangent)->second.stride);
            cache.bindings[binding++] = VertexDataElem::TextureCoordinate;
            cache.bindings[binding++] = VertexDataElem::Tangent;
        }

        auto vertexInputDesc = vertexInputBuilder.build();

        vk::PipelineRasterizationStateCreateInfo rasterizer = {
            .depthClampEnable = VK_FALSE,
            .rasterizerDiscardEnable = VK_FALSE,
            .polygonMode = vk::PolygonMode::eFill,
            .cullMode = vk::CullModeFlagBits::eBack,
            .frontFace = vk::FrontFace::eCounterClockwise,
            .depthBiasEnable = VK_FALSE,
            .lineWidth = 1.0f
        };

        vk::PipelineMultisampleStateCreateInfo multisampling = {
            .rasterizationSamples = vk::SampleCountFlagBits::e1,
            .sampleShadingEnable = VK_FALSE
        };

        vk::PipelineDepthStencilStateCreateInfo depthStencil = {
            .depthTestEnable = VK_TRUE,
            .depthWriteEnable = VK_TRUE,
            .depthCompareOp = vk::CompareOp::eLess,
            .depthBoundsTestEnable = VK_FALSE,
            .stencilTestEnable = VK_FALSE
        };

        vk::PipelineRenderingCreateInfo renderingInfo = {
            .colorAttachmentCount = 1,
            .pColorAttachmentFormats = &wm.swapChainImageFormat,
            .depthAttachmentFormat = vl.findDepthFormat()
        };

        vk::GraphicsPipelineCreateInfo pipelineInfo = {
            .pNext = &renderingInfo,
            .stageCount = 2,
            .pStages = shaders,
            .pVertexInputState = &vertexInputDesc,
            .pInputAssemblyState = &thePipelineBuildHelpers.triangleListInputAssembly,
            .pViewportState = &thePipelineBuildHelpers.dynamicViewportScissorState,
            .pRasterizationState = &rasterizer,
            .pMultisampleState = &multisampling,
            .pDepthStencilState = &depthStencil,
            .pColorBlendState = thePipelineBuildHelpers.getDisabledColorBlendingState<1>(),
            .pDynamicState = thePipelineBuildHelpers.getDynamicState<vk::DynamicState::eViewport, vk::DynamicState::eScissor>(),
            .layout = pipelineLayout,
            .renderPass = vl.dynamicRenderingAvailable ? nullptr : renderPass,
            .subpass = 0
        };

        return theVulkanLayer.name(theVulkanLayer.createGraphicsPipeline(pipelineInfo), "Renderer_Pipeline");
    });
}

void Renderer::createImages()
{
    depthImage = theVulkanLayer.name(theVulkanLayer.createImage(
        theWindowManager.swapChainExtent.width, theWindowManager.swapChainExtent.height, 1,
        vk::SampleCountFlagBits::e1,
        theVulkanLayer.findDepthFormat(), vk::ImageLayout::eDepthStencilAttachmentOptimal,
        vk::ImageAspectFlagBits::eDepth | vk::ImageAspectFlagBits::eStencil,
        vk::ImageUsageFlagBits::eDepthStencilAttachment
    ), "Renderer_DepthImage");

    postProcessImage = theVulkanLayer.name(theVulkanLayer.createImage(
        theWindowManager.swapChainExtent.width,
        theWindowManager.swapChainExtent.height,
        1,
        vk::SampleCountFlagBits::e1,
        theWindowManager.swapChainImageFormat,
        vk::ImageLayout::eColorAttachmentOptimal,
        vk::ImageAspectFlagBits::eColor,
        vk::ImageUsageFlagBits::eColorAttachment | vk::ImageUsageFlagBits::eSampled
    ), "Renderer_PostProcessImage");

    vk::SamplerCreateInfo samplerInfo = {
        .magFilter = vk::Filter::eLinear,
        .minFilter = vk::Filter::eLinear,
        .mipmapMode = vk::SamplerMipmapMode::eLinear,
        .addressModeU = vk::SamplerAddressMode::eClampToEdge,
        .addressModeV = vk::SamplerAddressMode::eClampToEdge,
        .addressModeW = vk::SamplerAddressMode::eClampToEdge,
        .mipLodBias = 0.0f,
        .anisotropyEnable = VK_FALSE,
        .maxAnisotropy = 1.0f,
        .compareEnable = VK_FALSE,
        .compareOp = vk::CompareOp::eAlways,
        .minLod = 0.0f,
        .maxLod = 0.0f,
        .borderColor = vk::BorderColor::eFloatOpaqueBlack,
        .unnormalizedCoordinates = VK_FALSE
    };
    postProcessSampler = theVulkanLayer.name(theVulkanLayer.device.createSampler(samplerInfo), "Renderer_PostProcessSampler");
}

void Renderer::createResources()
{
    // Calculate required alignment based on minimum device offset alignment
    vk::DeviceSize minUbAlignment = theVulkanLayer.physicalDeviceProperties.limits.minUniformBufferOffsetAlignment;
    dynamicAlignment = getAlignedSize(sizeof(PerObjectUniformData), minUbAlignment);
    size_t perObjectUbosBufferSize = MAX_OBJECT_NUM * dynamicAlignment;
    aUniformData = (PerObjectUniformData*)malloc(perObjectUbosBufferSize);

    for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; ++i) {
        perFrameBuffers[i] = theVulkanLayer.createBuffer(sizeof(PerFrameUniformData), vk::BufferUsageFlagBits::eUniformBuffer,
            vma::AllocationCreateFlagBits::eHostAccessSequentialWriteBit | vma::AllocationCreateFlagBits::eCreateMappedBit);
        perObjectBuffers[i] = theVulkanLayer.createBuffer(perObjectUbosBufferSize, vk::BufferUsageFlagBits::eUniformBuffer,
            vma::AllocationCreateFlagBits::eHostAccessSequentialWriteBit | vma::AllocationCreateFlagBits::eCreateMappedBit);
        pointLightBuffers[i] = theVulkanLayer.createBuffer(sizeof(PointLightsUBO), vk::BufferUsageFlagBits::eUniformBuffer,
            vma::AllocationCreateFlagBits::eHostAccessSequentialWriteBit | vma::AllocationCreateFlagBits::eCreateMappedBit);
        directionalLightBuffers[i] = theVulkanLayer.createBuffer(sizeof(DirectionalLightsUBO), vk::BufferUsageFlagBits::eUniformBuffer,
            vma::AllocationCreateFlagBits::eHostAccessSequentialWriteBit | vma::AllocationCreateFlagBits::eCreateMappedBit);
    }
}

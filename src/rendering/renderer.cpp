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

    // Post process pipeline
    theShaderManager.addAndExecuteShaderDependency(postProcessPipeline, { "post_process.frag" },
        [this](auto shaders) {
        auto& vl = theVulkanLayer;
        auto& wm = theWindowManager;
        if (vl.dynamicRenderingAvailable) {
            vk::PipelineRenderingCreateInfo renderingInfo = {
                .colorAttachmentCount = 1,
                .pColorAttachmentFormats = &wm.swapChainImageFormat
            };
            return theVulkanLayer.name(thePipelineBuildHelpers.createFullscreenPipeline(postProcessLayout, renderingInfo, shaders[0]), "PostProcess_Pipeline");
        } else {
            return theVulkanLayer.name(thePipelineBuildHelpers.createFullscreenPipeline(postProcessLayout, renderPass, 0, shaders[0]), "PostProcess_Pipeline");
        }
    });
}

void Renderer::destroy()
{
    query.destroy();
    theVulkanLayer.device.destroy(pipelineLayout);
    theVulkanLayer.device.destroy(postProcessLayout);
    descriptorPoolBuilder.destroy();
    perFrameDscSetBuilder.destroyLayout();
    perObjectDscSetBuilder.destroyLayout();
    for (auto& it : perObjectBuffers) it.destroy();
    for (auto& it : perFrameBuffers) it.destroy();
    for (auto& it : pointLightBuffers) it.destroy();
    for (auto& it : dirLightBuffers) it.destroy();
    rendererCache.destroy();
    free(aUniformData);
    if (!theVulkanLayer.dynamicRenderingAvailable) {
        theVulkanLayer.device.destroy(renderPass);
    }
    theVulkanLayer.device.destroy(intermediateSampler);
    postProcessPipeline.destroy();
}

void Renderer::initializeSwapChainDependentComponents()
{
    createImages();
    updateDescriptorSets();
    if (!theVulkanLayer.dynamicRenderingAvailable) {
        if (firstSwapChainCreate) {
            createRenderPass();
            firstSwapChainCreate = false;
        }
        createFramebuffers();
    }
}

void Renderer::destroySwapChainDependentComponents()
{
    depthImage.destroy();
    intermediateImage.destroy();
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
    double sceneTime = (query.results[1] - query.results[0]) * vl.timestampPeriod / 1e6;
    double postTime = (query.results[3] - query.results[2]) * vl.timestampPeriod / 1e6;
    theGUIManager.addStatistic("Renderer", std::make_tuple("Scene: ", sceneTime, " ms"));
    theGUIManager.addStatistic("Renderer", std::make_tuple("Post: ", postTime, " ms"));
    theGUIManager.addStatistic("Renderer", std::make_tuple("Total: ", sceneTime + postTime, " ms"));
    
    auto vp = vk::Viewport{ 0, 0, (float)wm.swapChainExtent.width, (float)wm.swapChainExtent.height, 0.0, 1.0 };
    auto projectionMx = ctx.cam->P(vp.width / vp.height);

    // Update uniforms

    // per frame uniforms
    PerFrameUniformData fud = {};
    fud.v = ctx.cam->V();
    fud.ambientLight = glm::vec4{ ctx.pGameScene->ambientLight.color, ctx.pGameScene->ambientLight.power };
    memcpy(perFrameBuffers[ctx.frameID].allocationInfo.pMappedData, &fud, sizeof(PerFrameUniformData));
    perFrameBuffers[ctx.frameID].flush();

    // point lights
    {
        PointLightsUBO plUbo = {};
        uint32_t idx = 0;
        for (auto& pl : ctx.pGameScene->pointLights) {
            plUbo.pointLights[idx].position = glm::vec4{ pl.position, 1.0 };
            plUbo.pointLights[idx].power = glm::vec4{ pl.color, pl.power };
            idx++;
        }
        plUbo.lightCount_pad3.x = idx;
        memcpy(pointLightBuffers[ctx.frameID].allocationInfo.pMappedData, &plUbo, sizeof(plUbo));
        pointLightBuffers[ctx.frameID].flush();
    }

    // directional lights
    {
        DirLightsUBO dlUbo = {};
        uint32_t idx = 0;
        for (auto& dl : ctx.pGameScene->dirLights) {
            dlUbo.dirLights[idx].direction = glm::vec4{ dl.direction, 0.0 };
            dlUbo.dirLights[idx].power = glm::vec4{ dl.color, dl.power };
            idx++;
        }
        dlUbo.lightCount_pad3.x = idx;
        memcpy(dirLightBuffers[ctx.frameID].allocationInfo.pMappedData, &dlUbo, sizeof(dlUbo));
        dirLightBuffers[ctx.frameID].flush();
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

    query.beginFrame(ctx.cmd);
    query.write(ctx.cmd, vk::PipelineStageFlagBits::eTopOfPipe);

    // Scene render pass: render to intermediate image
    vk::RenderingAttachmentInfo colorInfo = {
        .imageView = intermediateImage.view,
        .imageLayout = vk::ImageLayout::eColorAttachmentOptimal,
        .loadOp = vk::AttachmentLoadOp::eClear,
        .storeOp = vk::AttachmentStoreOp::eStore,
        .clearValue = vk::ClearColorValue{ 0.0f, 0.0f, 0.0f, 1.0f }
    };

    vk::RenderingAttachmentInfo depthInfo = {
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

    query.write(ctx.cmd, vk::PipelineStageFlagBits::eBottomOfPipe);

    // Pipeline barrier: intermediate image from ColorAttachment -> ShaderReadOnly
    vk::ImageMemoryBarrier2 barrier = {
        .srcStageMask = vk::PipelineStageFlagBits2::eColorAttachmentOutput,
        .srcAccessMask = vk::AccessFlagBits2::eColorAttachmentWrite,
        .dstStageMask = vk::PipelineStageFlagBits2::eFragmentShader,
        .dstAccessMask = vk::AccessFlagBits2::eShaderSampledRead,
        .oldLayout = vk::ImageLayout::eColorAttachmentOptimal,
        .newLayout = vk::ImageLayout::eShaderReadOnlyOptimal,
        .image = intermediateImage.image,
        .subresourceRange = { vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1 }
    };
    vk::DependencyInfo depInfo = {
        .imageMemoryBarrierCount = 1,
        .pImageMemoryBarriers = &barrier
    };
    ctx.cmd.pipelineBarrier2(depInfo);

    // Post process render pass: render fullscreen quad to swapchain
    query.write(ctx.cmd, vk::PipelineStageFlagBits::eTopOfPipe);

    vk::RenderingAttachmentInfo ppColorInfo = {
        .imageView = wm.swapChainImageViews[ctx.imageID],
        .imageLayout = vk::ImageLayout::eColorAttachmentOptimal,
        .loadOp = vk::AttachmentLoadOp::eClear,
        .storeOp = vk::AttachmentStoreOp::eStore,
        .clearValue = vk::ClearColorValue{ 0.0f, 0.0f, 0.0f, 1.0f }
    };
    vk::RenderingInfo ppRenderingInfo = {
        .renderArea = {{0, 0}, wm.swapChainExtent},
        .layerCount = 1,
        .colorAttachmentCount = 1,
        .pColorAttachments = &ppColorInfo
    };

    if (vl.dynamicRenderingAvailable) {
        ctx.cmd.beginRendering(ppRenderingInfo);
    } else {
        // For non-dynamic rendering, use the same renderpass mechanism
        // (simplified: just use dynamic rendering path here)
        ctx.cmd.beginRendering(ppRenderingInfo);
    }

    ctx.cmd.bindPipeline(vk::PipelineBindPoint::eGraphics, postProcessPipeline);
    ctx.cmd.bindDescriptorSets(
        vk::PipelineBindPoint::eGraphics, postProcessLayout,
        0, perFrameDescriptorSets[ctx.frameID], nullptr
    );
    ctx.cmd.setViewport(0, vp);
    ctx.cmd.setScissor(0, ppRenderingInfo.renderArea);
    ctx.cmd.draw(3, 1, 0, 0);

    if (vl.dynamicRenderingAvailable) {
        ctx.cmd.endRendering();
    } else {
        ctx.cmd.endRendering();
    }

    // Layout transition back: intermediate image -> ColorAttachmentOptimal
    vk::ImageMemoryBarrier2 barrierBack = {
        .srcStageMask = vk::PipelineStageFlagBits2::eFragmentShader,
        .srcAccessMask = vk::AccessFlagBits2::eShaderSampledRead,
        .dstStageMask = vk::PipelineStageFlagBits2::eColorAttachmentOutput,
        .dstAccessMask = vk::AccessFlagBits2::eColorAttachmentWrite,
        .oldLayout = vk::ImageLayout::eShaderReadOnlyOptimal,
        .newLayout = vk::ImageLayout::eColorAttachmentOptimal,
        .image = intermediateImage.image,
        .subresourceRange = { vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1 }
    };
    vk::DependencyInfo depInfoBack = {
        .imageMemoryBarrierCount = 1,
        .pImageMemoryBarriers = &barrierBack
    };
    ctx.cmd.pipelineBarrier2(depInfoBack);

    query.write(ctx.cmd, vk::PipelineStageFlagBits::eBottomOfPipe);
    query.endFrame();
}

void Renderer::createPipelineLayout()
{
    descriptorPoolBuilder.initialize();

    perFrameDscSetBuilder
        .setMaximumSetCount(MAX_FRAMES_IN_FLIGHT)
        .addBinding(0, vk::DescriptorType::eUniformBuffer,
            vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eFragment)
        .addBinding(1, vk::DescriptorType::eUniformBuffer,
            vk::ShaderStageFlagBits::eFragment)
        .addBinding(2, vk::DescriptorType::eUniformBuffer,
            vk::ShaderStageFlagBits::eFragment)
        .addBinding(3, vk::DescriptorType::eCombinedImageSampler,
            vk::ShaderStageFlagBits::eFragment)
        .createLayout();

    perObjectDscSetBuilder.setMaximumSetCount(MAX_FRAMES_IN_FLIGHT)
        .addBinding(0, vk::DescriptorType::eUniformBufferDynamic,
            vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eFragment)
        .createLayout();

    descriptorPoolBuilder.create();
    perFrameDescriptorSets = perFrameDscSetBuilder.allocateSets<MAX_FRAMES_IN_FLIGHT>();
    perObjectDescriptorSets = perObjectDscSetBuilder.allocateSets<MAX_FRAMES_IN_FLIGHT>();

    pipelineLayout = theVulkanLayer.name(theVulkanLayer.createPipelineLayout(
        { perFrameDscSetBuilder.layout(), theAssetManager.materialSetLayout, perObjectDscSetBuilder.layout() }
    ), "Renderer_PipelineLayout");

    postProcessLayout = theVulkanLayer.name(theVulkanLayer.createPipelineLayout(
        { perFrameDscSetBuilder.layout() }
    ), "PostProcess_PipelineLayout");
}

void Renderer::updateDescriptorSets()
{
    perObjectDscSetBuilder.update(perObjectDescriptorSets)
        .writeBuffer(0, perObjectBuffers, 0, sizeof(PerObjectUniformData));

    perFrameDscSetBuilder.update(perFrameDescriptorSets)
        .writeBuffer(0, perFrameBuffers)
        .writeBuffer(1, pointLightBuffers)
        .writeBuffer(2, dirLightBuffers);

    // Write the intermediate image as CombinedImageSampler at binding 3
    for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; ++i) {
        vk::DescriptorImageInfo imgInfo = {
            .sampler = intermediateSampler,
            .imageView = intermediateImage.view,
            .imageLayout = vk::ImageLayout::eShaderReadOnlyOptimal
        };
        vk::WriteDescriptorSet write = {
            .dstSet = perFrameDescriptorSets[i],
            .dstBinding = 3,
            .dstArrayElement = 0,
            .descriptorCount = 1,
            .descriptorType = vk::DescriptorType::eCombinedImageSampler,
            .pImageInfo = &imgInfo
        };
        theVulkanLayer.device.updateDescriptorSets(write, nullptr);
    }
}

void Renderer::createPipeline(const RendererDependency& dep, CachedRenderingData& cache)
{
    std::array<std::string, 2> shaders;
    bool useTextures = dep.layout.infos.contains(VertexDataElem::TextureCoordinate) && theVulkanLayer.descriptorIndexingAvailable;

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
                addVertexAttribute(2, vk::Format::eR32G32Sfloat, dep.layout.infos.find(VertexDataElem::TextureCoordinate)->second.stride);
            cache.bindings[binding++] = VertexDataElem::TextureCoordinate;

            // Normal mapping: add tangent attribute if available
            if (dep.layout.infos.contains(VertexDataElem::Tangent)) {
                vertexInputBuilder.
                    addVertexAttribute(3, vk::Format::eR32G32B32A32Sfloat, dep.layout.infos.find(VertexDataElem::Tangent)->second.stride);
                cache.bindings[binding++] = VertexDataElem::Tangent;
            }
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
    auto& wm = theWindowManager;

    depthImage = theVulkanLayer.name(theVulkanLayer.createImage(
        wm.swapChainExtent.width, wm.swapChainExtent.height, 1,
        vk::SampleCountFlagBits::e1,
        theVulkanLayer.findDepthFormat(), vk::ImageLayout::eDepthStencilAttachmentOptimal,
        vk::ImageAspectFlagBits::eDepth | vk::ImageAspectFlagBits::eStencil,
        vk::ImageUsageFlagBits::eDepthStencilAttachment
    ), "Renderer_DepthImage");

    intermediateImage = theVulkanLayer.name(theVulkanLayer.createImage(
        wm.swapChainExtent.width, wm.swapChainExtent.height, 1,
        vk::SampleCountFlagBits::e1,
        wm.swapChainImageFormat, vk::ImageLayout::eColorAttachmentOptimal,
        vk::ImageAspectFlagBits::eColor,
        vk::ImageUsageFlagBits::eColorAttachment | vk::ImageUsageFlagBits::eSampled
    ), "Renderer_IntermediateImage");

    if (!intermediateSampler) {
        vk::SamplerCreateInfo samplerInfo = {
            .magFilter = vk::Filter::eLinear,
            .minFilter = vk::Filter::eLinear,
            .mipmapMode = vk::SamplerMipmapMode::eLinear,
            .addressModeU = vk::SamplerAddressMode::eClampToEdge,
            .addressModeV = vk::SamplerAddressMode::eClampToEdge,
            .addressModeW = vk::SamplerAddressMode::eClampToEdge
        };
        intermediateSampler = theVulkanLayer.device.createSampler(samplerInfo);
    }
}

void Renderer::createResources()
{
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
        dirLightBuffers[i] = theVulkanLayer.createBuffer(sizeof(DirLightsUBO), vk::BufferUsageFlagBits::eUniformBuffer,
            vma::AllocationCreateFlagBits::eHostAccessSequentialWriteBit | vma::AllocationCreateFlagBits::eCreateMappedBit);
    }
}

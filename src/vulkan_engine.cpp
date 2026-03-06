#include "vulkan_engine.h"
#include "vulkan_layer.h"
#include "window_manager.h"
#include "gui/gui_manager.h"
#include "shader_manager.h"
#include "rendering/renderer.h"

namespace fs = std::filesystem;
using namespace glfwim;

void VulkanEngine::initialize()
{
    createCommandBuffers();
    createSemaphores();
}

void VulkanEngine::initializeSwapchainDependentComponents()
{
    renderFinishedSemaphores.reserve(theWindowManager.swapChainImages.size());
    vk::SemaphoreCreateInfo semaphoreInfo = {};
    for (size_t i = 0; i < theWindowManager.swapChainImages.size(); i++) {
        renderFinishedSemaphores.push_back(theVulkanLayer.device.createSemaphore(semaphoreInfo));
    }
}

void VulkanEngine::destroySwapchainDependentComponents()
{
    for (auto& it : renderFinishedSemaphores) {
        theVulkanLayer.device.destroy(it);
    }
    renderFinishedSemaphores.clear();
}

void VulkanEngine::destroy()
{
    auto& vl = theVulkanLayer;
    for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
        vl.device.destroy(imageAvailableSemaphores[i]);
        vl.device.destroy(inFlightFences[i]);
    }
    imageAvailableSemaphores.clear();
    inFlightFences.clear();

    vl.device.freeCommandBuffers(vl.commandPool, commandBuffers);
}

void VulkanEngine::updateGui()
{
    if (ImGui::Begin("Settings")) {
        if (ImGui::Button("Reload shaders")) {
            waitIdle();
            theShaderManager.rebuildShaders();
        }

        ImGui::SameLine();
        if (ImGui::Button("Reload all shaders")) {
            waitIdle();
            theShaderManager.rebuildShaders(true);
        }
    }

    ImGui::End();
}

void VulkanEngine::waitIdle()
{
    theVulkanLayer.safeWaitIdle();
}

VulkanEngine::Result VulkanEngine::drawFrame(Renderer* pRenderer, PerspectiveCamera* pCamera, GameScene* pGameScene)
{
    auto& vl = theVulkanLayer;

    if (theWindowManager.swapChainExtent.width == 0 || theWindowManager.swapChainExtent.height == 0) {
        return Result::SwapChainOutOfDate;
    }

    VK_CHECK_RESULT(vl.device.waitForFences(inFlightFences[currentFrame], VK_TRUE, std::numeric_limits<uint64_t>::max()))

    uint32_t imageIndex;
    auto result = vl.device.acquireNextImageKHR(
            theWindowManager.swapChain, std::numeric_limits<uint64_t>::max(),
            imageAvailableSemaphores[currentFrame], nullptr, &imageIndex
    );

    if (result == vk::Result::eErrorOutOfDateKHR) {
        return Result::SwapChainOutOfDate;
    } else if (result != vk::Result::eSuccess && result != vk::Result::eSuboptimalKHR) {
        throw std::runtime_error("failed to acquire swap chain image!");
    }

    vk::CommandBufferBeginInfo beginInfo = {}; beginInfo
        .setFlags(vk::CommandBufferUsageFlagBits::eOneTimeSubmit);
    commandBuffers[currentFrame].begin(beginInfo);

    vk::ImageMemoryBarrier2 bar = {}; bar
        .setImage(theWindowManager.swapChainImages[imageIndex])
        .setOldLayout(vk::ImageLayout::eUndefined)
        .setNewLayout(vk::ImageLayout::eColorAttachmentOptimal)
        .setSrcStageMask(vk::PipelineStageFlagBits2::eColorAttachmentOutput)
        .setSrcAccessMask(vk::AccessFlagBits2KHR::eColorAttachmentWrite | vk::AccessFlagBits2KHR::eColorAttachmentRead)
        .setDstStageMask(vk::PipelineStageFlagBits2::eColorAttachmentOutput)
        .setDstAccessMask(vk::AccessFlagBits2KHR::eColorAttachmentWrite | vk::AccessFlagBits2KHR::eColorAttachmentRead)
        .setSubresourceRange(vk::ImageSubresourceRange{
            vk::ImageAspectFlagBits::eColor /* aspectMask */,
            0 /* baseMipLevel */, 1 /* levelCount */,
            0 /* baseArrayLayer */, 1 /* layerCount*/
    });

    vk::DependencyInfo dep = {};
    dep.setImageMemoryBarriers(bar);
    commandBuffers[currentFrame].pipelineBarrier2(dep);

    ImGui::Begin("Renderer parameters");
    auto rctx = RenderContext{
        pGameScene, pCamera,
        commandBuffers[currentFrame],
        currentFrame, imageIndex
    };
    pRenderer->render(rctx);
    ImGui::End();

    theGUIManager.render(RenderContext {
            nullptr, nullptr,
            commandBuffers[currentFrame],
            currentFrame, imageIndex
    });

    commandBuffers[currentFrame].end();

    vk::PipelineStageFlags waitStages[] = { vk::PipelineStageFlagBits::eColorAttachmentOutput };
    vk::SubmitInfo submitInfo = {}; submitInfo
        .setWaitSemaphores(imageAvailableSemaphores[currentFrame])
        .setWaitDstStageMask(waitStages)
        .setCommandBuffers(commandBuffers[currentFrame])
        .setSignalSemaphores(renderFinishedSemaphores[imageIndex]);

    VK_CHECK_RESULT(vl.device.resetFences(1, &inFlightFences[currentFrame]))
    auto res = vl.safeSubmitGraphicsCommand(submitInfo, false, inFlightFences[currentFrame]);
    if (res != vk::Result::eSuccess) {
        vulkanLogger.LogError("Failed to submit draw command buffer!");
    }

    vk::PresentInfoKHR presentInfo = {}; presentInfo
        .setWaitSemaphores(renderFinishedSemaphores[imageIndex])
        .setSwapchains(theWindowManager.swapChain)
        .setImageIndices(imageIndex);

    Result ret = Result::Success;
    result = vk::Result::eErrorOutOfDateKHR;
    try {
        result = vl.safeSubmitPresentCommand(presentInfo);
    } catch (...){}

    if (result == vk::Result::eErrorOutOfDateKHR || result == vk::Result::eSuboptimalKHR || theWindowManager.frameBufferResized) {
        ret = Result::SwapChainOutOfDate;
    } else if (result != vk::Result::eSuccess) {
        vulkanLogger.LogError("Failed to present swap chain image!");
    }

    currentFrame = (currentFrame + 1ul) % MAX_FRAMES_IN_FLIGHT;
    return ret;
}

void VulkanEngine::createCommandBuffers()
{
    vk::CommandBufferAllocateInfo allocInfo = {}; allocInfo
        .setCommandPool(theVulkanLayer.commandPool)
        .setLevel(vk::CommandBufferLevel::ePrimary)
        .setCommandBufferCount((uint32_t)MAX_FRAMES_IN_FLIGHT);
    commandBuffers = theVulkanLayer.device.allocateCommandBuffers(allocInfo);
}

void VulkanEngine::createSemaphores()
{
    imageAvailableSemaphores.reserve(MAX_FRAMES_IN_FLIGHT);
    inFlightFences.reserve(MAX_FRAMES_IN_FLIGHT);

    vk::SemaphoreCreateInfo semaphoreInfo = {};

    vk::FenceCreateInfo fenceInfo = {}; fenceInfo
        .setFlags(vk::FenceCreateFlagBits::eSignaled);

    for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
        imageAvailableSemaphores.push_back(theVulkanLayer.device.createSemaphore(semaphoreInfo));
        inFlightFences.push_back(theVulkanLayer.device.createFence(fenceInfo));
    }
}

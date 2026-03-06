#include "rendering/render_context.h"
#include "rendering/common_rendering.h"
#include "asset_managment/mesh_traits.h"
#include "constants.h"
#include "shader_manager.h"

struct RendererDependency;

class Renderer {
public:
    Renderer() : perFrameDscSetBuilder{ descriptorPoolBuilder }, perObjectDscSetBuilder{ descriptorPoolBuilder } {}
    void initialize();
    void destroy();
    void initializeSwapChainDependentComponents();
    void destroySwapChainDependentComponents();
    void render(const RenderContext& ctx);

private:
    struct CachedRenderingData {
        VulkanPipeline pipeline;
        std::array<VertexDataElem, (int)VertexDataElem::_SIZE> bindings;

        CachedRenderingData() {
            bindings.fill(VertexDataElem::_SIZE);
        }

        void destroy()
        {
            if (pipeline) {
                pipeline.destroy();
            }
        }
    };

    struct PerFrameUniformData {
        glm::mat4 v;
        glm::vec4 ambientLight;
    };

    struct alignas(16) PerObjectUniformData {
        glm::mat4 mv, mvp, mvInvT;
        int materialIndex;
    };

    struct alignas(16) PerPointLightUniformData {
        glm::vec4 position, power;
    };

    static const size_t MAX_POINTLIGHT_COUNT = 50;
    struct PointLightsUBO {
        glm::uvec4 lightCount_pad3;
        PerPointLightUniformData pointLights[MAX_POINTLIGHT_COUNT];
    };

    // LABTODO: dir light data types

    void createPipelineLayout();
    void createPipeline(const RendererDependency& dep, CachedRenderingData& comp);
    void createResources();
    void createImages();
    void updateDescriptorSets();

private:
    // LABTODO: post process pipeline
    vk::PipelineLayout pipelineLayout;
    VulkanImage depthImage;
    QueryInfo<2> query;
    std::array<VulkanBuffer, MAX_FRAMES_IN_FLIGHT> pointLightBuffers, perFrameBuffers, perObjectBuffers;
    // LABTODO: dir light buffers
    PerObjectUniformData* aUniformData;
    vk::DeviceSize dynamicAlignment;
    DescriptorPoolBuilder descriptorPoolBuilder;
    DescriptorSetBuilder perFrameDscSetBuilder, perObjectDscSetBuilder;
    std::array<vk::DescriptorSet, MAX_FRAMES_IN_FLIGHT> perFrameDescriptorSets, perObjectDescriptorSets;
    RendererCache<CachedRenderingData> rendererCache;

    // no dynamic rendering
    void createRenderPass();
    void createFramebuffers();
    void startRenderpass(const RenderContext& ctx);
    bool firstSwapChainCreate = true;
    vk::RenderPass renderPass;
    std::vector<vk::Framebuffer> framebuffer;
};

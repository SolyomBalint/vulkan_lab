#ifndef VULKAN_INTRO_SHADER_MANAGER_H
#define VULKAN_INTRO_SHADER_MANAGER_H

#include "vulkan_layer.h"

struct ColouredMesh;

struct ShaderFileData {
    std::string name, path, rawContent, extension;
    std::vector<std::string> dependencies, dependants;
    std::filesystem::file_time_type lastWriteTime;
    std::optional<vk::ShaderStageFlagBits> stage;

    bool compilable() const {
        return extension == ".frag" || extension == ".vert" || extension == ".geom" ||
            extension == ".tesc" || extension == ".tese" || extension == ".comp" ||
            extension == ".rchit" || extension == ".rgen" || extension == ".rmiss" ||
            extension == ".rcall" || extension == ".rint" || extension == ".rahit" ||
            extension == ".mesh" || extension == ".task";
    }
    bool operator<(const ShaderFileData& that) const { return name < that.name; }
};

class ShaderBuilder {
public:
    ShaderBuilder(const ShaderFileData& shader) : shader{ shader } {}
    vk::PipelineShaderStageCreateInfo build() const;

private:
    const ShaderFileData& shader;
};

struct ShaderDependency {
    std::vector<std::string> shaders;
    std::function<void(const std::vector<std::string>&)> handler;

    void call() { handler(shaders); }
};

using ShaderDependencyReference = ShaderDependency*;

struct VulkanPipeline {
    ShaderDependencyReference shaderRef;
    vk::Pipeline pipeline = nullptr;

    VulkanPipeline() = default;
    void destroy();
    operator bool() { return pipeline; }
    VulkanPipeline& operator=(VulkanPipeline&& that) {
        if (pipeline) {
            destroy();
        }
        pipeline = that.pipeline; that.pipeline = nullptr;
        shaderRef = that.shaderRef;
        return *this;
    }
    VulkanPipeline(VulkanPipeline&& that) noexcept {
        pipeline = that.pipeline; that.pipeline = nullptr;
        shaderRef = that.shaderRef;
    }
    VulkanPipeline& operator=(const VulkanPipeline& that) = delete;
    VulkanPipeline(const VulkanPipeline& that) = delete;
    operator vk::Pipeline() { return pipeline; }
};

class ShaderManager {
public:
    static ShaderManager& instance();    

    void initialize();

    void rebuildShaders(bool all = false);
    ShaderBuilder getShader(const std::string& name);
    void destroyShaders(vk::ArrayProxy<vk::PipelineShaderStageCreateInfo> shaders);
    template <typename F, size_t S>
    void addAndExecuteShaderDependency(VulkanPipeline& pipeline, std::array<std::string, S> shaders, F handler)
    {
        shaderDependencies.push_back(std::make_unique<ShaderDependency>( {std::begin(shaders), std::end(shaders) },
            [handler = std::move(handler), this, shaders, &pipeline](const std::vector<std::string>&) {
                std::array<vk::PipelineShaderStageCreateInfo, S> ss;
                for (auto i = 0ull; i < S; ++i) {
                    ss[i] = ShaderBuilder{ find(shaders[i])}.build();
                }
                auto nextPipeline = handler(ss.data());
                destroyShaders(ss);
                if (nextPipeline) {
                    if (pipeline.pipeline) theVulkanLayer.device.destroy(pipeline.pipeline);
                    pipeline.pipeline = nextPipeline;
                }
            })
        );
        pipeline.shaderRef = shaderDependencies.back().get();
        pipeline.shaderRef->call();
    }
    void addAndExecuteShaderDependency(VulkanPipeline& pipeline, std::initializer_list<std::string> shaders, std::function<vk::Pipeline(vk::PipelineShaderStageCreateInfo*)> handler);
    void addAndExecuteShaderDependency(VulkanPipeline& pipeline, const std::vector<std::string>& shaders, std::function<vk::Pipeline(vk::PipelineShaderStageCreateInfo*)> handler);
    void addAndExecuteComputeShaderDependency(VulkanPipeline& pipeline, const std::string& shader, vk::PipelineLayout pipelineLayout, const char* name);
    void removeShaderDependency(ShaderDependencyReference ref);

    void destroyPipelines(std::initializer_list<std::pair<vk::Pipeline&, ShaderDependencyReference&>> list);

private:
    ShaderManager() = default;
    using FnCreatePayloadData = void(*)(void* pData, ColouredMesh& pMesh);
    std::set<ShaderDependencyReference> buildDependencyGraph();
    void compileShaders(bool all = false);
    ShaderFileData& find(const std::string& name);

private:
    std::set<ShaderFileData> shaderFiles;
    std::vector<std::unique_ptr<ShaderDependency>> shaderDependencies;
};

inline auto& theShaderManager = ShaderManager::instance();

#endif//VULKAN_INTRO_SHADER_MANAGER_H

# Lab 2 - Vulkan Introductory Exercises

## Exercise I: Directional Lights

### Theory
A directional light simulates infinitely distant light sources (like the sun). Unlike point lights which radiate from a position with distance-based attenuation, directional lights have a constant direction and their irradiance does not diminish with distance. The shading is computed as `max(dot(N, -L), 0) * lightPower`, where N is the surface normal and L is the light direction.

### Implementation

**Scene side** (`scene/lights.h`, `scene/game_scene.h/cpp`, `application.cpp`):
- Defined `DirectionalLight` struct with `direction`, `color`, `power`, and `name` fields.
- Added `std::vector<DirectionalLight> dirLights` to `GameScene`.
- Created ImGui UI controls using `DragFloat3` for direction/color and `DragFloat` for power.
- Initialized one directional light ("sun") pointing diagonally downward with warm white color.

**Renderer side** (`rendering/renderer.h/cpp`):
- Defined `PerDirLightUniformData` (direction + power as vec4) and `DirLightsUBO` (light count + array of up to 10 dir lights).
- Allocated dir light uniform buffers for each in-flight frame.
- Added descriptor binding 2 (uniform buffer, fragment shader stage) to the per-frame descriptor set.
- Every frame, the directional light data is uploaded to the mapped buffer.

**Shader side** (`forward_common.glsl`, `forward.frag`, `forward_simple.frag`):
- Added `DirLight` struct with `dir` and `power` vec4 fields.
- Declared UBO at set 0, binding 2 containing the directional light array.
- In the `shade()` function, the directional light direction is transformed to view space using the view matrix, and the PBR shading function `viewSpaceShading()` is called with the negated direction and the raw light power (no attenuation).

### What you should see
The Sponza scene and BrainStem model should now be lit by both the existing point light and a new directional "sun" light. Surfaces facing the sun direction will be brighter. The directional light parameters are editable from the "Outliner" ImGui window under "Directional lights".

---

## Exercise II: Normal Mapping

### Theory
Normal mapping adds surface detail without increasing geometric complexity. Instead of using the interpolated per-vertex normal for shading, a normal vector is read from a texture (the normal map). These normals are stored in tangent space -- a coordinate system defined by the surface's Tangent (T), Bitangent (B), and Normal (N) vectors. The TBN matrix transforms tangent-space normals into the shading coordinate system (view space in our case). The bitangent is computed as `B = cross(N, T) * handedness`, where handedness comes from the w-component of the tangent vector (stored as vec4 in GLTF format).

### Implementation

**C++ side** (`rendering/renderer.cpp` - `createPipeline`):
- When the mesh has tangent data (`VertexDataElem::Tangent`), it is added as vertex attribute at location 3 with format `R32G32B32A32Sfloat` (vec4 for handedness in w).
- The tangent binding is recorded in the cache for vertex buffer extraction.

**Vertex shader** (`forward.vert`):
- Added `layout(location = 3) in vec4 inTangent` input.
- Added `layout(location = 3) out mat3 TBN` output (occupies locations 3, 4, 5).
- Computed the TBN matrix: T and N are transformed to view space using `mvInvT`, B is computed as `cross(N, T) * inTangent.w`, and the result is `mat3(T, B, N)`.

**Fragment shader** (`forward.frag`, `material.glsl`):
- In `readMaterial()`, the normal texture is read at index `materialIndex + 1` from the material SSBO. The [0,1] range is remapped to [-1,1] via `sample * 2.0 - 1.0`. If no normal texture exists, `vec3(0)` is stored as a sentinel.
- In `main()`, if a valid normal was read from the texture, it is transformed via `TBN * normal` and normalized. Otherwise, the interpolated vertex normal is used as fallback.

### What you should see
The Sponza scene should show much more surface detail, especially on walls, floors, and pillars. Bricks and stone surfaces should appear to have depth and texture that changes with the lighting angle, even though the geometry is flat. Comparing with the previous state, the surfaces look significantly more detailed and realistic.

---

## Exercise III: Post Processing

### Theory
Post-processing applies image effects to the fully rendered scene. The scene is first rendered to an intermediate (offscreen) texture instead of directly to the swapchain. A second render pass then draws a fullscreen triangle that samples this intermediate texture and applies an image filter. The two passes must be synchronized with a pipeline barrier to ensure the first pass finishes writing before the second pass reads. The intermediate image's layout must also be transitioned between `ColorAttachmentOptimal` (for writing) and `ShaderReadOnlyOptimal` (for sampling).

### Implementation

**Intermediate texture + sampler** (`renderer.cpp` - `createImages`):
- Created an intermediate `VulkanImage` with the swapchain's dimensions and format, with usage flags `ColorAttachment | Sampled`.
- Created a linear sampler with `ClampToEdge` addressing.

**Render target redirection**:
- The main scene render pass now writes to `intermediateImage.view` instead of the swapchain image.

**Descriptor setup** (`renderer.cpp` - `createPipelineLayout`, `updateDescriptorSets`):
- Added binding 3 (CombinedImageSampler, fragment shader) to the per-frame descriptor set layout.
- The intermediate image and sampler are written into this binding.

**Post-process pipeline** (`renderer.cpp` - `initialize`):
- Created using `PipelineBuildHelpers::createFullscreenPipeline()` with the existing `full_screen_quad.vert` and the new `post_process.frag`.
- Registered with the ShaderManager for hot-reload support.

**Post-process shader** (`post_process.frag`):
- Implements a 5x5 Gaussian blur using separable 1D kernel weights `[0.06136, 0.24477, 0.38774, 0.24477, 0.06136]`.
- Samples the scene texture at 25 offset positions around the current UV coordinate.

**Pipeline barriers** (`renderer.cpp` - `render`):
- After the scene render pass ends, an `ImageMemoryBarrier2` transitions the intermediate image:
  - From: `ColorAttachmentOutput` stage, `ColorAttachmentWrite` access, `ColorAttachmentOptimal` layout
  - To: `FragmentShader` stage, `ShaderSampledRead` access, `ShaderReadOnlyOptimal` layout
- After the post-process pass, a second barrier transitions back to `ColorAttachmentOptimal`.

**New render pass**:
- The post-process pass renders to the swapchain image directly, binding the post-process pipeline, the per-frame descriptor set (which contains the intermediate texture at binding 3), and issuing a single `draw(3, 1, 0, 0)` call.

**Query extension**:
- Changed from `QueryInfo<2>` to `QueryInfo<4>` to measure scene time and post-process time separately.
- The ImGui statistics window now shows: Scene time, Post-process time, and Total time.

### What you should see
The entire rendered image should appear slightly blurred (soft/gaussian blur effect). This is most noticeable on sharp edges and text. The blur radius is intentionally subtle (5x5 kernel). The "Renderer" statistics section in the ImGui window should now display separate timing for the scene pass and the post-processing pass.

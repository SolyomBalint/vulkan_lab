cmake_minimum_required(VERSION 3.16) # minimum cmake version
project(vulkanIntro) # define project

set(CMAKE_CXX_STANDARD 23) # c++ standard version

add_compile_definitions(USE_EZPZLOGGER) # enable logging (see thirdparty/EZPZLogger/logger.hpp)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# list all source files in groups
# put additional source files into their relevant group
list(APPEND THIRDPARTY_FILES
    thirdparty/imgui/imgui.cpp
    thirdparty/imgui/imgui_draw.cpp
    thirdparty/imgui/imgui_stdlib.cpp
    thirdparty/imgui/imgui_widgets.cpp
    thirdparty/imgui/imgui_tables.cpp
    thirdparty/imgui/imgui_impl_glfw.cpp
    thirdparty/imgui/imgui_impl_vulkan.cpp
    thirdparty/imgui/imgui_demo.cpp
    thirdparty/imgui/ImGuizmo.cpp
    thirdparty/glfwim/input_manager_impl.cpp
    thirdparty/glfwim/input_manager.cpp)

list(APPEND RESOURCE_FILES
    src/resources/gpu_resources.cpp
    src/resources/resource_manager.cpp)

list(APPEND ASSET_MANAGMENT_FILES
    src/asset_managment/asset_manager.cpp
    src/asset_managment/asset_loader.cpp

    src/asset_managment/texture.cpp
    src/asset_managment/mesh.cpp
    src/asset_managment/material.cpp
    src/asset_managment/coloured_mesh.cpp
    src/asset_managment/composite_mesh.cpp

    src/asset_managment/material_traits.cpp

    src/asset_managment/asset_loader_gltf.cpp
    src/asset_managment/asset_loader_obj.cpp
    src/asset_managment/asset_loader_texture.cpp)

list(APPEND RENDERER_FILES
    src/rendering/renderer.cpp
    src/rendering/renderer_renderpass.cpp
    src/shader_manager.cpp
    src/rendering/common_rendering.cpp)

list(APPEND MAIN_FILES
    src/camera/perspective_camera.cpp
    src/scene/game_scene.cpp
    src/main.cpp
    src/application.cpp
    src/vulkan_engine.cpp
    src/window_manager.cpp
    src/vulkan_layer.cpp
    src/gui/gui_manager.cpp
    src/utility/utility.cpp
    src/gui/imgui_ext.cpp
    src/utility/mmath.cpp)

# create executable
add_executable(${PROJECT_NAME} # name of the executable
    ${MAIN_FILES} ${RENDERER_FILES} # source files (insert from previously defined variable)
    ${ASSET_MANAGMENT_FILES} ${RESOURCE_FILES} ${THIRDPARTY_FILES})
set_source_files_properties(${THIRDPARTY_FILES} PROPERTIES COMPILE_OPTIONS -w) # do not show warnings for files in the thirdparty directory

find_package(Vulkan REQUIRED) # load the Vulkan SDK
find_package(Threads REQUIRED) # load the library to use multiple threads
#find_package(glfw3 3.3 REQUIRED) # uncomment on linux

target_precompile_headers(${PROJECT_NAME} PRIVATE src/pch.h) # set orecompiled header
target_include_directories(${PROJECT_NAME} PRIVATE src) # add a search path for include directives
# files in the SYSTEM search paths will not report warnings
# Vulkan_INCLUDE_DIRS variable was set by the previous find_package directive
target_include_directories(${PROJECT_NAME} SYSTEM PRIVATE thirdparty thirdparty/imgui thirdparty/glm/include ${Vulkan_INCLUDE_DIRS})
target_include_directories(${PROJECT_NAME} SYSTEM PRIVATE thirdparty/glfw/include) # comment on linux

list(APPEND LIBRARIES # collect list of libraries into a variable
    ${Vulkan_LIBRARIES} # this was set by the previous find_package directive
    Threads::Threads
)
if (WIN32)
    list(APPEND LIBRARIES glfw3)
elseif(UNIX)
    list(APPEND LIBRARIES glfw)
endif()

if (WIN32)
    if (MSVC)
        target_link_options(${PROJECT_NAME} PRIVATE /INCREMENTAL /SAFESEH:NO) # enable incremental linking
        target_link_directories(${PROJECT_NAME} PRIVATE thirdparty/glfw/lib-vc2022) # set directory where the linker will search for libraries
    else()
        target_link_directories(${PROJECT_NAME} PRIVATE thirdparty/glfw/lib-mingw-w64)
    endif()
endif()

target_link_libraries(${PROJECT_NAME} ${LIBRARIES}) # link the specified libraries into the application

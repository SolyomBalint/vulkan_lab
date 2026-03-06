#ifndef VULKAN_INTRO_LIGHTS_H
#define VULKAN_INTRO_LIGHTS_H

struct PointLight {
	glm::vec3 position, color;
	float power;
	std::string name;
};

struct AmbientLight {
	glm::vec3 color;
	float power;
};

struct DirectionalLight {
	glm::vec3 direction, color;
	float power;
	std::string name;
};

#endif//VULKAN_INTRO_LIGHTS_H

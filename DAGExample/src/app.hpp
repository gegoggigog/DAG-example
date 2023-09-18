#pragma once
#include <chrono>
#include <glm/glm.hpp>
#include <GLFW/glfw3.h>
#include "utils/view.h"

class Timer {
	using clock_t = std::chrono::high_resolution_clock;
	using timepoint_t = std::chrono::time_point<clock_t>;
	using seconds_ft = std::chrono::duration<float>;
	using milliseconds_t = std::chrono::milliseconds;
	using nanoseconds_t = std::chrono::nanoseconds;
	timepoint_t start_tp;
	timepoint_t end_tp;
	nanoseconds_t diff_ns;

public:
	Timer() : start_tp(clock_t::now()), end_tp(clock_t::now()), diff_ns(0) {}
	float dt_seconds() const { return std::chrono::duration_cast<seconds_ft>(diff_ns).count(); }
	void start() { start_tp = clock_t::now(); }
	void end() {
		end_tp = clock_t::now();
		diff_ns = end_tp - start_tp;
	}
	void reset() {
		start_tp = clock_t::now();
		end_tp = clock_t::now();
		diff_ns = nanoseconds_t{ 0 };
	}
};

struct AppState {
	chag::view camera;
	glm::dvec2 old_mouse{ 0, 0 };
	glm::dvec2 new_mouse{ 0, 0 };
	bool loop = true;
	Timer frame_timer;
	GLFWwindow* window = nullptr;
	inline void handle_events() {
		glfwPollEvents();
		const float dt = frame_timer.dt_seconds();
		const float move_scale_factor{ 1000.0f * dt };
		if (GLFW_PRESS == glfwGetKey(window, GLFW_KEY_ESCAPE)) { glfwSetWindowShouldClose(window, GLFW_TRUE); }
		if (GLFW_PRESS == glfwGetKey(window, GLFW_KEY_W)) { camera.pos -= move_scale_factor * camera.R[2]; }
		if (GLFW_PRESS == glfwGetKey(window, GLFW_KEY_S)) { camera.pos += move_scale_factor * camera.R[2]; }
		if (GLFW_PRESS == glfwGetKey(window, GLFW_KEY_D)) { camera.pos += move_scale_factor * camera.R[0]; }
		if (GLFW_PRESS == glfwGetKey(window, GLFW_KEY_A)) { camera.pos -= move_scale_factor * camera.R[0]; }
		if (GLFW_PRESS == glfwGetKey(window, GLFW_KEY_E)) { camera.pos += move_scale_factor * camera.R[1]; }
		if (GLFW_PRESS == glfwGetKey(window, GLFW_KEY_Q)) { camera.pos -= move_scale_factor * camera.R[1]; }

		const bool ld = GLFW_PRESS == glfwGetMouseButton(window, 0);
		const bool md = GLFW_PRESS == glfwGetMouseButton(window, 1);
		const bool rd = GLFW_PRESS == glfwGetMouseButton(window, 2);

		glfwGetCursorPos(window, &new_mouse.x, &new_mouse.y);

		glm::vec2 delta = glm::vec2(new_mouse) - glm::vec2(old_mouse);

		const float mouse_scale_factor{ dt };
		if (ld && rd) {
			camera.pos += delta.y * mouse_scale_factor * camera.R[2];
		}
		else if (ld) {
			camera.pitch(-delta.y * mouse_scale_factor);
			camera.yaw(-delta.x * mouse_scale_factor);
		}
		else if (rd) {
			camera.roll(-delta.x * mouse_scale_factor);
		}
		else if (md) {
			camera.pos -= delta.y * mouse_scale_factor * camera.R[1];
			camera.pos += delta.x * mouse_scale_factor * camera.R[0];
		}
		old_mouse = new_mouse;
	}
};
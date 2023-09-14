#include "Aabb.h"
#include <limits>

namespace chag
{
	const glm::vec3 Aabb::getCentre() const {
		return (min + max) * 0.5f;
	}

	const glm::vec3 Aabb::getHalfSize() const {
		return (max - min) * 0.5f;
	}

	float Aabb::getVolume() const {
		glm::vec3 d = max - min;
		return d.x * d.y * d.z;
	}

	float Aabb::getArea() const {
		glm::vec3 d = max - min;
		return d.x * d.y * 2.0f + d.x * d.z * 2.0f + d.z * d.y * 2.0f;
	}

	Aabb make_aabb(const glm::vec3 &min, const glm::vec3 &max) {
		return { min, max };
	}

	Aabb make_aabb(const glm::vec3& position, const float radius) {
		return { position - radius, position + radius }; 
	}

	Aabb combine(const Aabb& a, const Aabb& b) {
		return { glm::min(a.min, b.min), glm::max(a.max, b.max) }; 
	}

	Aabb combine(const Aabb& a, const glm::vec3& pt) {
		return { glm::min(a.min, pt),    glm::max(a.max, pt) };
	}

	Aabb make_inverse_extreme_aabb() {
		return { 
			glm::vec3{std::numeric_limits<float>::max()},
			glm::vec3{std::numeric_limits<float>::lowest()}
		};
	}

	Aabb make_aabb(const glm::vec3 *positions, const size_t numPositions) {
		Aabb result = make_inverse_extreme_aabb();
		for (size_t i = 0; i < numPositions; ++i) {
			result = combine(result, positions[i]);
		}
		return result;
	}

	bool overlaps(const Aabb& a, const Aabb& b)
	{
		return a.max.x > b.min.x && a.min.x < b.max.x
			&& a.max.y > b.min.y && a.min.y < b.max.y
			&& a.max.z > b.min.z && a.min.z < b.max.z;

	}
} // namespace chag

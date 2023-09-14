#include "Aabb.h"
#include <limits>
//#include <float.h>

using vec3 = glm::vec3;
namespace chag
{
	const vec3 Aabb::getCentre() const {
		return (min + max) * 0.5f;
	}

	const vec3 Aabb::getHalfSize() const {
		return (max - min) * 0.5f;
	}

	float Aabb::getVolume() const {
		vec3 d = max - min;
		return d.x * d.y * d.z;
	}

	float Aabb::getArea() const {
		vec3 d = max - min;
		return d.x * d.y * 2.0f + d.x * d.z * 2.0f + d.z * d.y * 2.0f;
	}

	Aabb make_aabb(const vec3 &min, const vec3 &max) {
		return { min, max };
	}

	Aabb make_aabb(const vec3& position, const float radius) {
		return { position - radius, position + radius }; 
	}

	Aabb combine(const Aabb& a, const Aabb& b) {
		return { glm::min(a.min, b.min), glm::max(a.max, b.max) }; 
	}

	Aabb combine(const Aabb& a, const vec3& pt) {
		return { glm::min(a.min, pt),    glm::max(a.max, pt) };
	}

	Aabb make_inverse_extreme_aabb() {
		return { 
			vec3{std::numeric_limits<float>::max()},
			vec3{std::numeric_limits<float>::lowest()}
		};
	}

	Aabb make_aabb(const vec3 *positions, const size_t numPositions) {
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

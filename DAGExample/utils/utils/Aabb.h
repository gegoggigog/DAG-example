#pragma once
#include <glm/glm.hpp>
namespace chag
{
	struct Aabb
	{
		glm::vec3 min;
		glm::vec3 max;
		const glm::vec3 getCentre() const;
		const glm::vec3 getHalfSize() const;
		float getVolume() const;
		float getArea() const;
	};

	Aabb combine(const Aabb &a, const Aabb &b);
	Aabb combine(const Aabb &a, const glm::vec3 &pt);
	Aabb make_inverse_extreme_aabb();
	Aabb make_aabb(const glm::vec3 &min,       const glm::vec3 &max);
	Aabb make_aabb(const glm::vec3 &position,  const float     radius);
	Aabb make_aabb(const glm::vec3 *positions, const size_t    numPositions);
	bool overlaps(const Aabb &a, const Aabb &b);

	// Intersect with a ray (from pbrt)
	//inline bool intersect(const Aabb &a, const ray &r, float *hitt0 = nullptr, float *hitt1 = nullptr)
	//{
	//	float t0 = r.mint, t1 = r.maxt;
	//	for(int i=0; i<3; i++){
	//		float invRayDir = 1.0f / r.d[i];
	//		float tNear = (a.min[i] - r.o[i]) * invRayDir; 
	//		float tFar =  (a.max[i] - r.o[i]) * invRayDir; 
	//		if(tNear > tFar) { //swap(tNear, tFar); 
	//			float temp = tNear; 
	//			tNear = tFar; 
	//			tFar = temp; 
	//		}
	//		t0 = tNear > t0 ? tNear : t0; 
	//		t1 = tFar < t1 ? tFar : t1; 
	//		if(t0 > t1) return false; 
	//	}
	//	if(hitt0) *hitt0 = t0;
	//	if(hitt1) *hitt1 = t1;
	//	return true; 
	//}



} // namespace chag

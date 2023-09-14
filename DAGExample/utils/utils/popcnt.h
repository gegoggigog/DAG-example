#pragma once
#include <bitset>
#include <numeric>
template <typename T>
inline unsigned popcnt(T v)
{
	return static_cast<unsigned>(std::bitset<std::numeric_limits<T>::digits>(v).count());
}


//#ifdef _MSC_VER
//#include <intrin.h>
//#else
//#include <popcntintrin.h>
//#endif
//
//inline unsigned int popcnt(unsigned int in)
//{
//#ifdef _MSC_VER
//	return __popcnt(in);
//#else
//#if 0
//	unsigned int r = 0;
//	asm("popcnt %1,%0" : "=r"(r) : "r"(in));
//	return r;
//#else
//	__builtin_popcount(foo());
//#endif
//#endif
//}
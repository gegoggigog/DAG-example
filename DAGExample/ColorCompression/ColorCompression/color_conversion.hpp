#pragma once
#include <algorithm>
#include <glm/glm.hpp>
#include "color_layout.hpp"
namespace ours_varbit {
    using glm::vec3;
    using std::max;
    using std::min;
    // clang-format off
    ///////////////____R____///////////////
    vec3 r4_to_float3(uint32_t rgb)
    {
        return vec3(
            float((rgb >> 0) & 0xF) / 15.0f,
            0.0f,
            0.0f
        );
    }

    vec3 r8_to_float3(uint32_t rgb)
    {
        return vec3(
            float((rgb >> 0) & 0xFF) / 255.0f,
            0.0f,
            0.0f
        );
    }

    vec3 r16_to_float3(uint32_t rgb)
    {
        return vec3(
            float((rgb >> 0) & 0xFFFF) / 65535.0f,
            0.0f,
            0.0f
        );
    }
    ///////////////____RG____///////////////
    vec3 rg88_to_float3(uint32_t rgb)
    {
        return vec3(
            ((rgb >> 0) & 0xFF) / 255.0f,
            ((rgb >> 8) & 0xFF) / 255.0f,
            0.0f
        );
    }

    vec3 rg1616_to_float3(uint32_t rgb)
    {
        return vec3(
            ((rgb >> 0) & 0xFFFF) / 65535.0f,
            ((rgb >> 16) & 0xFFFF) / 65535.0f,
            0.0f
        );
    }
    ///////////////____RGB____///////////////
    vec3 rgb888_to_float3(uint32_t rgb)
    {
        return vec3(
            ((rgb >> 0) & 0xFF) / 255.0f,
            ((rgb >> 8) & 0xFF) / 255.0f,
            ((rgb >> 16) & 0xFF) / 255.0f
        );
    }

    vec3 rgb101210_to_float3(uint32_t rgb)
    {
        return vec3(
            ((rgb >> 0) & 0x3FF) / 1023.0f,
            ((rgb >> 10) & 0xFFF) / 4095.0f,
            ((rgb >> 22) & 0x3FF) / 1023.0f
        );
    }

    vec3 rgb565_to_float3(uint32_t rgb)
    {
        return vec3(
            ((rgb >> 0) & 0x1F) / 31.0f,
            ((rgb >> 5) & 0x3F) / 63.0f,
            ((rgb >> 11) & 0x1F) / 31.0f
        );
    }

    ///////////////____R____///////////////
    uint32_t float3_to_r4(vec3 c)
    {
        float R = min(1.0f, max(0.0f, c.x));
        return
            (uint32_t(round(R * 15.0f)) << 0);
    }

    uint32_t float3_to_r8(vec3 c)
    {
        float R = min(1.0f, max(0.0f, c.x));
        return
            (uint32_t(round(R * 255.0f)) << 0);
    }

    uint32_t float3_to_r16(vec3 c)
    {
        float R = min(1.0f, max(0.0f, c.x));
        return
            (uint32_t(round(R * 65535.0f)) << 0);
    }
    ///////////////____RG____///////////////
    uint32_t float3_to_rg88(vec3 c)
    {
        float R = min(1.0f, max(0.0f, c.x));
        float G = min(1.0f, max(0.0f, c.y));
        return
            (uint32_t(round(R * 255.0f)) << 0) |
            (uint32_t(round(G * 255.0f)) << 8);
    }

    uint32_t float3_to_rg1616(vec3 c)
    {
        float R = min(1.0f, max(0.0f, c.x));
        float G = min(1.0f, max(0.0f, c.y));
        return
            (uint32_t(round(R * 65535.0f)) << 0) |
            (uint32_t(round(G * 65535.0f)) << 16);
    }
    ///////////////____RGB____///////////////
    uint32_t float3_to_rgb888(vec3 c)
    {
        float R = min(1.0f, max(0.0f, c.x));
        float G = min(1.0f, max(0.0f, c.y));
        float B = min(1.0f, max(0.0f, c.z));
        return
            (uint32_t(round(R * 255.0f)) << 0) |
            (uint32_t(round(G * 255.0f)) << 8) |
            (uint32_t(round(B * 255.0f)) << 16);
    }

    uint32_t float3_to_rgb101210(vec3 c)
    {
        float R = min(1.0f, max(0.0f, c.x));
        float G = min(1.0f, max(0.0f, c.y));
        float B = min(1.0f, max(0.0f, c.z));
        return
            (uint32_t(round(R * 1023.0f)) << 0) |
            (uint32_t(round(G * 4095.0f)) << 10) |
            (uint32_t(round(B * 1023.0f)) << 22);
    }

    uint32_t float3_to_rgb565(vec3 c)
    {
        float R = min(1.0f, max(0.0f, c.x));
        float G = min(1.0f, max(0.0f, c.y));
        float B = min(1.0f, max(0.0f, c.z));
        return
            (uint32_t(round(R * 31.0f)) << 0) |
            (uint32_t(round(G * 63.0f)) << 5) |
            (uint32_t(round(B * 31.0f)) << 11);
    }

    uint32_t float3_to_rgbxxx(vec3 c, ColorLayout layout)
    {
        switch (layout)
        {
            case R_4:          return float3_to_r4(c);
            case R_8:          return float3_to_r8(c);
            case R_16:         return float3_to_r16(c);
            case RG_8_8:       return float3_to_rg88(c);
            case RG_16_16:     return float3_to_rg1616(c);
            case RGB_8_8_8:    return float3_to_rgb888(c);
            case RGB_10_12_10: return float3_to_rgb101210(c);
            case RGB_5_6_5:    return float3_to_rgb565(c);
            default: break;
        }
        return 0;
    }

    vec3 rgbxxx_to_float3(uint32_t rgb, ColorLayout layout)
    {
        switch (layout)
        {
            case R_4:          return r4_to_float3(rgb);
            case R_8:          return r8_to_float3(rgb);
            case R_16:         return r16_to_float3(rgb);
            case RG_8_8:       return rg88_to_float3(rgb);
            case RG_16_16:     return rg1616_to_float3(rgb);
            case RGB_8_8_8:    return rgb888_to_float3(rgb);
            case RGB_10_12_10: return rgb101210_to_float3(rgb);
            case RGB_5_6_5:    return rgb565_to_float3(rgb);
            default: break;
        }
        return vec3(0.f, 0.f, 0.f);
    }

    vec3 minmaxCorrectedColor(const vec3& c, ColorLayout layout)
    {
        return rgbxxx_to_float3(float3_to_rgbxxx(c, layout), layout);
    }

    vec3 minmaxSingleCorrectedColor(const vec3& c, ColorLayout layout)
    {
        ColorLayout single_color_layout;
        switch (layout)
        {
            case R_4:       single_color_layout = R_8;          break;
            case R_8:       single_color_layout = R_16;         break;
            case RG_8_8:    single_color_layout = RG_16_16;     break;
            case RGB_5_6_5: single_color_layout = RGB_10_12_10; break;
            default:        single_color_layout = NONE;         break;
        }
        return minmaxCorrectedColor(c, single_color_layout);
    }
    // clang-format on
}

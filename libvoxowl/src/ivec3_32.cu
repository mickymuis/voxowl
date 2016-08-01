#include "ivec3_32.h"
#include "libdivide.h"
#include <emmintrin.h>

VOXOWL_HOST_AND_DEVICE 
ivec3_32_t 
ivec3_add32( ivec3_32_t a, ivec3_32_t b ) {
    ivec3_32_t r;
    r.x =a.x + b.x;
    r.y =a.y + b.y;
    r.z =a.z + b.z;
    return r;
}

VOXOWL_HOST_AND_DEVICE 
ivec3_32_t 
ivec3_sub32( ivec3_32_t a, ivec3_32_t b ) {
    ivec3_32_t r;
    r.x =a.x - b.x;
    r.y =a.y - b.y;
    r.z =a.z - b.z;
    return r;
}

VOXOWL_HOST_AND_DEVICE 
ivec3_32_t 
ivec3_mult32( ivec3_32_t a, ivec3_32_t b ) {
    ivec3_32_t r;
    r.x =a.x * b.x;
    r.y =a.y * b.y;
    r.z =a.z * b.z;
    return r;
}

VOXOWL_HOST_AND_DEVICE 
ivec3_32_t 
ivec3_div32( ivec3_32_t a, ivec3_32_t b ) {
    ivec3_32_t r;
    r.x =a.x / b.x;
    r.y =a.y / b.y;
    r.z =a.z / b.z;
    return r;
}

// Manipulations by scalar

VOXOWL_HOST_AND_DEVICE 
ivec3_32_t 
ivec3_mults32( ivec3_32_t a, int32_t i ) {
    ivec3_32_t r;
    r.x =a.x * i;
    r.y =a.y * i;
    r.z =a.z * i;
    return r;
}

VOXOWL_HOST_AND_DEVICE 
ivec3_32_t 
ivec3_divs32( ivec3_32_t a, int32_t i ) {
    ivec3_32_t r;
    r.x =a.x / i;
    r.y =a.y / i;
    r.z =a.z / i;
    return r;
}

VOXOWL_HOST_AND_DEVICE 
ivec3_32_t 
ivec3_fast_divs32( ivec3_32_t a, struct libdivide::libdivide_s32_branchfree_t* div ) {
    ivec3_32_t r;
    r.x =libdivide::libdivide_s32_branchfree_do( a.x, div );
    r.y =libdivide::libdivide_s32_branchfree_do( a.y, div );
    r.z =libdivide::libdivide_s32_branchfree_do( a.z, div );
    return r;
}

VOXOWL_HOST_AND_DEVICE 
ivec3_32_t ivec3_sar32( ivec3_32_t a ) {
    ivec3_32_t r;
    r.x =a.x >> 1;
    r.y =a.y >> 1;
    r.z =a.z >> 1;
    return r;
}

VOXOWL_HOST_AND_DEVICE 
ivec3_32_t ivec3_sal32( ivec3_32_t a ) {
    ivec3_32_t r;
    r.x =a.x << 1;
    r.y =a.y << 1;
    r.z =a.z << 1;
    return r;
}

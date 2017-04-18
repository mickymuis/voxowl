#pragma once

#include "platform.h"
#include <glm/vec3.hpp>
#include <inttypes.h>
//#include "libdivide.h"

namespace {
namespace libdivide {
    struct libdivide_s32_branchfree_t;
}; };

typedef struct { 
    int32_t x;
    int32_t y;
    int32_t z;
} ivec3_32_t;


SVMM_HOST_AND_DEVICE inline ivec3_32_t ivec3_32( const uint16_t x, uint16_t y, uint16_t z ) { ivec3_32_t v = { x, y, z }; return v; }
SVMM_HOST_AND_DEVICE inline ivec3_32_t ivec3_32( uint16_t n ) { ivec3_32_t v = { n, n, n }; return v; }
SVMM_HOST_AND_DEVICE inline ivec3_32_t ivec3_32( glm::ivec3 gv ) { ivec3_32_t v = { (uint16_t)gv.x, (uint16_t)gv.y, (uint16_t)gv.z }; return v; }
SVMM_HOST_AND_DEVICE inline glm::ivec3 glm_ivec3_32( ivec3_32_t v ) { return glm::ivec3( v.x, v.y, v.z ); }

// Straight vector manipulations

SVMM_HOST_AND_DEVICE ivec3_32_t ivec3_add32( ivec3_32_t a, ivec3_32_t b );
SVMM_HOST_AND_DEVICE ivec3_32_t ivec3_sub32( ivec3_32_t a, ivec3_32_t b );
SVMM_HOST_AND_DEVICE ivec3_32_t ivec3_mult32( ivec3_32_t a, ivec3_32_t b );
SVMM_HOST_AND_DEVICE ivec3_32_t ivec3_div32( ivec3_32_t a, ivec3_32_t b );

// Manipulations by scalar

SVMM_HOST_AND_DEVICE ivec3_32_t ivec3_mults32( ivec3_32_t a, int32_t i );
SVMM_HOST_AND_DEVICE ivec3_32_t ivec3_divs32( ivec3_32_t a, int32_t i );
SVMM_HOST_AND_DEVICE ivec3_32_t ivec3_fast_divs32( ivec3_32_t a, struct libdivide::libdivide_s32_branchfree_t* div );
SVMM_HOST_AND_DEVICE ivec3_32_t ivec3_sar32( ivec3_32_t a );
SVMM_HOST_AND_DEVICE ivec3_32_t ivec3_sal32( ivec3_32_t a );

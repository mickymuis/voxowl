/*
 *  Z a n d b a k 
 *  met vormpjes!
 */
#pragma once

#define GLM_SWIZZLE

#include "glm/glm.hpp"
#include "glm/vec2.hpp"
#include "glm/vec3.hpp"
#include "glm/vec4.hpp"

#include "main.h"

typedef struct {
    glm::vec3 origin;
    glm::vec3 direction;
} ray_t;

typedef struct {
    glm::vec3 min;
    glm::vec3 max;
} box_t;

typedef struct {
    glm::vec4 color; //rgba
    glm::vec3 position;
    glm::vec3 position_vs; // position in continuous voxel space
    glm::vec3 normal;
} fragment_t;

/* Produces an Axis Aligned Bounding Box (AABB) from a given size vector
 *    - The resulting box is axis aligned, unit sized and centered around (0,0,0)
 *       - The ratios width/height/depth match the given size vector 
 *          - The largest direction in the size vector will measure 1.0 in the box */
box_t volumeSizeToAABB( glm::ivec3 size );

/* Given a ray r and an AABB b, return true if r intersects b in any point
 *    Additionally, the entry and exit constants tmin and tmax are set */
bool rayAABBIntersect( const ray_t &r, const box_t& b, double& tmin, double& tmax );

int svmmRaycast( volume_t* v, const ray_t& r, int steps, bool );

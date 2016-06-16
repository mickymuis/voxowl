#pragma once

#include "renderer.h"
#include "platform.h"

#include "glm/mat4x4.hpp"
#include "glm/vec3.hpp"

typedef struct {
    glm::vec3 origin;
    glm::vec3 direction;
} ray_t;

typedef struct {
    glm::vec3 min;
    glm::vec3 max;
} box_t;

typedef struct {
    glm::mat4 matModelView;
    glm::mat4 matInvModelView;
    glm::vec3 upperLeftNormal;
    glm::vec3 upperRightNormal;
    glm::vec3 lowerLeftNormal;
    glm::vec3 lowerRightNormal;
    glm::vec3 leftNormalYDelta;
    glm::vec3 rightNormalYDelta;
    glm::vec3 origin;
    float invHeight;
    float invWidth;
} raycastInfo_t;

/*typedef struct {
    glm::ivec3 size;
    glm::mat4 matModel;
} volumeInfo_t;

typedef struct {
    int width, height;
    int aaXSamples, aaYSamples;
} framebufferInfo_t;*/

/* Produces an Axis Aligned Bounding Box (AABB) from a given size vector
   - The resulting box is axis aligned, unit sized and centered around (0,0,0)
   - The ratios width/height/depth match the given size vector 
   - The largest direction in the size vector will measure 1.0 in the box */
VOXOWL_HOST_AND_DEVICE box_t volumeSizeToAABB( glm::ivec3 size );

/* Given a ray r and an AABB b, return true if r intersects b in any point
   Additionally, the entry and exit constants tmin and tmax are set */
VOXOWL_HOST_AND_DEVICE bool rayAABBIntersect( const ray_t &r, const box_t& b, double& tmin, double& tmax );

/* Set projection planes and view matrix using the given model, view and projection matrices/
   x_size and y_size give the screen's width and height in pixels, respectively */
void raycastSetMatrices( raycastInfo_t* raycast_info, glm::mat4 mat_model, glm::mat4 mat_view, glm::mat4 mat_proj, int x_size, int y_size );

/*class RaycasterCPU : public Renderer {
    public:

};*/

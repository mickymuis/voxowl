#include "raycast.h"

/* Produces an Axis Aligned Bounding Box (AABB) from a given size vector
   - The resulting box is axis aligned, unit sized and centered around (0,0,0)
   - The ratios width/height/depth match the given size vector 
   - The largest direction in the size vector will measure 1.0 in the box */
VOXOWL_HOST_AND_DEVICE
box_t
volumeSizeToAABB( const glm::ivec3 &size ) {
    int largest = max( size.x, max( size.y, size.z ) );
    largest *=2;
    box_t b;
    b.min = -glm::vec3 (
        (float)size.x / (float)largest,
        (float)size.y / (float)largest,
        (float)size.z / (float)largest );
    b.max =-b.min - glm::vec3(0.0) ;
    return b;
}

/* Given a ray r and an AABB b, return true if r intersects b in any point
   Additionally, the entry and exit constants tmin and tmax are set */
VOXOWL_HOST_AND_DEVICE
bool
rayAABBIntersect( const ray_t &r, const box_t& b, float& tmin, float& tmax ) {
    glm::vec3 n_inv = glm::vec3( 
        1.f / r.direction.x,
        1.f / r.direction.y,
        1.f / r.direction.z );
    float tx1 = (b.min.x - r.origin.x)*n_inv.x;
    float tx2 = (b.max.x - r.origin.x)*n_inv.x;
 
    tmin = fminf(tx1, tx2);
    tmax = fmaxf(tx1, tx2);
 
    float ty1 = (b.min.y - r.origin.y)*n_inv.y;
    float ty2 = (b.max.y - r.origin.y)*n_inv.y;
 
    tmin = fmaxf(tmin, fminf(ty1, ty2));
    tmax = fminf(tmax, fmaxf(ty1, ty2));
    
    float tz1 = (b.min.z - r.origin.z)*n_inv.z;
    float tz2 = (b.max.z - r.origin.z)*n_inv.z;
 
    tmin = fmaxf(tmin, fminf(tz1, tz2));
    tmax = fminf(tmax, fmaxf(tz1, tz2));
 
    return tmax >= tmin;
}

VOXOWL_HOST_AND_DEVICE
void
blendF2B_fast( const glm::vec4 &src, glm::vec4 &dst ) {
    dst.r = dst.a*(src.r * src.a) + dst.r;
    dst.g = dst.a*(src.g * src.a) + dst.g;
    dst.b = dst.a*(src.b * src.a) + dst.b;
    dst.a = (1.f - src.a) * dst.a;
}

/* Set projection planes and view matrix using the given model, view and projection matrices 
   x_size and y_size give the screen's width and height in pixels, respectively */
void 
raycastSetMatrices( raycastInfo_t* raycast_info, glm::mat4 mat_model, glm::mat4 mat_view, glm::mat4 mat_proj, int x_size, int y_size ) {
    // We assume a symmetric projection matrix
    const float near = ( 2.0f * mat_proj[3][2] ) / ( 2.0f * mat_proj[2][2] - 2.0f );
    const float far = ( (mat_proj[2][2] - 1.0f) * near ) / ( mat_proj[2][2] + 1.0 );
//    const float near =1.f;
//    const float far =100.f;
    const float right =near / mat_proj[0][0];
    const float top =near / mat_proj[1][1];
    const float left =-right, bottom =-top;

    // Setup the arguments to the raycaster
    raycast_info->upperLeftNormal =glm::normalize( glm::vec3( left, top, -near ) );
    raycast_info->upperRightNormal =glm::normalize( glm::vec3( right, top, -near ) );
    glm::vec3 lowerLeftNormal =glm::normalize( glm::vec3( left, bottom, -near ) );
    glm::vec3 lowerRightNormal =glm::normalize( glm::vec3( right, bottom, -near ) );
    

    // Calculate the ray-normal interpolation constants
    float invHeight = 1.f / (float)y_size;
    raycast_info->invWidth = 1.f / (float)x_size;

    raycast_info->leftNormalYDelta = (lowerLeftNormal - raycast_info->upperLeftNormal) * invHeight;
    raycast_info->rightNormalYDelta =(lowerRightNormal - raycast_info->upperRightNormal) * invHeight;

    // Compute the fragment width per unit in the Z-direction (assuming square pixels!)
    raycast_info->fragmentWidthWorldDelta =glm::length(raycast_info->leftNormalYDelta) / near;

    // Compute the inverse modelview matrix and use it the compute the ray origin
    glm::mat4 mat_modelview =mat_view * mat_model;
    glm::mat4 mat_inv_modelview =glm::inverse( mat_modelview );
    
    raycast_info->origin = glm::vec3( mat_inv_modelview * glm::vec4(0,0,0,1) );
    raycast_info->matModelView =mat_modelview;
    raycast_info->matInvModelView = mat_inv_modelview;

    // TEST
    glm::vec3 vnear =glm::vec3( mat_inv_modelview * glm::vec4(0,0,-near,1) );
    raycast_info->fragmentWidthDelta =glm::length(raycast_info->leftNormalYDelta) / glm::distance( raycast_info->origin, vnear );
    
}

#include "raycast.h"

/* Produces an Axis Aligned Bounding Box (AABB) from a given size vector
   - The resulting box is axis aligned, unit sized and centered around (0,0,0)
   - The ratios width/height/depth match the given size vector 
   - The largest direction in the size vector will measure 1.0 in the box */
VOXOWL_HOST_AND_DEVICE
box_t
volumeSizeToAABB( glm::ivec3 size ) {
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
rayAABBIntersect( const ray_t &r, const box_t& b, double& tmin, double& tmax ) {
    glm::vec3 n_inv = glm::vec3( 
        1.f / r.direction.x,
        1.f / r.direction.y,
        1.f / r.direction.z );
    double tx1 = (b.min.x - r.origin.x)*n_inv.x;
    double tx2 = (b.max.x - r.origin.x)*n_inv.x;
 
    tmin = min(tx1, tx2);
    tmax = max(tx1, tx2);
 
    double ty1 = (b.min.y - r.origin.y)*n_inv.y;
    double ty2 = (b.max.y - r.origin.y)*n_inv.y;
 
    tmin = max(tmin, min(ty1, ty2));
    tmax = min(tmax, max(ty1, ty2));
    
    double tz1 = (b.min.z - r.origin.z)*n_inv.z;
    double tz2 = (b.max.z - r.origin.z)*n_inv.z;
 
    tmin = max(tmin, min(tz1, tz2));
    tmax = min(tmax, max(tz1, tz2));
 
    return tmax >= tmin;
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
    raycast_info->lowerLeftNormal =glm::normalize( glm::vec3( left, bottom, -near ) );
    raycast_info->lowerRightNormal =glm::normalize( glm::vec3( right, bottom, -near ) );
    

    // Calculate the ray-normal interpolation constants
    raycast_info->invHeight = 1.f / (float)y_size;
    raycast_info->invWidth = 1.f / (float)x_size;

    raycast_info->leftNormalYDelta = (raycast_info->lowerLeftNormal - raycast_info->upperLeftNormal) * raycast_info->invHeight;
    raycast_info->rightNormalYDelta =(raycast_info->lowerRightNormal - raycast_info->upperRightNormal) * raycast_info->invHeight;

    // Compute the inverse modelview matrix and use it the compute the ray origin
    glm::mat4 mat_modelview =mat_view * mat_model;
    glm::mat4 mat_inv_modelview =glm::inverse( mat_modelview );
    
    raycast_info->origin = glm::vec3( mat_inv_modelview * glm::vec4(0,0,0,1) );
    raycast_info->matModelView =mat_modelview;
    raycast_info->matInvModelView = mat_inv_modelview;
}

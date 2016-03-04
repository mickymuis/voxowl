#include <stdio.h>
#include <stdint.h>
#include "platform.h"
#include "bmp.h"
#include <vector>
#include <fstream>
#include <glm/mat4x4.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtx/transform.hpp>

typedef struct {
    glm::vec3 origin;
    glm::vec3 direction;
} ray;

typedef struct {
    glm::vec3 min;
    glm::vec3 max;
} box;

typedef glm::vec4 voxel;

__device__ __host__
voxel* 
voxel3d( voxel* V, glm::ivec3 size, glm::ivec3 pos ) {
    return (V + size.z * size.y * pos.x + size.z * pos.y + pos.z );
}

__device__ __host__
void
packRGBA32( uint32_t* rgba, glm::vec4 v ) {
    *rgba =
        ((uint32_t)(v.r * 255) << 24) |
        ((uint32_t)(v.g * 255) << 16) |
        ((uint32_t)(v.b * 255) << 8) |
        (uint32_t)(v.a * 255);
}

void menger( voxel* V, glm::ivec3 size, glm::ivec3 cube, glm::ivec3 offset ) {
    uint32_t step = cube.x / 3;
    if( step < 1 )
        return;
    for( int x =0; x < 3; x++ )
        for( int y =0; y < 3; y++ )
            for( int z =0; z < 3; z++ ) {
                glm::ivec3 offs = offset + glm::ivec3( x * step, y * step, z * step );
                glm::ivec3 new_cube( step, step, step );
                // middle element
                if( ( z == 1 && ( y == 1 || x == 1 ) )
                    || ( x == 1 && y == 1 ) ) {
                    for( uint32_t i = offs.x; i < offs.x + step; i++ )
                        for( uint32_t j = offs.y; j < offs.y + step; j++ )
                            for( uint32_t k = offs.z; k < offs.z + step; k++ )
                                voxel3d( V, size, glm::ivec3( i, j, k ))->a =0.0f;
                }
                // corner element, expand recursively
                else
                    menger( V, size, new_cube, offs );
            }
}

void 
makeSponge( voxel* V, glm::ivec3 size ) {

    for( uint32_t i=0; i < size.x * size.y * size.z; i++ ) {
            V[i].r = 0.1f;
            V[i].g = 0.8f;
            V[i].b = 0.7f;
            V[i].a = 1.0f;
    }
    menger( V, size, size, glm::ivec3(0) );
}

__device__ __host__
glm::vec3
voxelIndexToCoord( glm::ivec3 size, glm::ivec3 pos ) {
    int largest = size.x > size.y ? size.x : size.y;
    largest = largest > size.z ? largest : size.z;
    return glm::vec3( 
        (float)pos.x/(float)largest,
        (float)pos.y/(float)largest,
        (float)pos.z/(float)largest );
}

box
voxelSizeToAABB( glm::ivec3 size ) {
    int largest = size.x > size.y ? size.x : size.y;
    largest = largest > size.z ? largest : size.z;
    largest *=2;
    box b;
    b.max = glm::vec3 (
        (float)size.x / (float)largest,
        (float)size.y / (float)largest,
        (float)size.z / (float)largest );
    b.min =-b.max;
    return b;
}

bool
rayAABBIntersect( const ray &r, const box& b ) {
    glm::vec3 n_inv = glm::vec3( 
        1.f / r.direction.x,
        1.f / r.direction.y,
        1.f / r.direction.z );
    double tx1 = (b.min.x - r.origin.x)*n_inv.x;
    double tx2 = (b.max.x - r.origin.x)*n_inv.x;
 
    double tmin = min(tx1, tx2);
    double tmax = max(tx1, tx2);
 
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

glm::vec4
raycast( voxel* V, glm::ivec3 size, const ray& r ) {
    if( rayAABBIntersect( r, voxelSizeToAABB( size ) ) )
        return glm::vec4( 1.0, 0.0, 0.0, 1.0 );
    return glm::vec4( 0.0, 0.0, 0.3, 1.0 );
}

int 
main( int argc, char **argv ) {

    // Make a test sponge V of size L
    const uint32_t L =81;
    const int width =800;
    const int height =600;
    const glm::ivec3 size( L, L, L );
    voxel* V = (voxel*)malloc( sizeof( voxel ) * L * L * L );
    makeSponge( V, size );


    // Setup the camera
    float near =0.1f;
    glm::vec3 viewpoint( 0.4f, 0.5f, -2.f );
    glm::vec3 target(0);
    glm::vec3 up( 0, 1, 0 );
    glm::mat4 mat_view =glm::lookAt( viewpoint, target, up );

    // We assume a symmetric projection matrix
    glm::mat4 mat_proj =glm::perspective( 40.f, (float)width/height, near, 200.f );
    const float right =near / mat_proj[0][0];
    const float top =near / mat_proj[1][1];
    const float left =-right, bottom =-top;
    glm::vec3 upperLeftNormal =glm::normalize( glm::vec3( left, top, near ) );
    glm::vec3 upperRightNormal =glm::normalize( glm::vec3( right, top, near ) );
    glm::vec3 lowerLeftNormal =glm::normalize( glm::vec3( left, bottom, near ) );
    glm::vec3 lowerRightNormal =glm::normalize( glm::vec3( right, bottom, near ) );

    // Calculate the ray-normal interpolation constants
    const float invHeight = 1.f / (float)height;
    const float invWidth = 1.f / (float)width;

    glm::vec3 leftNormalYDelta = (lowerLeftNormal - upperLeftNormal) * invHeight;
    glm::vec3 rightNormalYDelta =(lowerRightNormal - upperRightNormal) * invHeight;

    // Setup the framebuffer
    uint32_t* FB = (uint32_t*)malloc( sizeof(uint32_t) * width * height );

    
    // Test
/*    for( uint32_t x = 0; x < L; x++ )
        for( uint32_t y = 0; y < L; y++ )
            packRGBA32( &FB[y*L+x], *voxel3d( V, glm::ivec3( L, L, L ), glm::ivec3( x, y, 0 ) ) );
    */

    // Ray-cast main-loop
    glm::mat4 mat_model(1.f);
    glm::mat4 mat_modelview =mat_view * mat_model;
    glm::mat4 mat_inv_modelview =glm::inverse( mat_modelview );
    glm::vec3 leftNormal = upperLeftNormal;
    glm::vec3 rightNormal = upperRightNormal;
    ray r;
    r.origin = viewpoint;

    for( int y=0; y < height; y++ ) {
        
        r.direction = leftNormal;
        glm::vec3 normalXDelta = (rightNormal - leftNormal) * invWidth;

        for( int x=0; x < width; x++ ) {
            
            // Transform the ray from world-space to unit-cube-space
            ray r_cube;
            r_cube.direction =glm::normalize( glm::mat3( mat_inv_modelview ) * r.direction );
            r_cube.origin =glm::mat3( mat_inv_modelview ) * r.origin;

            // Cast the ray and set the framebuffer accordingly
            glm::vec4 color = raycast( V, size, r_cube );
            uint32_t rgba;
            packRGBA32( &rgba, color );

            FB[y*width + x] = rgba;

            // Shift within the current scanline
            r.direction +=normalXDelta;
        }
        // Shift to the next scanline
        leftNormal += leftNormalYDelta;
        rightNormal += rightNormalYDelta;
    }

    
    // Write the buffer as BMP
    std::vector<uint8_t> output;
    size_t output_size =bitmap_encode_rgba( FB, width, height, output );

    std::ofstream file_output;
    file_output.open("../buffer.bmp");
    file_output.write((const char*)&output[0], output_size);
    file_output.close();

    // Cleanup
    free( V );
    free( FB );
    return 0;
}

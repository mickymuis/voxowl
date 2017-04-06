#define GLM_SWIZZLE 
#include <stdio.h>
#include <stdint.h>
#include "platform.h"
#include "bmp.h"
#include <vector>
#include <fstream>
#include <glm/mat4x4.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtx/transform.hpp>
#include <glm/gtc/noise.hpp>

typedef struct {
    glm::vec3 origin;
    glm::vec3 direction;
} ray;

typedef struct {
    glm::vec3 min;
    glm::vec3 max;
} box;

typedef glm::vec4 voxel;

void
printVec( glm::vec3 v ) {
    printf( "vec3: (%f, %f, %f)\n", v.x, v.y, v.z );
}

void printVec( glm::ivec3 v ) {
    printf( "ivec3: (%d, %d, %d)\n", v.x, v.y, v.z );
}


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

glm::vec4
blendF2B( glm::vec4 src, glm::vec4 dst ) {
    glm::vec4 c;
    c.r = dst.a*(src.r * src.a) + dst.r;
    c.g = dst.a*(src.g * src.a) + dst.g;
    c.b = dst.a*(src.b * src.a) + dst.b;
    c.a = (1.f - src.a) * dst.a;
    return c;
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

    for( uint32_t x=0; x < size.x; x++ )
        for( uint32_t y=0; y < size.y; y++ )
            for( uint32_t z=0; z < size.z; z++ ) {
                (*voxel3d(V,size,glm::ivec3(x,y,z))).r = (float)x / (float)(size.x-1);
                (*voxel3d(V,size,glm::ivec3(x,y,z))).g = (float)y / (float)(size.y-1);
                (*voxel3d(V,size,glm::ivec3(x,y,z))).b = (float)z / (float)(size.z-1);
                (*voxel3d(V,size,glm::ivec3(x,y,z))).a = 1.0f;
            }
   
    menger( V, size, size, glm::ivec3(0) );
}

void
makeSphere( voxel* V, glm::ivec3 size ) {
    float maxDist =min(size.x, min( size.y, size.z ) ) / 2;
    float minDist =(float) maxDist * 0.8;
    glm::ivec3 center( size.x / 2, size.y / 2, size.z / 2 );

    for( uint32_t x=0; x < size.x; x++ )
        for( uint32_t y=0; y < size.y; y++ )
            for( uint32_t z=0; z < size.z; z++ ) {
                float dist =glm::distance( center, glm::ivec3( x, y, z ) );
                if( dist <= maxDist && dist >= minDist )
                    *voxel3d(V,size,glm::ivec3(x,y,z)) = glm::vec4( 1.f, 1.f, 1.f, .1f );
                else
                    *voxel3d(V,size,glm::ivec3(x,y,z)) = glm::vec4( 0.f );
            }
}

glm::vec3 
bezier3(
  float t,
  const glm::vec3 &c1,
  const glm::vec3 &c2,
  const glm::vec3 &c3,
  const glm::vec3 &c4)
{
    return   glm::pow( (1.f-t), 3 ) * c1 +
           3.f * glm::pow( 1.f-t, 2 )*t*c2 +
           3.f * (1.f-t) * glm::pow( t, 2) * c3 + 
           glm::pow( t,3 ) * c4 ;
}

glm::vec4
terrainLUT( float alt ) {
    return glm::vec4( bezier3( alt,
        glm::vec3( 0.4f, 0.4f, .1f ),
        glm::vec3( 0.7f, 0.5f, 0.2f ),
        glm::vec3( 0.4f, 1.0f, 0.2f ),
        glm::vec3( 0.8f, 1.0f, 0.6f ) ) , 1.0f );
}


void
makeTerrain( voxel *V, glm::ivec3 size ) {

    float sealevel =0.2f * size.y;
    glm::vec4 water( .4f, .8f, .9f, .2f );
    for( uint32_t x=0; x < size.x; x++ )
        for( uint32_t z=0; z < size.z; z++ ) {
            float noise_x = ((float)x / (float)size.x) * 2.f - 1.f;// * scale.x;
            float noise_z = ((float)z / (float)size.z) *2.f - 1.f;// * scale.z; 
            float density;
            
            *voxel3d(V,size,glm::ivec3(x,0,z)) =
                    terrainLUT( 0 );

            for( uint32_t y=1; y < size.y; y++ ) {
                    float noise_y = ((float)y / (float)size.y) * 2.f - 1.f;

                    glm::vec3 ws( noise_x, noise_y, noise_z );
//                    ws += glm::perlin( ws * 5.004f ) *0.5;
                    density =-noise_y;
                    density +=glm::perlin( ws * 17.7f ) * 0.05f;
                    density +=glm::perlin( ws * 15.1f ) * 0.03f;
                    density +=glm::perlin( ws * 9.3f ) * 0.10f;
                    density +=glm::perlin( ws * 4.03f ) * 0.25f;
                    density +=glm::perlin( ws * 2.4f ) * 0.50f;
                    density +=glm::perlin( ws * 1.75f ) * 0.90f;
                    density +=glm::perlin( ws * 1.07f ) * 1.30f;
                    
                    if( density > 0.f ) {
                        *voxel3d(V,size,glm::ivec3(x,y,z)) =
                            terrainLUT( (noise_y + 1.f) / 2.f);
                        continue;
                    }
                    else if( y < sealevel ) {
                        *voxel3d(V,size,glm::ivec3(x,y,z)) = water;
                        continue;
                    }
                    else       
                        *voxel3d(V,size,glm::ivec3(x,y,z)) = glm::vec4( 0.f );

            }   
        }
}


__device__ __host__
glm::vec3
voxelIndexToCoord( glm::ivec3 size, glm::ivec3 pos ) {
    int largest = max( size.x, max( size.y, size.z ) )-1;
    return glm::vec3( 
        (float)pos.x/(float)largest-.5f,
        (float)pos.y/(float)largest-.5f,
        (float)pos.z/(float)largest-.5f );
}

glm::ivec3
voxelCoordToIndex( glm::ivec3 size, glm::vec3 v ) {
    int largest = max( size.x, max( size.y, size.z ) )-1;
    return glm::ivec3(
        (int)round((v.x+0.5f)*(float)largest),
        (int)round((v.y+0.5f)*(float)largest),
        (int)round((v.z+0.5f)*(float)largest) );
}

glm::vec3
voxelCoordToIndexf( glm::ivec3 size, glm::vec3 v ) {
    int largest = max( size.x, max( size.y, size.z ) )-1;
    return glm::vec3(
        (v.x+0.5f)*(float)largest,
        (v.y+0.5f)*(float)largest,
        (v.z+0.5f)*(float)largest );
}

box
voxelSizeToAABB( glm::ivec3 size ) {
    int largest = max( size.x, max( size.y, size.z ) );
    largest *=2;
    box b;
    b.min = -glm::vec3 (
        (float)size.x / (float)largest,
        (float)size.y / (float)largest,
        (float)size.z / (float)largest );
    b.max =-b.min - glm::vec3(0.0) ;
    return b;
}

bool
rayAABBIntersect( const ray &r, const box& b, double& tmin, double& tmax ) {
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

glm::vec4
raycast( voxel* V, glm::ivec3 size, const ray& r ) {
    double tmin, tmax;
    box b = voxelSizeToAABB( size );
    if( !rayAABBIntersect( r, b, tmin, tmax ) )
        return glm::vec4( 0.0, 0.0, 0.0, 1.0 );

    glm::vec3 rayEntry = r.origin + r.direction * (float)max( 0.0, tmin );
    glm::vec3 rayExit = r.origin + r.direction * (float)tmax;

    // Determine the side of the unit cube the ray enters
    // In order to do this, we need the component with the largest absolute number
    // These lines are optimized to do so without branching
    const glm::ivec3 box_plane( 0, 1, 2 ); // X, Y and Z dividing planes
    glm::vec3 r0 = glm::abs( rayEntry / b.max );
    float largest =max( r0.x, max( r0.y, r0.z ) ); // Largest relative component
    glm::ivec3 r1 = glm::floor( r0 / largest ); // Vector with a '1' at the largest component
    int side = glm::clamp( glm::dot( box_plane, r1 ), 0, 2 );
   
    // Map the ray entry from unit-cube space to voxel space
    largest =max( size.x, max( size.y, size.z ) );
    rayEntry =(rayEntry + b.max) * largest;

    // Calculate the index in the volume by chopping off the decimal part
    glm::ivec3 index = glm::clamp( 
        glm::ivec3( glm::floor( rayEntry ) ) ,
        glm::ivec3( 0 ),
        glm::ivec3( size.x-1, size.y-1, size.z-1 ) );


    // Determine the sign of the stepping through the volume
    glm::ivec3 step = glm::sign( r.direction );

    // deltaDist gives the distance on the ray path for each following dividing plane
    glm::vec3 deltaDist =glm::abs( glm::vec3( glm::length( r.direction ) ) / r.direction );

    // Computes the distances to the next voxel for each component
    glm::vec3 boxDist = ( sign( r.direction ) * (glm::vec3(index) - rayEntry)
                        + (sign( r.direction ) * 0.5f ) + 0.5f ) * deltaDist;

    glm::vec4 color( 0.f, 0.f, 0.f, 1.f );

    while(1) {

        if( index[side] < 0 || index[side] >= size[side] )
            break;
        
        glm::vec4 vox =*voxel3d( V, size, index );

        vox.r *= (3-side)/3.f;
        vox.g *= (3-side)/3.f;
        vox.b *= (3-side)/3.f;

        color =blendF2B( vox, color );

        if( vox.a == 1.f )
            break;

/*
        for( int i =0; i < 3; i++ ) 
            if( boxDist[side] > boxDist[i] )
                side =i;*/

        /*float smallest =min( sideDist.x, min( sideDist.y, sideDist.z ) );
        glm::ivec3 mask =glm::floor( sideDist / smallest );
        side = glm::clamp( glm::dot( box_plane, mask ), 0, 2 );*/

        glm::bvec3 b0= glm::lessThan( boxDist.xyz(), boxDist.yzx() );
        glm::bvec3 b1= glm::lessThanEqual( boxDist.xyz(), boxDist.zxy() );
        glm::ivec3 mask =glm::ivec3( b0.x && b1.x, b0.y && b1.y, b0.z && b1.z );
        side = glm::dot( box_plane, mask );

        boxDist[side] += deltaDist[side];
        index[side] += step[side];


    }

    color.a = 1.f - color.a;
    return color;
}

int 
main( int argc, char **argv ) {

    // Make a test sponge V of size L
    const int width =1024;
    const int height =768;
    const glm::ivec3 size( 243, 243, 243 );
    voxel* V = (voxel*)malloc( sizeof( voxel ) * size.x * size.y * size.z );
    makeSponge( V, size );


    // Setup the camera
    float near =1.0f, far =100.f;
    glm::vec3 viewpoint( 1.f, .3f, -1.0f );
    glm::vec3 target( 0.0f, 0.0f, 0.0f );
    glm::vec3 viewdir =target - viewpoint;
    glm::vec3 up( 0, 1, 0 );
    glm::mat4 mat_view =glm::lookAt( viewpoint, target, up );

    // We assume a symmetric projection matrix
    glm::mat4 mat_proj =glm::perspective( glm::radians( 60.f ), (float)width/height, near, far );
    const float right =near / mat_proj[0][0];
    const float top =near / mat_proj[1][1];
    const float left =-right, bottom =-top;
    glm::vec3 upperLeftNormal =glm::normalize( glm::vec3( left, top, -near ) );
    glm::vec3 upperRightNormal =glm::normalize( glm::vec3( right, top, -near ) );
    glm::vec3 lowerLeftNormal =glm::normalize( glm::vec3( left, bottom, -near ) );
    glm::vec3 lowerRightNormal =glm::normalize( glm::vec3( right, bottom, -near ) );
    

    // Calculate the ray-normal interpolation constants
    const float invHeight = 1.f / (float)height;
    const float invWidth = 1.f / (float)width;

    glm::vec3 leftNormalYDelta = (lowerLeftNormal - upperLeftNormal) * invHeight;
    glm::vec3 rightNormalYDelta =(lowerRightNormal - upperRightNormal) * invHeight;

    // Setup the framebuffer
    uint32_t* FB = (uint32_t*)malloc( sizeof(uint32_t) * width * height );

    // Ray-cast main-loop
    glm::mat4 mat_model(1.f);
 //   glm::mat4 mat_model =glm::rotate( 45.0f, glm::vec3(0,1,0) );
//    glm::mat4 mat_model =glm::scale( glm::vec3(2.5f) );
    glm::mat4 mat_modelview =mat_view * mat_model;
    glm::mat4 mat_inv_modelview =glm::inverse( mat_modelview );
    glm::vec3 leftNormal = upperLeftNormal;
    glm::vec3 rightNormal = upperRightNormal;
    ray r;
    r.origin = glm::vec3( mat_inv_modelview * glm::vec4(0,0,0,1) );
//    r.origin = glm::vec3(0);
    printVec( r.origin);

    for( int y=0; y < height; y++ ) {
        
        r.direction = leftNormal;
        glm::vec3 normalXDelta = (rightNormal - leftNormal) * invWidth;

        for( int x=0; x < width; x++ ) {
            
            // Transform the ray from world-space to unit-cube-space
            ray r_cube;
            r_cube.direction =glm::normalize( glm::mat3( mat_inv_modelview ) * r.direction );
//            r_cube.direction =r.direction;
            r_cube.origin =r.origin;
            //r_cube.origin =viewpoint;

            ray r_cube2;
            r_cube2.direction =r_cube.direction + normalXDelta / 2.0f;
            r_cube2.origin =r_cube.origin;

            // Cast the ray and set the framebuffer accordingly
            glm::vec4 color1 = raycast( V, size, r_cube );
            glm::vec4 color2 = raycast( V, size, r_cube2 );

            uint32_t rgba;
            packRGBA32( &rgba, color1 * 0.5f + color2 * 0.5f );

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

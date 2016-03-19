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

    glm::vec3 scale( 5.f, 1.f, 5.f );
    float sealevel =0.2f * size.y;
    glm::vec4 water( .6f, .8f, .9f, .1f );
    for( uint32_t x=0; x < size.x; x++ )
        for( uint32_t z=0; z < size.z; z++ ) {
            float noise_x = ((float)x / (float)size.x) * scale.x;
            float noise_z = ((float)z / (float)size.z) * scale.z; 

            float height = 
                glm::perlin( glm::vec2(noise_x, noise_z ) );
            height += 1.0f;
            height *= 0.5f * size.y;
            for( uint32_t y=0; y < size.y; y++ ) {
                if( y < height ) {
                    float noise_y = ((float)y / (float)size.y);
                    float v =glm::perlin( glm::vec3 ( noise_x, noise_y * scale.y, noise_z ) );
                    if( v > 0.f ) {
                        *voxel3d(V,size,glm::ivec3(x,y,z)) =
                            terrainLUT( noise_y );
                        continue;
                    }
                    else if( y < sealevel ) {
                        *voxel3d(V,size,glm::ivec3(x,y,z)) = water;
                        continue;
                    }
                }            
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
    b.max =-b.min - glm::vec3(0.001) ;
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

glm::vec3
diffDistance( glm::vec3 s, glm::vec3 ds ) {
    glm::vec3 diff;
    for( int i =0; i < 3; i++ ) {
        float is;
        if( s[i] < 0.f )
            is = s[i] - floor( -1.f - s[i] );
        else
            is = s[i] - floor( s[i] );

        if ( ds[i] > 0.f )
            diff[i] = (1.f-is) / ds[i];
        else
            diff[i] = is / (-ds[i]);
    }
    return diff;
}

glm::vec4
raycast( voxel* V, glm::ivec3 size, const ray& r ) {
    double tmin, tmax;
    if( !rayAABBIntersect( r, voxelSizeToAABB( size ), tmin, tmax ) )
        return glm::vec4( 0.0, 0.0, 0.0, 1.0 );
//    else
//      return glm::vec4( 1.0, 0.0, 0.0, 1.0 );

    glm::vec3 rayEntry = r.origin + r.direction * (float)max( 0.0, tmin );
    glm::vec3 rayExit = r.origin + r.direction * (float)tmax;
    
    int side;
    if( fabs(rayEntry.x) > fabs(rayEntry.z) ) {
        if( fabs( rayEntry.x ) > fabs( rayEntry.y ) )
            side = 0;
        else
            side = 1;
    }
    else if( fabs( rayEntry.y ) > fabs( rayEntry.z ) )
        side = 1;
    else
        side = 2;

    glm::vec4 color( 0.f, 0.f, 0.f, 1.f );

//    ray r0;
//    r0.origin =(r.origin + glm::vec3(0.499)) * glm::vec3(size);
//    r0.direction =(r.direction + glm::vec3(0.499)) * glm::vec3(size);
    rayEntry =(rayEntry + glm::vec3(0.5)) * glm::vec3(size);

//    printVec( rayEntry );
    glm::ivec3 index = glm::clamp( 
        glm::ivec3( glm::floor( rayEntry ) ) ,
        glm::ivec3( 0 ),
        glm::ivec3( size.x, size.y, size.z ) );

//    color =*voxel3d( V, size, index );
//    return color;

    //glm::ivec3 index =voxelCoordToIndex( size, rayEntry );
//    printVec( index ); 
//    printf( "Voxel@ %d, %d, %d\n", entry.x, entry.y, entry.z );
//    glm::vec4 color =*voxel3d( V, size, entry );

    glm::ivec3 step = glm::sign( r.direction );

    glm::vec3 deltaDist =glm::abs( glm::vec3( glm::length( r.direction ) ) / r.direction );

    glm::vec3 sideDist = ( sign( r.direction ) * (glm::vec3(index) - rayEntry)
                        + (sign( r.direction ) * 0.5f ) + 0.5f ) * deltaDist;

//    glm::vec3 sideDist = diffDistance( rayEntry, r.direction );
//    glm::vec3 sideDist = (sign( r.direction ) * (voxelCoordToIndexf( size, rayEntry ) - glm::vec3(index))) / glm::vec3(size);
//    printVec( sideDist );
//    glm::vec3 sideDist = (sign(r.direction) * (voxelIndexToCoord( size,index) - r.origin) ) * deltaDist;

//    glm::vec3 sideDist = sign(r.direction) * deltaDist;

  //  glm::vec3 sideDist =( rayEntry - voxelIndexToCoord( size, index + step ) ) * glm::vec3(step) * deltaDist;

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

        side =0;
        for( int i =0; i < 3; i++ ) 
            if( sideDist[side] > sideDist[i] )
                side =i;

        sideDist[side] += deltaDist[side];
        index[side] += step[side];


    }

    color.a = 1.f - color.a;
//    color.a = 1.0f;
    return color;
}

int 
main( int argc, char **argv ) {

    // Make a test sponge V of size L
//    const uint32_t L =256;//243;
    const int width =1024;
    const int height =768;
    const glm::ivec3 size( 32, 32, 32 );
    voxel* V = (voxel*)malloc( sizeof( voxel ) * size.x * size.y * size.z );
    makeSphere( V, size );


    // Setup the camera
    float near =1.0f, far =100.f;
    glm::vec3 viewpoint( 0.f, .8f, 2.0f );
    glm::vec3 target(0);
    glm::vec3 viewdir =target - viewpoint;
    glm::vec3 up( 0, 1, 0 );
    glm::mat4 mat_view =glm::lookAt( viewpoint, target, up );

    // We assume a symmetric projection matrix
    glm::mat4 mat_proj =glm::perspective( glm::radians( 45.f ), (float)width/height, near, far );
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

    
    // Test
/*    for( uint32_t x = 0; x < L; x++ )
        for( uint32_t y = 0; y < L; y++ )
            packRGBA32( &FB[y*L+x], *voxel3d( V, glm::ivec3( L, L, L ), glm::ivec3( x, y, 0 ) ) );
    */

    // Ray-cast main-loop
 //   glm::mat4 mat_model(1.f);
    glm::mat4 mat_model =glm::rotate( 45.0f, glm::vec3(0,1,0) );
 //   glm::mat4 mat_model =glm::scale( glm::vec3(1.5f) );
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

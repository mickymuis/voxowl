#include "raycast_cuda.h"
#include "dda_cuda.h"
#include "svmm_cuda.h"
#include "platform.h"
#include "performance_counter.h"

#include "framebuffer.h"
#include "volume.h"
#include "volume_detail.h"
#include "camera.h"

#include "voxel.h"
#include <stdio.h>
#include <stdint.h>
#include <sstream>
#include <glm/mat4x4.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtx/transform.hpp>
#include <glm/gtc/random.hpp>
#include <vector_types.h>
#include <fstream>

#include <libdivide.h>

#define FAR -1000.f

// Define global volume and framebuffer handles, for now at least
texture<float4,1> ssao_kernel;
texture<float4,1> ssao_noise;
surface<void, 2> fb_color_surface;
surface<void, 2> fb_normal_depth_surface;

VOXOWL_HOST
float
random( float min, float max ) {
    return min + static_cast <float> (rand()) /( static_cast <float> (RAND_MAX/(max-min)));
}

// Simple return-on-error mechanism to improve readability
#define RETURN_IF_ERR(err) { if( setCudaErrorStr((err), __FILE__, __LINE__) ) return false; }
inline bool
RaycasterCUDA::setCudaErrorStr(cudaError_t code, const char *file, int line )
{
    if (code != cudaSuccess) 
    {
        std::stringstream s;
        s <<  "Error (cuda): '" 
          << cudaGetErrorString(code) 
          <<  "' in "
          << file 
          << " at line "
          << line;

        setError( true, s.str() );
        return true;
    }
    setError( false, std::string() );
    return false;
}

VOXOWL_DEVICE
static inline int fragIndex( const int &x, const int &y, const int &z, const int &dimX, const int &dimY ) {
    return z * dimX * dimY + y* dimX + x;
}


/* Parallel raycast kernel. Computes one fragment depending on position in the threadblock and writes in to the framebuffer */
VOXOWL_CUDA_KERNEL
void
computeFragment( raycastInfo_t raycast_info, volumeDevice_t volume, framebufferDevice_t framebuffer ) {
    extern __shared__ fragment_t frags[];

    // Calculate screen coordinates
    const unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
    const unsigned int z = blockIdx.z * blockDim.z + threadIdx.z;

    // We store the calculated fragment in shared memory
    fragment_t* f =&frags[fragIndex( threadIdx.x, threadIdx.y, z, blockDim.x, blockDim.y )];
    f->color =glm::vec4( 0,0,0,1 ); // Background color
    f->normal =glm::vec3(0);
    f->position =glm::vec3(0,0,FAR); // Non-intersecting rays have 'infinite depth'
    f->hit =false;

    // Next, we calculate the ray vector based on the screen coordinates
    glm::vec3 leftNormal = raycast_info.upperLeftNormal;
    glm::vec3 rightNormal = raycast_info.upperRightNormal;
       
    ray_t r;
    r.origin =raycast_info.origin;
    leftNormal += raycast_info.leftNormalYDelta * (float)y;
    rightNormal += raycast_info.rightNormalYDelta * (float)y;
    r.direction = leftNormal +  (rightNormal - leftNormal) * raycast_info.invWidth * (float)(x+1) + raycast_info.leftNormalYDelta;
    
/*    glm::vec3 normalXDelta = (rightNormal - leftNormal) * raycast_info.invWidth;
    r.direction +=normalXDelta * (float)x;*/

//    glm::vec3 raydir = r.direction + normalXDelta + raycast_info.leftNormalYDelta;

    // Transform the ray from world-space to unit-cube-space
  //  ray_t r_cube;
    r.direction =glm::normalize( glm::mat3( raycast_info.matInvModelView ) * r.direction );
//    r_cube.origin =r.origin;

    // Test for ray-volume intersection and prepare the ray segment
    
    float tmin, tmax;  

    if( rayAABBIntersect( r, volume.bounding_box, tmin, tmax ) ) {

        // Calculate ray entry and exit from ray vector and box coords
        glm::vec3 rayEntry = r.origin + r.direction * fmaxf( 0.f, tmin );
        //glm::vec3 rayExit = r.origin + r.direction * tmax;

        // Calculate the ray segment depending on thread's Z index
        //float segmentLen =glm::length( rayExit - rayEntry ) / ray_segments;
        //glm::vec3 rayBegin = rayEntry + r.direction * (segmentLen * z);
        //glm::vec3 rayEnd = rayBegin + r.direction * segmentLen;
            
        // Traverse the volume, pick the algorithm depending on the storage class
        switch( volume.mode ) {
            case VOXELMAP:
                voxelmapRaycast( *f, &volume.volume.voxelmap, volume.bounding_box, r, rayEntry );
                break;
            case SVMM: {
                float fragment_width =(raycast_info.fragmentWidthDelta * glm::length( r.direction * fmaxf( 0.f, tmin) ) ) / volume.voxel_width;
                float fragment_width_step = raycast_info.fragmentWidthDelta / volume.voxel_width;
                if( x == 500 && y == 400 )
                    printf( "Fragment width: before %f ", fragment_width );
                svmmRaycast( *f, &volume.volume.svmm, volume.bounding_box, r, rayEntry, fragment_width, fragment_width_step );
                if( x == 500 && y == 400 )
                    printf( "after %f ", fragment_width );
                break;
            }
            default:
                break;
        }

        // VSAO, only when using voxelmaps (for now )
/*        if( false && volume.mode == VOXELMAP ) {
            float4 noise =tex1D( ssao_noise, (float)x*y );
            glm::vec3 rvec( noise.x, noise.y, noise.z );

            // Setup the TBN matrix
            glm::vec3 tangent = glm::normalize(rvec - f->normal * glm::dot(rvec, f->normal));
            glm::vec3 bitangent = glm::cross(f->normal, tangent);
            glm::mat3 tbn(tangent, bitangent, f->normal);

            // Obtain the samples
            float occlusion =0.f;
            for (int i = 0; i < ssao_info.kernelSize; ++i) {
                // get sample position:

                float4 k =tex1D( ssao_kernel, i );
                glm::vec3 kernel_i( k.x, k.y, k.z );
                glm::vec3 sample = tbn * kernel_i;
                sample = sample * ssao_info.radius + f->position_vs + f->normal;
                
                glm::vec4 vox =voxelTex3D_clamp( 
                        volume.volume.voxelmap.texture, 
                        volume.volume.voxelmap.format, 
                        glm::floor( sample ), 
                        volume.volume.voxelmap.size );
                occlusion += vox.a;
            }

            
            f->color *= (1.f - occlusion / ssao_info.kernelSize );
        }*/
    }



    // Convert both the position and the normal to view-space
    // We export (some of) these values for use in later passes
    if( f->position.z != FAR ) {
        f->position = glm::vec3( glm::mat3(raycast_info.matModelView) * ( f->position - r.origin ) );
        f->normal = glm::normalize( glm::mat3( raycast_info.matModelView ) * f->normal );
    }
    float depth =f->position.z;

    __syncthreads();

   /* if( z ) {
        f->color.a = 1.f - f->color.a;
        
        return;
    }

    for( int i =1; i < ray_segments; i++ ) {
        fragment_t* g =&frags[fragIndex(threadIdx.x, threadIdx.y,i, blockDim.x, blockDim.y)];
        g->hit = g->hit && !frags[fragIndex(threadIdx.x, threadIdx.y,i-1, blockDim.x, blockDim.y)].hit;
        f->color =blendF2B( g->color * glm::vec4( g->hit ),
                f->color );
        if( g->hit ) {
            f->normal =g->normal;
            f->position =g->position;
        }
    }*/
   
    f->color =blendF2B( framebuffer.clear_color, f->color );
    f->color.a = 1.f - f->color.a;

    if( x % framebuffer.aaXSamples || y % framebuffer.aaYSamples )
        return;

    const float inv_samples =1.f / (float)(framebuffer.aaXSamples * framebuffer.aaYSamples);
    const int screen_x =x / framebuffer.aaXSamples;
    const int screen_y =y / framebuffer.aaYSamples;

    fragment_t frag;
    frag.position =f->position;
    frag.normal =f->normal;
    frag.color =glm::vec4(0);

    for( int i =0; i < framebuffer.aaXSamples; i++ )
        for( int j =0; j < framebuffer.aaYSamples; j++ ) {
            frag.color +=frags[fragIndex(threadIdx.x + i, threadIdx.y + j,0, blockDim.x, blockDim.y)].color * inv_samples;
        }
    
    // Write the color information to the framebuffer
    uint32_t rgba;
//    packRGBA_UINT32( &rgba, glm::vec4( frag.position.z, frag.position.z, frag.position.z, 1.f ) );
    packRGBA_UINT32( &rgba, frag.color  );

    // Workaround to be able to write to a 24bit buffer. Saves conversion later
    surf2Dwrite<uint8_t>( (uint8_t)( (rgba >> 24) & 0xFF), fb_color_surface, screen_x*3, screen_y, cudaBoundaryModeTrap );
    surf2Dwrite<uint8_t>( (uint8_t)( (rgba >> 16) & 0xFF), fb_color_surface, screen_x*3+1, screen_y, cudaBoundaryModeTrap );
    surf2Dwrite<uint8_t>( (uint8_t)( (rgba >> 8) & 0xFF), fb_color_surface, screen_x*3+2, screen_y, cudaBoundaryModeTrap );

    // Write the normal and depth values using a regular 32 float4 texture
    float4 normal_depth;
    normal_depth.w =depth;
    normal_depth.x =frag.normal.x;
    normal_depth.y =frag.normal.y;
    normal_depth.z =frag.normal.z;

    surf2Dwrite<float4>( normal_depth, fb_normal_depth_surface, screen_x * sizeof( float4 ), screen_y, cudaBoundaryModeTrap ); 
}

VOXOWL_DEVICE
inline glm::vec2
normalizedFragmentCoord( const framebufferDevice_t& framebuffer, int x, int y ) {
    return glm::vec2( (float)x / (float)framebuffer.width, (float)y / (float)framebuffer.height );
}

VOXOWL_DEVICE
inline glm::vec3
positionFromDepth( const raycastInfo_t& raycast_info, const unsigned int &x, const unsigned int &y, const float &z ) {
    // Calculate the origin (position) of the fragment in view space using the depth
    glm::vec3 origin;
    glm::vec3 leftNormal = raycast_info.upperLeftNormal;
    glm::vec3 rightNormal = raycast_info.upperRightNormal;

    leftNormal += raycast_info.leftNormalYDelta * (float)y;
    rightNormal += raycast_info.rightNormalYDelta * (float)y;
    origin = leftNormal;
    
    glm::vec3 normalXDelta = (rightNormal - leftNormal) * raycast_info.invWidth;
    origin +=normalXDelta * (float)x;

    // We extend the calculated ray vector along the z axis by the fragment's depth
    return origin / origin.z * (float)z;
}

VOXOWL_CUDA_KERNEL
void
computeFragmentNormal( raycastInfo_t raycast_info, volumeDevice_t volume, framebufferDevice_t framebuffer ) {
    const unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;

    // Obtain the normal and depth values for this fragment
    //float4 normal_depth =surf2Dread<float4>( fb_normal_depth_surface, x * sizeof( float4 ), y, cudaBoundaryModeTrap );
    float4 normal_depth =tex2D<float4>( framebuffer.normal_depth_sampler, 
            (float)x / (float)framebuffer.width, 
            (float)y / (float)framebuffer.height );
    float depth =normal_depth.w;
    int pixels_per_voxel =1;//(int)ceilf( volume.voxel_width / (raycast_info.fragmentWidthWorldDelta * -depth) );
    glm::vec3 p = positionFromDepth( raycast_info, x, y, depth );

    glm::vec3 normal(0);// = glm::vec3( normal_depth.x, normal_depth.y, normal_depth.z );
    const int max_radius =(int)ceilf( volume.voxel_width / (raycast_info.fragmentWidthWorldDelta * -depth) );
    glm::vec3 incident =-glm::normalize( raycast_info.origin );
    
    // Obtain a random number
    float4 noise =tex1D( ssao_noise, (float)x*y );
    glm::vec3 rvec( noise.x, noise.y, noise.z );
//    p+=rvec*volume.voxel_width;

    const glm::vec2 offsets[8] = {
        glm::vec2( 1, 0 ),
        glm::vec2( 1, 1 ),
        glm::vec2( 0, 1 ),
        glm::vec2( -1, 1 ),
        glm::vec2( -1, 0 ),
        glm::vec2( -1, -1 ),
        glm::vec2( 0, -1 ),
        glm::vec2( 1, -1 ) };
    const int n_offsets =8;

    for( int r =1; r <= max_radius; r++ ) {

        glm::vec3 p1(0), p2(0);

        for( int i =0; i < n_offsets; i++ ) {
            glm::vec2 pixel =glm::vec2(x,y) + offsets[i]*(float)r;
            float4 sample_depth =tex2D<float4>( framebuffer.normal_depth_sampler, 
                    pixel.x / (float)framebuffer.width, pixel.y / (float)framebuffer.height );
            if( fabs( sample_depth.w - normal_depth.w ) > volume.voxel_width )
                continue;
            p1 =p2;
            p2 = positionFromDepth( raycast_info, pixel.x, pixel.y, sample_depth.w );
            
            if( p1 != glm::vec3(0) && p2 != glm::vec3(0) ) {
                glm::vec3 n = glm::normalize( glm::cross( p1 - p, p2 - p ) );
                if( !glm::any( glm::isnan( n ) ) )
                    normal +=glm::faceforward( n, n, -glm::vec3( normal_depth.x, normal_depth.y, normal_depth.z ) );// * (float)r;
                    //normal +=glm::faceforward( n, n, incident ) * (float)r;
                    //normal -=n;
            }
        }
    }

/*    const int max_radius =(int)ceilf( volume.voxel_width / (raycast_info.fragmentWidthWorldDelta * -depth) );
    const int segments =8;
    
    // Obtain a random number
    float4 noise =tex1D( ssao_noise, (float)x*y );
    glm::vec3 rvec( noise.x, noise.y, noise.z );

    for( int r =2; r <= max_radius; r++ ) {
        float theta = 2.f * 3.1415926f / float(segments); 
        float c = cosf(theta); //precalculate the sine and cosine
        float s = sinf(theta);
        float t;

        float rx = r;//we start at angle = 0 
        float ry = 0;

        glm::vec3 p1(0), p2(0);

        for(int i = 0; i <= segments; i++) 
        { 
            
            glm::ivec2 pixel =glm::ivec2(x+rx,y+ry);
            float4 sample_depth =surf2Dread<float4>( fb_normal_depth_surface, 
                pixel.x * sizeof( float4 ), pixel.y, cudaBoundaryModeClamp );
            if( fabs( sample_depth.w - normal_depth.w ) > 2.f * volume.voxel_width )
                continue;
            p1 =p2;
            p2 = positionFromDepth( raycast_info, x + (int)(rx+rvec.x), y + (int)(ry+rvec.y), sample_depth.w );
            
            if( p1 != glm::vec3(0) && p2 != glm::vec3(0) ) {
                glm::vec3 n = glm::normalize( glm::cross( p1 - p, p2 - p ) );
                if( !glm::any( glm::isnan( n ) ) )
                    //normal +=glm::faceforward( n, n, -glm::vec3( normal_depth.x, normal_depth.y, normal_depth.z ) ) * (float)r;
                    normal +=n;
            }

            //n+= glm::vec3(sample_depth.x, sample_depth.y, sample_depth.z);
            //n.x += (p1.y-p2.y) * (p1.z+p2.z);
            //n.y += (p1.z-p2.z) * (p1.x+p2.x);
            //n.z += (p1.x-p2.x) * (p1.y+p2.y);

            // next cycle apply the rotation matrix
            t = rx;
            rx = c * rx - s * ry;
            ry = s * t + c * ry;
        }
        //normal +=n*(float)r;
    }*/

/*    const int SAMPLES =(int)ceilf( volume.voxel_width / (raycast_info.fragmentWidthWorldDelta * -depth) );
    const int SAMPLES2 =64;

    glm::vec3 prev_sample =p;

    for( int i =-SAMPLES; i <= SAMPLES; i++ ) {
        //if( i == 0 )
          //  continue;
        for( int j =-SAMPLES; j <= SAMPLES; j++ ) {
            //if( j==0 )
              //  continue;

            float4 normal_depth1 =surf2Dread<float4>( fb_normal_depth_surface, (x + pixels_per_voxel*i) * sizeof( float4 ), y + pixels_per_voxel*j, cudaBoundaryModeClamp );
            if( fabs( normal_depth1.w - normal_depth.w ) > 2.f * volume.voxel_width )
                continue;
            glm::vec3 p1 = positionFromDepth( raycast_info, x + pixels_per_voxel*i, y + pixels_per_voxel*j, normal_depth1.w );
            
            j++;

//            float4 normal_depth2 =surf2Dread<float4>( fb_normal_depth_surface, (x + pixels_per_voxel*i) * sizeof( float4 ), y + pixels_per_voxel*j, cudaBoundaryModeClamp );
  //          if( fabs( normal_depth2.w - normal_depth.w ) > 2.f * volume.voxel_width )
  //              continue;
  //          glm::vec3 p2 = positionFromDepth( raycast_info, x + pixels_per_voxel*i, y + pixels_per_voxel*j, normal_depth2.w );

           // glm::vec3 n = glm::normalize( glm::cross( p1 - p, p2 - p ) );

//            float weight =1.f / (SAMPLES2);
         //   normal +=glm::faceforward( n, n, -glm::vec3( normal_depth.x, normal_depth.y, normal_depth.z ) );
           // normal +=n;
            normal+= glm::vec3(normal_depth1.x, normal_depth1.y, normal_depth1.z) * sqrtf(i*i+j*j);

//            normal.x += (prev_sample.y-sample.y) * (prev_sample.z+sample.z);
//            normal.y += (prev_sample.z-sample.z) * (prev_sample.x+sample.x);
//            normal.z += (prev_sample.x-sample.x) * (prev_sample.y+sample.y);
           // glm::vec3 n =glm::cross( prev_sample, sample );
           // normal +=glm::faceforward( n, n, -glm::vec3( normal_depth.x, normal_depth.y, normal_depth.z ) );
            //normal +=glm::faceforward( n, n, glm::vec3( 0, 0, -1 ) );

           // prev_sample =sample;
        }
    }*/
    normal =glm::normalize( normal );
//    normal =glm::faceforward( normal, normal, glm::vec3( 0, 0, -1 ) );
  //  normal =glm::faceforward( normal, normal, -glm::vec3( normal_depth.x, normal_depth.y, normal_depth.z ) );

    if( x==600 && y==400 ) {
        printf( "old normal (%f,%f,%f) new normal (%f,%f,%f) pixels per voxel %f\n", 
                normal_depth.x, normal_depth.y, normal_depth.z, normal.x, normal.y, normal.z,
                volume.voxel_width / (raycast_info.fragmentWidthWorldDelta * -depth) );
    }

    normal_depth.x = normal.x;
    normal_depth.y = normal.y;
    normal_depth.z = normal.z;
    
    surf2Dwrite<float4>( normal_depth, fb_normal_depth_surface, x * sizeof( float4 ), y, cudaBoundaryModeTrap ); 
}

VOXOWL_CUDA_KERNEL
void
computeFragmentLighting( raycastInfo_t raycast_info, framebufferDevice_t framebuffer ) {
    const unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;

    // Read back the values from the previous pass
    uint8_t r =surf2Dread<uint8_t>( fb_color_surface, x*3, y, cudaBoundaryModeTrap );
    uint8_t g =surf2Dread<uint8_t>( fb_color_surface, x*3+1, y, cudaBoundaryModeTrap );
    uint8_t b =surf2Dread<uint8_t>( fb_color_surface, x*3+2, y, cudaBoundaryModeTrap );
    
    // Obtain the normal and depth values for this fragment
    //float4 normal_depth =surf2Dread<float4>( fb_normal_depth_surface, x * sizeof( float4 ), y, cudaBoundaryModeTrap );
    float4 normal_depth =tex2D<float4>( framebuffer.normal_depth_sampler, 
            (float)x / (float)framebuffer.width, 
            (float)y / (float)framebuffer.height );

    if( normal_depth.w == FAR )
        return;

    glm::vec3 normal( normal_depth.x, normal_depth.y, normal_depth.z );
    
    // Calculate the origin (position) of the fragment in view space using the depth
    glm::vec3 origin =positionFromDepth( raycast_info, x, y, normal_depth.w );
    
    const glm::vec3 lightPos(1.f,2.f,1.f);
    const glm::vec3 specColor = glm::vec3(1.f, .96f, .87f) * .6f;
    const glm::vec3 ambient = glm::vec3( .09f, .22f, .31f  ) * .5f;

    glm::vec3 lightDir = glm::normalize(lightPos - origin);

    float lambertian = glm::max(glm::dot(lightDir,normal), 0.f);
    float specular = 0.0;

    if(lambertian > 0.0) {

        glm::vec3 viewDir = glm::normalize(-origin);

        glm::vec3 halfDir = glm::normalize(lightDir + viewDir);
        float specAngle = glm::max(glm::dot(halfDir, normal), 0.f);
        specular = glm::pow(specAngle, 8.0);
    
   /*     glm::vec3 reflectDir = glm::reflect(-lightDir, normal);
        float specAngle = glm::max(glm::dot(reflectDir, viewDir), 0.f);
        specular = glm::pow(specAngle, 4.0);*/
    }

    // Apply the result
    glm::vec3 color( r, g, b );
    color /= glm::vec3( 255.f );
    color = glm::clamp( ambient + lambertian*color + specular*specColor, glm::vec3(0), glm::vec3(1) );

    // Gamma step
    
    //color =glm::pow( color, glm::vec3( 1.f / 2.2f ) );
    
    surf2Dwrite<uint8_t>( color.r * 255.f, fb_color_surface, x*3, y, cudaBoundaryModeTrap );
    surf2Dwrite<uint8_t>( color.g * 255.f, fb_color_surface, x*3+1, y, cudaBoundaryModeTrap );
    surf2Dwrite<uint8_t>( color.b * 255.f, fb_color_surface, x*3+2, y, cudaBoundaryModeTrap );
}
VOXOWL_CUDA_KERNEL
void
computeFragmentSSAO( raycastInfo_t raycast_info, ssaoInfo_t ssao_info, framebufferDevice_t framebuffer, glm::mat4 mat_projection ) {
    const unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
    float occlusion =0.f;

    // Read back the values from the previous pass
    uint8_t r =surf2Dread<uint8_t>( fb_color_surface, x*3, y, cudaBoundaryModeTrap );
    uint8_t g =surf2Dread<uint8_t>( fb_color_surface, x*3+1, y, cudaBoundaryModeTrap );
    uint8_t b =surf2Dread<uint8_t>( fb_color_surface, x*3+2, y, cudaBoundaryModeTrap );
    
    // Obtain the normal and depth values for this fragment
    //float4 normal_depth =surf2Dread<float4>( fb_normal_depth_surface, x * sizeof( float4 ), y, cudaBoundaryModeTrap );
    float4 normal_depth =tex2D<float4>( framebuffer.normal_depth_sampler, 
            (float)x / (float)framebuffer.width, 
            (float)y / (float)framebuffer.height );
    glm::vec3 normal( normal_depth.x, normal_depth.y, normal_depth.z );
    
    // Calculate the origin (position) of the fragment in view space using the depth
    glm::vec3 origin =positionFromDepth( raycast_info, x, y, normal_depth.w );
/*    glm::vec3 leftNormal = raycast_info.upperLeftNormal;
    glm::vec3 rightNormal = raycast_info.upperRightNormal;

    leftNormal += raycast_info.leftNormalYDelta * (float)y;
    rightNormal += raycast_info.rightNormalYDelta * (float)y;
    origin = leftNormal;
    
    glm::vec3 normalXDelta = (rightNormal - leftNormal) * raycast_info.invWidth;
    origin +=normalXDelta * (float)x;

    // We extend the calculated ray vector along the z axis by the fragment's depth
    origin =origin / origin.z * normal_depth.w;*/


    // Obtain a random number to rotate the sampple matrix
    float4 noise =tex1D( ssao_noise, (float)x*y );
    glm::vec3 rvec( noise.x, noise.y, noise.z );

    // Setup the TBN matrix
    glm::vec3 tangent = glm::normalize(rvec - normal * glm::dot(rvec, normal));
    glm::vec3 bitangent = glm::cross(normal, tangent);
    glm::mat3 tbn(tangent, bitangent, normal);

    /*    if( r > 1 ) {
            printf( "origin %f %f %f screenpos %d %d normal %f %f %f\n",  origin.x, origin.y, origin.z, x, y, normal.x, normal.y, normal.z );
        }*/
    
        // Obtain the samples
    for (int i = 0; i < ssao_info.kernelSize; ++i) {
        // get sample position:

        float4 k =tex1D( ssao_kernel, i );
        
        glm::vec3 kernel_i( k.x, k.y, k.z );
        glm::vec3 sample_i = tbn * kernel_i;
        glm::vec3 sample = sample_i * ssao_info.radius + origin;

        // project sample position:
        glm::vec4 offset = mat_projection * glm::vec4( sample, 1.f );
        glm::vec2 screenpos = glm::vec2( offset ) / offset.w ;
        screenpos.y =-screenpos.y;
        screenpos = screenpos * 0.5f + glm::vec2(0.5f);
       // screenpos *= glm::vec2( framebuffer.width, framebuffer.height );

        // get sample depth:
/*        float4 normal_depth =surf2Dread<float4>( fb_normal_depth_surface, 
            (int)screenpos.x * sizeof( float4 ), 
            (int)screenpos.y, 
            cudaBoundaryModeClamp );*/
        float4 normal_depth =tex2D<float4>( framebuffer.normal_depth_sampler, 
            screenpos.x, 
            screenpos.y );
        float sampleDepth = normal_depth.w;

        // range check & accumulate:
        float rangeCheck= (glm::abs(origin.z - sampleDepth) < ssao_info.radius) ? 1.0 : 0.0;
        occlusion += (sampleDepth >= sample.z ? 1.0 : 0.0) * rangeCheck;

/*       if( x == 767 && y == 753 ) {
       printf( "frag: %d %d sample depth %f sample %f %f %f origin %f %f %f screenpos %f %f kernel %f %f %f\n", x,y, sampleDepth, sample.x, sample.y, sample.z, origin.x, origin.y, origin.z, screenpos.x, screenpos.y, sample_i.x, sample_i.y, sample_i.z );
       } */
    }

    // Apply the result
    occlusion = 1.f - occlusion / ssao_info.kernelSize;


    glm::vec3 color( r, g, b );
    color /= glm::vec3( 255.f );
    color *= occlusion;

    // Gamma step
    
    color =glm::pow( color, glm::vec3( 1.f / 2.2f ) );
    
    surf2Dwrite<uint8_t>( color.r * 255.f, fb_color_surface, x*3, y, cudaBoundaryModeTrap );
    surf2Dwrite<uint8_t>( color.g * 255.f, fb_color_surface, x*3+1, y, cudaBoundaryModeTrap );
    surf2Dwrite<uint8_t>( color.b * 255.f, fb_color_surface, x*3+2, y, cudaBoundaryModeTrap );
}



RaycasterCUDA::RaycasterCUDA( const char* name, Object* parent ) 
    : Renderer( name, parent ) {
    bzero( &d_volume, sizeof( volumeDevice_t ) );
    bzero( &framebuffer, sizeof( framebuffer_t ) );
    if (!initSSAO() ) {
        fprintf( stderr, "initSSAO(): %s\n", errorString().c_str() ); 
    }
    cudaEventCreate( &render_begin );
    cudaEventCreate( &render_finish );
    cudaEventCreate( &ssao_step );
    cudaEventCreate( &ssna_step );
    cudaEventCreate( &aa_step );
    cudaEventCreate( &lighting_step );

    PerformanceCounter::create( "frame", "Total render time", "fps" );
    PerformanceCounter::create( "raycast", "Raycast kernel", "ms" );
    PerformanceCounter::create( "ssao", "SSAO kernel", "ms" );
    PerformanceCounter::create( "ssna", "SSNA kernel", "ms" );
    PerformanceCounter::create( "aa", "AA kernel", "ms" );
    PerformanceCounter::create( "lighting", "Lighting kernel", "ms" );
}

RaycasterCUDA::~RaycasterCUDA() {

    if( framebuffer.color_data )
        cudaFreeArray( framebuffer.color_data );
    if( framebuffer.normal_depth_data )
        cudaFreeArray( framebuffer.normal_depth_data );
    freeVolumeMem();
    cudaEventDestroy( render_begin );
    cudaEventDestroy( render_finish );
    cudaEventDestroy( ssao_step );
    cudaEventDestroy( ssna_step );
    cudaEventDestroy( aa_step );
    cudaEventDestroy( lighting_step );
}

bool
RaycasterCUDA::initVoxelmap( voxelmap_t *voxelmap ) {
    
    if( !freeVolumeMem() )
        return false;

    if( !voxelmap || !voxelmap->data )
        return setError( true, "No data in voxelmap" );

    // Initialize the storage mode
    d_volume.mode =VOXELMAP;
    bzero( &d_volume.volume.voxelmap, sizeof( voxelmapDevice_t ) );
    voxelmapDevice_t *v = &d_volume.volume.voxelmap;

    // Copy the properties from the input voxelmap
    d_volume.bounding_box =volumeSizeToAABB( glm_ivec3_32( voxelmap->size ) );
    d_volume.voxel_width =2.f * d_volume.bounding_box.max.x / voxelmap->size.x;
    v->size =glm_ivec3_32( voxelmap->size );
    v->blocks =glm_ivec3_32( voxelmap->blocks );
    v->format =voxelmap->format;

    // Allocate an 3D array on the device
    //
    cudaExtent v_extent;
    cudaChannelFormatDesc v_channelDesc;

    printf( "Allocating texture for voxelmap, blocks=(%d,%d,%d), bytes per block=%d\n", 
            voxelmap->blocks.x, voxelmap->blocks.y, voxelmap->blocks.z, bytesPerBlock( voxelmap->format ) );

    int bits_per_block =bytesPerBlock( voxelmap->format ) * 8;

    v_extent = make_cudaExtent( v->blocks.x, v->blocks.y, v->blocks.z );
    v_channelDesc = cudaCreateChannelDesc( bits_per_block,0,0,0,cudaChannelFormatKindUnsigned);

    RETURN_IF_ERR( cudaMalloc3DArray( &v->data, &v_channelDesc, v_extent ) );

    // Prepare the parameters to the texture binding
    //
    cudaResourceDesc resDesc;
    memset(&resDesc, 0, sizeof(resDesc));
    resDesc.resType = cudaResourceTypeArray;
    resDesc.res.array.array = (cudaArray_t)v->data;

    cudaTextureDesc texDesc;
    memset(&texDesc, 0, sizeof(texDesc));
    texDesc.readMode = cudaReadModeElementType;
    texDesc.filterMode = cudaFilterModePoint;

    // Bind the texture object
    
    v->texture =0;
    RETURN_IF_ERR( cudaCreateTextureObject( &v->texture, &resDesc, &texDesc, NULL ) );

    
    // Copy the volume to the device
    // TODO: make separate step
    cudaMemcpy3DParms copyParams = {0};
    copyParams.srcPtr   = make_cudaPitchedPtr( voxelmap->data, v_extent.width * bytesPerBlock(voxelmap->format), v_extent.width, v_extent.height);
    copyParams.dstArray = v->data;
    copyParams.extent   = v_extent;
    copyParams.kind     = cudaMemcpyHostToDevice;
    RETURN_IF_ERR( cudaMemcpy3D(&copyParams) );
    return true;
}

bool
RaycasterCUDA::initSVMM( svmipmap_t *svmm ) {
    if( !freeVolumeMem() )
        return false;

    if( !svmm || !svmm->data_ptr )
        return setError( true, "No data in voxelmap" );
    
    d_volume.mode =SVMM;
    d_volume.bounding_box =volumeSizeToAABB( glm_ivec3_32( svmm->header.volume_size ) );
    d_volume.voxel_width =2.f * d_volume.bounding_box.max.x / svmm->header.volume_size.x;
    svmipmapDevice_t *d_svmm =&d_volume.volume.svmm;
    bzero( d_svmm, sizeof( svmipmapDevice_t ) );

    RETURN_IF_ERR( svmmCopyToDevice( d_svmm, svmm ) );
    return true;
}

bool
RaycasterCUDA::initFramebuffer() {
    return true;
}

bool 
RaycasterCUDA::beginRender() {

    
    if( !getFramebuffer() )
        return setError( true, "No framebuffer set" );
    if( !getVolume() || !getVolume()->storageDetail() )
        return setError( true, "No input volume set" );
    if( !getCamera() )
        return setError( true, "No camera set" );

    const int width =getFramebuffer()->getWidth();
    const int height =getFramebuffer()->getHeight();
    const int ray_segments =1;
    const dim3 blocksize(16, 16, ray_segments);

    // Reallocate and upload the volume
    if( last_config_volume != getVolume()->getConfiguration() ) {
        last_config_volume = getVolume()->getConfiguration();

        // First, determine the storage type
        VolumeVoxelmapStorageDetail* vol;
        VolumeSVMipmapStorageDetail* vol_svmm;
        if( (vol =dynamic_cast<VolumeVoxelmapStorageDetail*>(getVolume()->storageDetail())) ) {
            if( !initVoxelmap( vol->getVoxelmap() ) )
                return setError( true, "Could not initilize voxelmap" );
        } else if( vol_svmm =dynamic_cast<VolumeSVMipmapStorageDetail*>(getVolume()->storageDetail()) ) {
            if( !initSVMM( vol_svmm->getSVMipmap() ) )
                return setError( true, "Could not initilize SVMM" );
        } else {
            return setError( true, "Volume has unsupported storage class" );
        }
    }
    
    
    // Allocate the framebuffer on the device, if neccesary
    bool realloc_framebuffer = !framebuffer.color_data 
        || ( framebuffer.fb_d.width != width ) 
        || ( framebuffer.fb_d.height != height )
        || ( (int)framebuffer.fb_d.format != getFramebuffer()->getPixelFormat() );
    
    framebuffer.fb_d.aaXSamples =getFramebuffer()->getAAXSamples();
    framebuffer.fb_d.aaYSamples =getFramebuffer()->getAAYSamples();

    if( realloc_framebuffer ) { // Reallocate the framebuffer on the device end
        if( framebuffer.color_data )
            RETURN_IF_ERR( cudaFreeArray( framebuffer.color_data ) );
        if( framebuffer.normal_depth_data )
            RETURN_IF_ERR( cudaFreeArray( framebuffer.normal_depth_data ) );

        framebuffer.fb_d.width = width;
        framebuffer.fb_d.height = height;
        framebuffer.fb_d.format =(voxowl_pixel_format_t)getFramebuffer()->getPixelFormat();
        framebuffer.fb_d.clear_color =getFramebuffer()->getClearColor();

        int bytes_per_pixel;
        // TODO: this could use a nice function
        if( framebuffer.fb_d.format == VOXOWL_PF_RGB888 )
            bytes_per_pixel =3;
        else
            return false;

        // We allocate one buffer for the color data and optionally another for depth/normal data
        cudaChannelFormatDesc fb_channelDesc = cudaCreateChannelDesc(8,0,0,0,cudaChannelFormatKindUnsigned);
        RETURN_IF_ERR( cudaMallocArray( &(framebuffer.color_data), &fb_channelDesc, width * bytes_per_pixel, height, cudaArraySurfaceLoadStore ) );
        RETURN_IF_ERR( cudaBindSurfaceToArray( fb_color_surface, framebuffer.color_data ) );
        
        // We use a 32bit 4float type for the normal+depth buffer
        cudaChannelFormatDesc fb_channelDesc2 = cudaCreateChannelDesc(32,32,32,32,cudaChannelFormatKindFloat);
        RETURN_IF_ERR( cudaMallocArray( &(framebuffer.normal_depth_data), &fb_channelDesc2, width, height, cudaArraySurfaceLoadStore ) );
        RETURN_IF_ERR( cudaBindSurfaceToArray( fb_normal_depth_surface, framebuffer.normal_depth_data ) );
    
        // Beside the surface, we also use a texture object to sample the normal-depth texture
        cudaResourceDesc resDesc;
        memset(&resDesc, 0, sizeof(resDesc));
        resDesc.resType = cudaResourceTypeArray;
        resDesc.res.array.array = (cudaArray_t)framebuffer.normal_depth_data;

        cudaTextureDesc texDesc;
        memset(&texDesc, 0, sizeof(texDesc));
        texDesc.readMode = cudaReadModeElementType;
        texDesc.filterMode = cudaFilterModeLinear; // enable linear filtering of the normal-depth
        texDesc.normalizedCoords =true;

        // Bind the texture object
        
        framebuffer.fb_d.normal_depth_sampler =0;
        RETURN_IF_ERR( cudaCreateTextureObject( &(framebuffer.fb_d.normal_depth_sampler), &resDesc, &texDesc, NULL ) );
    }

    
    // Setup the raycast parameters based on the matrices from the camera and the 'model'
    // TODO: some kind of caching?
    raycastInfo_t raycast_info;
    getCamera()->setAspect( (float)width/(float)height );
    raycastSetMatrices( &raycast_info, 
                        getVolume()->modelMatrix(), 
                        getCamera()->getViewMatrix(), 
                        getCamera()->getProjMatrix(), 
                        width * framebuffer.fb_d.aaXSamples, 
                        height* framebuffer.fb_d.aaYSamples );

    //float sample_width_delta = (d_volume.bounding_box.max.y * 2) / getVolume()->size().y;
    printf( "Fragment width delta: %f (modelspace) %f (worldspace)\n", raycast_info.fragmentWidthDelta, raycast_info.fragmentWidthWorldDelta );

    // Divide the invidual fragments over N / blocksize blocks
    // Run the raycast kernel on the device
    const dim3 numblocks( (width * framebuffer.fb_d.aaXSamples) / blocksize.x, 
                          (height * framebuffer.fb_d.aaYSamples) / blocksize.y, 
                          1 );
    size_t sm_bytes =blocksize.x * blocksize.y * blocksize.z * sizeof( fragment_t );
    RETURN_IF_ERR( cudaBindSurfaceToArray( fb_color_surface, framebuffer.color_data ) );
    RETURN_IF_ERR( cudaBindSurfaceToArray( fb_normal_depth_surface, framebuffer.normal_depth_data ) );
    cudaFuncSetCacheConfig( computeFragment, cudaFuncCachePreferL1 );
    cudaEventRecord( render_begin );
    computeFragment<<<numblocks, blocksize, sm_bytes>>>( raycast_info, d_volume, framebuffer.fb_d );
    RETURN_IF_ERR( cudaGetLastError() );

    // Run screen space deferred render passes
    dim3 blocksize_filter( 16, 16 );
    dim3 numblocks_filter( width / blocksize_filter.x, height / blocksize_filter.y );
    raycastSetMatrices( &raycast_info, 
                        getVolume()->modelMatrix(), 
                        getCamera()->getViewMatrix(), 
                        getCamera()->getProjMatrix(), 
                        width, 
                        height );
    cudaEventRecord( ssna_step );

    if( isEnabled( FEATURE_SSNA ) ) {
        RETURN_IF_ERR( cudaDeviceSynchronize() );
        computeFragmentNormal<<<numblocks_filter, blocksize_filter>>>( raycast_info, d_volume, framebuffer.fb_d );
        RETURN_IF_ERR( cudaGetLastError() );
    }
    
    cudaEventRecord( aa_step );
   
    if( isEnabled( FEATURE_AA ) ) {
  /*  RETURN_IF_ERR( cudaDeviceSynchronize() );
    computeFragmentAA<<<numblocks_filter, blocksize_filter>>>( raycast_info, d_volume, framebuffer.fb_d );
    RETURN_IF_ERR( cudaGetLastError() );*/
    }
    
    cudaEventRecord( ssao_step );
    
    if( isEnabled( FEATURE_SSAO ) ) {
        RETURN_IF_ERR( cudaDeviceSynchronize() );
        computeFragmentSSAO<<<numblocks_filter, blocksize_filter>>>( raycast_info, ssao_info, framebuffer.fb_d, getCamera()->getProjMatrix() );
        RETURN_IF_ERR( cudaGetLastError() );
    }
    
    cudaEventRecord( lighting_step );

    if( isEnabled( FEATURE_LIGHTING ) ) {
        RETURN_IF_ERR( cudaDeviceSynchronize() );
        computeFragmentLighting<<<numblocks_filter, blocksize_filter>>>( raycast_info, framebuffer.fb_d );
        RETURN_IF_ERR( cudaGetLastError() );
    }
    
    cudaEventRecord( render_finish );

    return true;
}

bool 
RaycasterCUDA::synchronize() {
    // Wait for the running kernel to finish
    //RETURN_IF_ERR( cudaDeviceSynchronize() );

    // Copy the framebuffer to the host

    // TODO: use framebuffer format
    int bpp =3;
    void* data_ptr =getFramebuffer()->data();
    int width =getFramebuffer()->getWidth();
    int height =getFramebuffer()->getHeight();


    RETURN_IF_ERR ( cudaMemcpyFromArray( data_ptr, framebuffer.color_data, 0, 0, width*height*bpp, cudaMemcpyDeviceToHost ) );
    
    cudaEventSynchronize( render_finish );
    float ms_raycast, ms_ssao, ms_ssna, ms_aa, ms_lighting;

    cudaEventElapsedTime( &ms_raycast, render_begin, ssna_step );
    cudaEventElapsedTime( &ms_ssna, ssna_step, aa_step );
    cudaEventElapsedTime( &ms_aa, aa_step, ssao_step );
    cudaEventElapsedTime( &ms_ssao, ssao_step, lighting_step );
    cudaEventElapsedTime( &ms_lighting, lighting_step, render_finish );

    int fps =1.f / (ms_raycast + ms_ssao + ms_aa + ms_lighting) * 1000.f;

    printf( "Frame: raycast %f ms, %d fps)\n", ms_raycast, fps );

    PerformanceCounter::update( "raycast", ms_raycast );
    PerformanceCounter::update( "ssna", ms_ssna );
    PerformanceCounter::update( "aa", ms_aa );
    PerformanceCounter::update( "ssao", ms_ssao );
    PerformanceCounter::update( "lighting", ms_lighting );
    PerformanceCounter::update( "frame", (float)fps );

    return true;
}

VOXOWL_HOST
bool
RaycasterCUDA::freeVolumeMem() {

    switch( d_volume.mode ) {
        case VOXELMAP:
            if( d_volume.volume.voxelmap.data ) {
                RETURN_IF_ERR( cudaDestroyTextureObject( d_volume.volume.voxelmap.texture ) );
                RETURN_IF_ERR( cudaFreeArray( d_volume.volume.voxelmap.data ) );
            }

            break;
        case SVMM:
            break;
        default:
            break;
    }
    return true;
}

VOXOWL_HOST
bool
RaycasterCUDA::freeFramebufferMem() {
    return true;
}

VOXOWL_HOST
bool
RaycasterCUDA::initSSAO() {
    static const int KERNEL_SIZE =64;
    static const int NOISE_SIZE =16;
    static const float RADIUS =.4f;
    // for VSAO
    //static const float RADIUS =20.0f;

    ssao_info.kernelSize =KERNEL_SIZE;
    ssao_info.noiseSize =NOISE_SIZE;
    ssao_info.radius =RADIUS;
    
    glm::vec4 kernel[ssao_info.kernelSize];
    glm::vec4 noise[ssao_info.noiseSize];


    for (int i = 0; i < ssao_info.kernelSize; ++i) {
        float scale = (float)i / (float)ssao_info.kernelSize;
        scale = glm::mix(0.1f, 1.0f, scale * scale);
        // for VSAO
        //scale = glm::mix(0.5f, 1.0f, scale * scale);
        
        kernel[i] = glm::normalize( glm::vec4 (
            random(-1.0f, 1.0f),
            random(-1.0f, 1.0f),
            random(0.f, 1.0f),
            0.f ) );
        kernel[i] *= random(0.1f, 1.0f);
        kernel[i] *= scale;
//        kernel[i].z += 1.f;
    }

    for (int i = 0; i < ssao_info.noiseSize; ++i) {
        noise[i] = glm::normalize( glm::vec4(
            random(-1.0f, 1.0f),
            random(-1.0f, 1.0f),
            0.0f, 0.0f  ) ) ;
    }

    cudaChannelFormatDesc fb_channelDesc = cudaCreateChannelDesc(32,32,32,32,cudaChannelFormatKindFloat);
    RETURN_IF_ERR( cudaMallocArray( &(ssao_info.noise), &fb_channelDesc, ssao_info.noiseSize, 1, 0 ) );
    ssao_noise.filterMode = cudaFilterModeLinear;
    ssao_noise.normalized = false;                      
    ssao_noise.addressMode[0] = cudaAddressModeWrap;   
    RETURN_IF_ERR( cudaBindTextureToArray( ssao_noise, ssao_info.noise ) );
    RETURN_IF_ERR( cudaMemcpyToArray( ssao_info.noise, 0, 0, noise, ssao_info.noiseSize * sizeof( float4 ), cudaMemcpyHostToDevice ) );

    RETURN_IF_ERR( cudaMallocArray( &(ssao_info.sampleKernel), &fb_channelDesc, ssao_info.kernelSize, 1, 0 ) );
    ssao_kernel.filterMode = cudaFilterModeLinear;
    ssao_kernel.normalized = false;                      
    ssao_kernel.addressMode[0] = cudaAddressModeWrap;   
    RETURN_IF_ERR( cudaBindTextureToArray( ssao_kernel, ssao_info.sampleKernel ) );
    RETURN_IF_ERR( cudaMemcpyToArray( ssao_info.sampleKernel, 0, 0, kernel, ssao_info.kernelSize * sizeof( float4 ), cudaMemcpyHostToDevice ) );
    
    return true;
}

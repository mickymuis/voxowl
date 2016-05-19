#define GLM_SWIZZLE 
//#define GLM_FORCE_CXX11
#include <stdio.h>
#include <stdint.h>
#include "platform.h"
#include "bmp.h"
#include "main-gpu.h"
#include <vector>
#include <fstream>
#include <glm/mat4x4.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtx/transform.hpp>
#include <glm/gtc/noise.hpp>
#include <vector_types.h>

#define ASSERT_GPU(err) { assertGpu((err), __FILE__, __LINE__); }
inline void
assertGpu(cudaError_t code, char *file, int line, bool abort=true)
{
    if (code != cudaSuccess) 
    {
        fprintf(stderr,"Error (cuda): %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) 
            exit(code);
    }
}

typedef struct {
    glm::vec3 origin;
    glm::vec3 direction;
} ray;

typedef struct {
    glm::vec3 min;
    glm::vec3 max;
} box;

typedef uint32_t voxel;
typedef uint32_t fragment;

typedef struct {
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

typedef struct {
    glm::ivec3 size;
    glm::mat4 matModel;
    cudaArray *data;
} volume_t;

typedef struct {
    int width, height;
    cudaArray *data;
    int aaXSamples, aaYSamples;
} framebuffer_t;

texture<voxel,3> volume_texture;
surface<void, 2> fb_surface; 

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

__device__ __host__
glm::vec4
unpackRGBA32( const uint32_t& rgba ) {
    return glm::vec4(
        (float)((rgba >> 24) & 0xff) / 255.f,
        (float)((rgba >> 16) & 0xff) / 255.f,
        (float)((rgba >> 8) & 0xff) / 255.f,
        (float)(rgba & 0xff) / 255.f );
}

__device__ __host__
glm::vec4
blendF2B( glm::vec4 src, glm::vec4 dst ) {
    glm::vec4 c;
    c.r = dst.a*(src.r * src.a) + dst.r;
    c.g = dst.a*(src.g * src.a) + dst.g;
    c.b = dst.a*(src.b * src.a) + dst.b;
    c.a = (1.f - src.a) * dst.a;
    return c;
}

__device__ __host__
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
                            for( uint32_t k = offs.z; k < offs.z + step; k++ ) {
                                glm::vec4 color =unpackRGBA32( *voxel3d( V, size, glm::ivec3(i,j,k) ) );
                                color.a =0.f;
                                packRGBA32( voxel3d( V, size, glm::ivec3( i, j, k )), color );
                            }
                }
                // corner element, expand recursively
                else
                    menger( V, size, new_cube, offs );
            }
}

__device__ __host__
void 
makeSponge( voxel* V, glm::ivec3 size ) {

    for( uint32_t x=0; x < size.x; x++ )
        for( uint32_t y=0; y < size.y; y++ )
            for( uint32_t z=0; z < size.z; z++ ) {
                packRGBA32(voxel3d(V,size,glm::ivec3(x,y,z)), glm::vec4(
                    (float)x / (float)(size.x-1),
                    (float)y / (float)(size.y-1),
                    (float)z / (float)(size.z-1),
                    1.f ));
            }
   
    menger( V, size, size, glm::ivec3(0) );
}

__device__ __host__
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
                    *voxel3d(V,size,glm::ivec3(x,y,z)) = (uint32_t)0xFFFFFFFF;
                else
                    *voxel3d(V,size,glm::ivec3(x,y,z)) = 0;
            }
}

__device__ __host__
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

__device__ __host__
glm::vec4
terrainLUT( float alt ) {
    return glm::vec4( bezier3( alt,
        glm::vec3( 0.4f, 0.4f, .1f ),
        glm::vec3( 0.7f, 0.5f, 0.2f ),
        glm::vec3( 0.4f, 1.0f, 0.2f ),
        glm::vec3( 0.8f, 1.0f, 0.6f ) ) , 1.0f );
}


__device__ __host__
void
makeTerrain( voxel *V, glm::ivec3 size ) {

    float sealevel =0.2f * size.y;
    glm::vec4 water( .4f, .8f, .9f, .2f );
    for( uint32_t x=0; x < size.x; x++ )
        for( uint32_t z=0; z < size.z; z++ ) {
            float noise_x = ((float)x / (float)size.x) * 2.f - 1.f;// * scale.x;
            float noise_z = ((float)z / (float)size.z) *2.f - 1.f;// * scale.z; 
            float density;
            
            packRGBA32( voxel3d(V,size,glm::ivec3(x,0,z)),
                    terrainLUT( 0 ) );

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
                        packRGBA32( voxel3d(V,size,glm::ivec3(x,y,z)),
                            terrainLUT( (noise_y + 1.f) / 2.f) );
                        continue;
                    }
                    else if( y < sealevel ) {
                        packRGBA32( voxel3d(V,size,glm::ivec3(x,y,z)), water );
                        continue;
                    }
                    else       
                        *voxel3d(V,size,glm::ivec3(x,y,z)) = 0;

            }   
        }
}

__device__ __host__
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

__device__ __host__
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

__device__
glm::vec4
raycast( volume_t v, const ray& r ) {
    double tmin, tmax;
    glm::ivec3 size =v.size;
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
        
       // glm::vec4 vox =unpackRGBA32( *voxel3d( V, size, index ) );
        glm::vec3 cont_index = glm::vec3(index);//(boxDist) * largest;
        glm::vec4 vox =unpackRGBA32( tex3D( volume_texture, cont_index.z, cont_index.y, cont_index.x ) );

        vox.r *= (3-side)/3.f;
        vox.g *= (3-side)/3.f;
        vox.b *= (3-side)/3.f;

        color =blendF2B( vox, color );

        if( vox.a == 1.f )
            break;


        // Branchless equivalent for
        //for( int i =0; i < 3; i++ ) 
        //    if( boxDist[side] > boxDist[i] )
        //        side =i;*/
        glm::bvec3 b0= glm::lessThan( boxDist, glm::vec3( boxDist.y, boxDist.z, boxDist.x ) /*boxDist.yzx()*/ );
        glm::bvec3 b1= glm::lessThanEqual( boxDist, glm::vec3( boxDist.z, boxDist.x, boxDist.y ) /*boxDist.zxy()*/ );
        glm::ivec3 mask =glm::ivec3( b0.x && b1.x, b0.y && b1.y, b0.z && b1.z );
        side = glm::dot( box_plane, mask );

        boxDist[side] += deltaDist[side];
        index[side] += step[side];
    }

    color.a = 1.f - color.a;
    return color;
}


__global__
void
computeFragment( raycastInfo_t raycast_info, volume_t volume, framebuffer_t framebuffer ) {
    const unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    glm::vec3 leftNormal = raycast_info.upperLeftNormal;
    glm::vec3 rightNormal = raycast_info.upperRightNormal;
       
    ray r;
    r.origin =raycast_info.origin;
    leftNormal += raycast_info.leftNormalYDelta * (float)y;
    rightNormal += raycast_info.rightNormalYDelta * (float)y;
    r.direction = leftNormal;
    
    glm::vec3 normalXDelta = (rightNormal - leftNormal) * raycast_info.invWidth;
    r.direction +=normalXDelta * (float)x;

    const float inv_aa_samples =1.f / (float)(framebuffer.aaXSamples * framebuffer.aaYSamples);
    glm::vec4 frag(0);

    for( int i =0; i < framebuffer.aaXSamples; i++ )
        for( int j =0; j < framebuffer.aaYSamples; j++ ) {

            // Shift the ray direction depending on the AA sample
            glm::vec3 raydir = r.direction 
                + (float)i / ( 1 * framebuffer.aaXSamples) * normalXDelta
                + (float)j / ( 1 * framebuffer.aaYSamples) * raycast_info.leftNormalYDelta;
    
            // Transform the ray from world-space to unit-cube-space
            ray r_cube;
            r_cube.direction =glm::normalize( glm::mat3( raycast_info.matInvModelView ) * raydir );
            r_cube.origin =r.origin;

            frag += inv_aa_samples * raycast( volume, r_cube );

//    ray r_cube2;
//    r_cube2.direction =r_cube.direction + normalXDelta / 2.0f;
//    r_cube2.origin =r_cube.origin;
        }

    // Cast the ray and set the framebuffer accordingly
//    glm::vec4 color1 = raycast( volume, r_cube );
//    glm::vec4 color2 = raycast( volume, r_cube2 );

    uint32_t rgba;
    packRGBA32( &rgba, frag );

//    FB[y*width + x] = rgba;



   // uint32_t color =tex3D( volume_texture, 0, x, y );
    surf2Dwrite( rgba, fb_surface, x*4, y, cudaBoundaryModeTrap );
}

extern int 
main_gpu( int argc, char **argv ) {

    const int width =1024;
    const int height =768;
    const dim3 blocksize(16, 16);
    const glm::ivec3 size( 243, 243, 243 );
    
    
    // Allocate the volume on the device
    volume_t d_volume;
    d_volume.size =size;
    const cudaExtent v_extent = make_cudaExtent( d_volume.size.x, d_volume.size.y, d_volume.size.z );
//    cudaChannelFormatDesc v_channelDesc = cudaCreateChannelDesc<voxel>();
    cudaChannelFormatDesc v_channelDesc = cudaCreateChannelDesc(32,0,0,0,cudaChannelFormatKindUnsigned);
    ASSERT_GPU( cudaMalloc3DArray( &d_volume.data, &v_channelDesc, v_extent ) );
    volume_texture.normalized = false;                      
    volume_texture.filterMode = cudaFilterModePoint;      
    volume_texture.addressMode[0] = cudaAddressModeClamp;   
    volume_texture.addressMode[1] = cudaAddressModeClamp;
    volume_texture.addressMode[2] = cudaAddressModeClamp;
    ASSERT_GPU( cudaBindTextureToArray( volume_texture, d_volume.data, v_channelDesc ) );
    
// Allocate the framebuffer on the device
    framebuffer_t d_fb;
    d_fb.width = width;
    d_fb.height = height;
    d_fb.aaXSamples =4;
    d_fb.aaYSamples =4;
//    const cudaExtent fb_extent = make_cudaExtent( width, height, 1 );
    cudaChannelFormatDesc fb_channelDesc = cudaCreateChannelDesc(32,0,0,0,cudaChannelFormatKindUnsigned);
    ASSERT_GPU( cudaMallocArray( &(d_fb.data), &fb_channelDesc, width, height, cudaArraySurfaceLoadStore ) );
    ASSERT_GPU( cudaBindSurfaceToArray( fb_surface, d_fb.data ) );

    // Setup the volume on the CPU and upload to the device
    voxel* V = (voxel*)malloc( sizeof( voxel ) * size.x * size.y * size.z );
    makeSponge( V, size );
    cudaMemcpy3DParms copyParams = {0};
    copyParams.srcPtr   = make_cudaPitchedPtr((void*)V, v_extent.width*sizeof(voxel), v_extent.width, v_extent.height);
    copyParams.dstArray = d_volume.data;
    copyParams.extent   = v_extent;
    copyParams.kind     = cudaMemcpyHostToDevice;
    ASSERT_GPU( cudaMemcpy3D(&copyParams) );

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

    // Setup the arguments to the raycaster
    raycastInfo_t raycast_info;
    raycast_info.upperLeftNormal =glm::normalize( glm::vec3( left, top, -near ) );
    raycast_info.upperRightNormal =glm::normalize( glm::vec3( right, top, -near ) );
    raycast_info.lowerLeftNormal =glm::normalize( glm::vec3( left, bottom, -near ) );
    raycast_info.lowerRightNormal =glm::normalize( glm::vec3( right, bottom, -near ) );
    

    // Calculate the ray-normal interpolation constants
    raycast_info.invHeight = 1.f / (float)height;
    raycast_info.invWidth = 1.f / (float)width;

    raycast_info.leftNormalYDelta = (raycast_info.lowerLeftNormal - raycast_info.upperLeftNormal) * raycast_info.invHeight;
    raycast_info.rightNormalYDelta =(raycast_info.lowerRightNormal - raycast_info.upperRightNormal) * raycast_info.invHeight;

    // Setup the host framebuffer
    uint32_t* FB = (uint32_t*)malloc( sizeof(uint32_t) * width * height );
//    for( uint32_t i=0; i < width*height; i++ ) FB[i] = 0x00ff00ff;

    // Ray-cast main-loop
    glm::mat4 mat_model(1.f);
 //   glm::mat4 mat_model =glm::rotate( 45.0f, glm::vec3(0,1,0) );
//    glm::mat4 mat_model =glm::scale( glm::vec3(2.5f) );
    glm::mat4 mat_modelview =mat_view * mat_model;
    glm::mat4 mat_inv_modelview =glm::inverse( mat_modelview );
    glm::vec3 leftNormal = raycast_info.upperLeftNormal;
    glm::vec3 rightNormal = raycast_info.upperRightNormal;
   // ray r;
    raycast_info.origin = glm::vec3( mat_inv_modelview * glm::vec4(0,0,0,1) );
    raycast_info.matInvModelView = mat_inv_modelview;
//    r.origin = glm::vec3(0);
    //printVec( r.origin);

/*    for( int y=0; y < height; y++ ) {
        
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
    }*/

    // Divide the invidual fragments over N / blocksize blocks
    // Run the raycast kernel on the device
    const dim3 numblocks( width / blocksize.x, height / blocksize.y );
    ASSERT_GPU( cudaBindSurfaceToArray( fb_surface, d_fb.data ) );
    computeFragment<<<numblocks, blocksize>>>( raycast_info, d_volume, d_fb );
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) 
            printf("Error: %s\n", cudaGetErrorString(err));
    ASSERT_GPU( cudaDeviceSynchronize() );

    // Copy the framebuffer to the host
    ASSERT_GPU ( cudaMemcpyFromArray( FB, d_fb.data, 0, 0, width*height*sizeof(fragment), cudaMemcpyDeviceToHost ) ); 
//    ASSERT_GPU ( cudaMemcpy2DFromArray( FB, 0, d_fb.data, 0, 0, width*sizeof(fragment), height, cudaMemcpyDeviceToHost ) );

    printf( "FB[9999]: 0x%x\n", FB[9999] );

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

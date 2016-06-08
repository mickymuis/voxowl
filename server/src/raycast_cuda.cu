#include "raycast_cuda.h"
#include "platform.h"

#include "framebuffer.h"
#include "volume.h"
#include "camera.h"

#include "voxel.h"
#include <stdio.h>
#include <stdint.h>
#include <sstream>
#include <glm/mat4x4.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtx/transform.hpp>
//#include <glm/gtc/noise.hpp>
#include <vector_types.h>
#include "bmp.h"
#include <fstream>

// Define global volume and framebuffer handles, for now at least
texture<uint32_t,3> volume_texture;
surface<void, 2> fb_surface; 

// Simple return-on-error mechanism to improve readability
#define RETURN_IF_ERR(err) { if( setCudaErrorStr((err), __FILE__, __LINE__) ) return false; }
inline bool
RaycasterCUDA::setCudaErrorStr(cudaError_t code, char *file, int line )
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

/* Cast one ray r into the volume bounded by v. The actual volume data is obtained from the global volume texture */
VOXOWL_DEVICE
glm::vec4
raycast( volumeDevice_t v, const ray_t& r ) {
    double tmin, tmax;
    glm::ivec3 size =v.size;
    box_t b = volumeSizeToAABB( size );
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
        glm::vec4 vox =unpackRGBA_UINT32( tex3D( volume_texture, cont_index.z, cont_index.y, cont_index.x ) );

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

/* Parallel raycast kernel. Computes one fragment depending on position in the threadblock and writes in to the framebuffer */
VOXOWL_CUDA_KERNEL
void
computeFragment( raycastInfo_t raycast_info, volumeDevice_t volume, framebufferDevice_t framebuffer ) {
    const unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    glm::vec3 leftNormal = raycast_info.upperLeftNormal;
    glm::vec3 rightNormal = raycast_info.upperRightNormal;
       
    ray_t r;
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
            ray_t r_cube;
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
    packRGBA_UINT32( &rgba, frag );

//    FB[y*width + x] = rgba;



   // uint32_t color =tex3D( volume_texture, 0, x, y );
//    surf2Dwrite( rgba, fb_surface, x*4, y, cudaBoundaryModeTrap );
    surf2Dwrite<uint8_t>( (uint8_t)( (rgba >> 24) & 0xFF), fb_surface, x*3, y, cudaBoundaryModeTrap );
    surf2Dwrite<uint8_t>( (uint8_t)( (rgba >> 16) & 0xFF), fb_surface, x*3+1, y, cudaBoundaryModeTrap );
    surf2Dwrite<uint8_t>( (uint8_t)( (rgba >> 8) & 0xFF), fb_surface, x*3+2, y, cudaBoundaryModeTrap );
}

/*bool
framebufferDeviceFree( framebufferDevice_t * );*/

RaycasterCUDA::RaycasterCUDA( const char* name, Object* parent ) 
    : Renderer( name, parent ) {
    bzero( &d_volume, sizeof( volumeDevice_t ) );
    bzero( &d_framebuffer, sizeof( framebufferDevice_t ) );
}

RaycasterCUDA::~RaycasterCUDA() {

        if( d_framebuffer.data )
            cudaFreeArray( d_framebuffer.data );
        if( d_volume.data )
            cudaFreeArray( d_volume.data ); 
}

bool 
RaycasterCUDA::beginRender() {

    if( !getFramebuffer() )
        return setError( true, "No framebuffer set" );
    if( !getVolume() )
        return setError( true, "No input volume set" );
    if( !getCamera() )
        return setError( true, "No camera set" );

    const int width =getFramebuffer()->getWidth();
    const int height =getFramebuffer()->getHeight();
    const dim3 blocksize(16, 16);

    // For now, we only use a voxelmap as input
    voxelmap_t voxelmap =getVolume()->data();
    
    // Allocate the volume on the device, if neccesary
    bool realloc_volume = !d_volume.data || ( d_volume.size != voxelmap.size ) || ( d_volume.format != voxelmap.format );

    if( realloc_volume ) { // Reallocate the volume on the device end
        if( d_volume.data )
            RETURN_IF_ERR( cudaFreeArray( d_volume.data ) );

        d_volume.size =voxelmap.size;
        const cudaExtent v_extent = make_cudaExtent( d_volume.size.x, d_volume.size.y, d_volume.size.z );
        // TODO: use format
        cudaChannelFormatDesc v_channelDesc = cudaCreateChannelDesc(32,0,0,0,cudaChannelFormatKindUnsigned);
        RETURN_IF_ERR( cudaMalloc3DArray( &d_volume.data, &v_channelDesc, v_extent ) );
        volume_texture.normalized = false;                      
        volume_texture.filterMode = cudaFilterModePoint;      
        volume_texture.addressMode[0] = cudaAddressModeClamp;   
        volume_texture.addressMode[1] = cudaAddressModeClamp;
        volume_texture.addressMode[2] = cudaAddressModeClamp;
        RETURN_IF_ERR( cudaBindTextureToArray( volume_texture, d_volume.data, v_channelDesc ) );
        
        // Copy the volume to the device
        // TODO: make separate step
        cudaMemcpy3DParms copyParams = {0};
        copyParams.srcPtr   = make_cudaPitchedPtr( voxelmap.data, v_extent.width * voxelSize(voxelmap.format), v_extent.width, v_extent.height);
        copyParams.dstArray = d_volume.data;
        copyParams.extent   = v_extent;
        copyParams.kind     = cudaMemcpyHostToDevice;
        RETURN_IF_ERR( cudaMemcpy3D(&copyParams) );
    }
    
    // Allocate the framebuffer on the device, if neccesary
    bool realloc_framebuffer = !d_framebuffer.data 
        || ( d_framebuffer.width != width ) 
        || ( d_framebuffer.height != height )
        || ( (int)d_framebuffer.format != getFramebuffer()->getPixelFormat() );
    
    d_framebuffer.aaXSamples =getFramebuffer()->getAAXSamples();
    d_framebuffer.aaYSamples =getFramebuffer()->getAAYSamples();

    if( realloc_framebuffer ) { // Reallocate the framebuffer on the device end
        if( d_framebuffer.data )
            RETURN_IF_ERR( cudaFreeArray( d_framebuffer.data ) );

        d_framebuffer.width = width;
        d_framebuffer.height = height;
        d_framebuffer.format =(voxowl_pixel_format_t)getFramebuffer()->getPixelFormat();
        cudaChannelFormatDesc fb_channelDesc = cudaCreateChannelDesc(8,0,0,0,cudaChannelFormatKindUnsigned);
        RETURN_IF_ERR( cudaMallocArray( &(d_framebuffer.data), &fb_channelDesc, width * 3, height, cudaArraySurfaceLoadStore ) );
        RETURN_IF_ERR( cudaBindSurfaceToArray( fb_surface, d_framebuffer.data ) );
    }

    
    // Setup the raycast parameters based on the matrices from the camera and the 'model'
    // TODO: some kind of caching?
    raycastInfo_t raycast_info;
    getCamera()->setAspect( (float)width/(float)height );
    raycastSetMatrices( &raycast_info, getVolume()->modelMatrix(), getCamera()->getViewMatrix(), getCamera()->getProjMatrix(), width, height );

    // Divide the invidual fragments over N / blocksize blocks
    // Run the raycast kernel on the device
    const dim3 numblocks( width / blocksize.x, height / blocksize.y );
    RETURN_IF_ERR( cudaBindSurfaceToArray( fb_surface, d_framebuffer.data ) );
    computeFragment<<<numblocks, blocksize>>>( raycast_info, d_volume, d_framebuffer );
    RETURN_IF_ERR( cudaGetLastError() );

    return true;
}

bool 
RaycasterCUDA::synchronize() {
    // Wait for the running kernel to finish
    RETURN_IF_ERR( cudaDeviceSynchronize() );

    // Copy the framebuffer to the host

    // TODO: use framebuffer format
    int bpp =3;
    void* data_ptr =getFramebuffer()->data();
    int width =getFramebuffer()->getWidth();
    int height =getFramebuffer()->getHeight();


    RETURN_IF_ERR ( cudaMemcpyFromArray( data_ptr, d_framebuffer.data, 0, 0, width*height*bpp, cudaMemcpyDeviceToHost ) );
    
//    uint32_t* FB = (uint32_t*)malloc( sizeof(uint32_t) * width * height );
//    RETURN_IF_ERR ( cudaMemcpyFromArray( FB, d_framebuffer.data, 0, 0, width*height*4, cudaMemcpyDeviceToHost ) );

    // Write the buffer as a BMP, for testing
    std::vector<uint8_t> output;
    size_t output_size =bitmap_encode_multichannel_8bit( (const uint8_t*)data_ptr, width, height, 3, output );
    //size_t output_size =bitmap_encode_rgba( (const uint32_t*)FB, width, height, output );

    std::ofstream file_output;
    file_output.open("../buffer.bmp");
    file_output.write((const char*)&output[0], output_size);
    file_output.close();

    return true;
}

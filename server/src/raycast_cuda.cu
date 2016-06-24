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
#include <glm/gtc/random.hpp>
#include <vector_types.h>
#include "bmp.h"
#include <fstream>

// Define global volume and framebuffer handles, for now at least
texture<uint32_t,3> volume_texture;
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

VOXOWL_DEVICE
glm::vec4
voxel( volumeDevice_t v, glm::ivec3 index ) {
    glm::ivec3 block =glm_ivec3_16( blockPosition( v.format, ivec3_16( index ) ) );
    glm::vec4 vox;

    switch( v.format ) {
        case VOXEL_RGBA_UINT32: {
            vox =unpackRGBA_UINT32( tex3D( volume_texture, block.x, block.y, block.z ) );
            break;
        }
        case VOXEL_INTENSITY_UINT8: {
            uint8_t gray =tex3D( volume_texture, block.x, block.y, block.z );
            vox =glm::vec4( unpackINTENSITY_UINT8( gray ) );
            vox.a =(float)(vox.r != 0.f);
            break;
        }
        case VOXEL_BITMAP_UINT8: {
            uint8_t bitmap =tex3D( volume_texture, block.x, block.y, block.z );
            int bit_offs =index.z % voxelsPerBlock( v.format );
            //int bit_offs =index.z - block.z*8;
            vox =glm::vec4( (int)unpackBIT_UINT8( bitmap, bit_offs ) );
            break;
        }
        case VOXEL_RGB24_8ALPHA1_UINT32: {
            uint32_t rgb24_8alpha1 =tex3D( volume_texture, block.x, block.y, block.z );
            int bit_offs =index.z % voxelsPerBlock( v.format );
            vox =unpackRGBA_RGB24_8ALPHA1_UINT32( rgb24_8alpha1, bit_offs );

            break;
        }
        case VOXEL_RGB24A1_UINT32: {
            vox =unpackRGB24A1_UINT32( tex3D( volume_texture, block.x, block.y, block.z ) );
            break;
        }
    }
    vox.a *= (float)( glm::all( glm::greaterThanEqual( index, glm::ivec3(0) ) ) && glm::all( glm::lessThan( index, v.size ) ) );

    return vox;
}

/* Cast one ray r into the volume bounded by v. The actual volume data is obtained from the global volume texture */
VOXOWL_DEVICE
fragment_t
raycast( volumeDevice_t v, const ray_t& r ) {
    double tmin, tmax;
    glm::ivec3 size =v.size;
    
    fragment_t frag;
    frag.color =glm::vec4( 0,0,0,1 );
    frag.normal =glm::vec3(0);
    frag.position =glm::vec3(0);
    frag.position_vs =glm::vec3(0);

    box_t b = volumeSizeToAABB( size );
    if( !rayAABBIntersect( r, b, tmin, tmax ) )
        return frag;

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
    glm::vec3 rayEntry_vs =(rayEntry + b.max) * largest;

    // Calculate the index in the volume by chopping off the decimal part
    glm::ivec3 index = glm::clamp( 
        glm::ivec3( glm::floor( rayEntry_vs ) ) ,
        glm::ivec3( 0 ),
        glm::ivec3( size.x-1, size.y-1, size.z-1 ) );

    frag.position_vs = glm::clamp( 
        glm::vec3( rayEntry_vs ) ,
        glm::vec3( 0 ),
        glm::vec3( size.x-1, size.y-1, size.z-1 ) );

    // Determine the sign of the stepping through the volume
    glm::ivec3 step = glm::sign( r.direction );

    // deltaDist gives the distance on the ray path for each following dividing plane
    glm::vec3 deltaDist =glm::abs( glm::vec3( glm::length( r.direction ) ) / r.direction );

    // Computes the distances to the next voxel for each component
    glm::vec3 boxDist = ( sign( r.direction ) * (glm::vec3(index) - rayEntry_vs)
                        + (sign( r.direction ) * 0.5f ) + 0.5f ) * deltaDist;

    while(1) {

        if( index[side] < 0 || index[side] >= size[side] )
            break;
        
        glm::vec4 vox = voxel( v, glm::ivec3( glm::floor( frag.position_vs ) ) );
/*        vox.r *= (3-side)/3.f;
        vox.g *= (3-side)/3.f;
        vox.b *= (3-side)/3.f;*/

        frag.color =blendF2B( vox, frag.color );

        if( vox.a == 1.f ) {
            // We calculate the position in unit-cube space..
            frag.position =frag.position_vs / (float)largest - b.max;
            // ..and the normal of the current 'face' of the voxel
            frag.normal[side] = -step[side];
            break;
        }


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
        frag.position_vs[side] += step[side];
    }

    frag.color.a = 1.f - frag.color.a;

    return frag;
}

/* Parallel raycast kernel. Computes one fragment depending on position in the threadblock and writes in to the framebuffer */
VOXOWL_CUDA_KERNEL
void
computeFragment( raycastInfo_t raycast_info, volumeDevice_t volume, framebufferDevice_t framebuffer, ssaoInfo_t ssao_info, glm::mat4 mat_projection ) {
    // Calculate screen coordinates
    const unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    // Next, we calculate the ray vector based on the screen coordinates
    glm::vec3 leftNormal = raycast_info.upperLeftNormal;
    glm::vec3 rightNormal = raycast_info.upperRightNormal;
       
    ray_t r;
    r.origin =raycast_info.origin;
    leftNormal += raycast_info.leftNormalYDelta * (float)y;
    rightNormal += raycast_info.rightNormalYDelta * (float)y;
    r.direction = leftNormal;
    
    glm::vec3 normalXDelta = (rightNormal - leftNormal) * raycast_info.invWidth;
    r.direction +=normalXDelta * (float)x;

    // Initialize variables used during sampling
    const float inv_aa_samples =1.f / (float)(framebuffer.aaXSamples * framebuffer.aaYSamples);
    fragment_t frag;
    frag.color =glm::vec4(0);
    frag.position =glm::vec3(0);
    frag.position_vs =glm::vec3(0);
    frag.normal =glm::vec3(0);

    for( int i =0; i < framebuffer.aaXSamples; i++ )
        for( int j =0; j < framebuffer.aaYSamples; j++ ) {

            // Shift the ray direction depending on the AA sample
            glm::vec3 raydir = r.direction 
                + (float)i /*/ ( 1 * framebuffer.aaXSamples)*/ * normalXDelta
                + (float)j /*/ ( 1 * framebuffer.aaYSamples)*/ * raycast_info.leftNormalYDelta;
    
            // Transform the ray from world-space to unit-cube-space
            ray_t r_cube;
            r_cube.direction =glm::normalize( glm::mat3( raycast_info.matInvModelView ) * raydir );
            r_cube.origin =r.origin;
            
            // Cast the ray and average it with the other samples
            fragment_t f = raycast( volume, r_cube );
            frag.color += f.color;
            frag.position += f.position;
            frag.position_vs +=f.position_vs;
            frag.normal +=  f.normal;

        }

    // Average out
    frag.color *=inv_aa_samples;
    frag.position *=inv_aa_samples;
    frag.position_vs *=inv_aa_samples;
    frag.normal *=inv_aa_samples;

    // VSAO
    float4 noise =tex1D( ssao_noise, (float)x*y );
    glm::vec3 rvec( noise.x, noise.y, noise.z );

    // Setup the TBN matrix
    glm::vec3 tangent = glm::normalize(rvec - frag.normal * glm::dot(rvec, frag.normal));
    glm::vec3 bitangent = glm::cross(frag.normal, tangent);
    glm::mat3 tbn(tangent, bitangent, frag.normal);

    // Obtain the samples
    float occlusion =0.f;
    for (int i = 0; i < ssao_info.kernelSize; ++i) {
        // get sample position:

        float4 k =tex1D( ssao_kernel, i );
        glm::vec3 kernel_i( k.x, k.y, k.z );
        glm::vec3 sample = tbn * kernel_i;
        sample = sample * ssao_info.radius + frag.position_vs + frag.normal;
        
        glm::vec4 vox =voxel( volume, glm::floor( sample ) );
        occlusion += vox.a;
    }

    
    frag.color *= (1.f - occlusion / ssao_info.kernelSize );


    // Convert both the position and the normal to view-space
    // We export (some of) these values for use in later passes
    frag.position = glm::vec3( glm::mat3(raycast_info.matModelView) * ( frag.position - r.origin ) );
    frag.normal = glm::normalize( glm::mat3( raycast_info.matModelView ) * frag.normal );
    float depth =frag.position.z;

    
    // Write the color information to the framebuffer
    uint32_t rgba;
//    packRGBA_UINT32( &rgba, glm::vec4( frag.position.z, frag.position.z, frag.position.z, 1.f ) );
    packRGBA_UINT32( &rgba, frag.color  );

    // Workaround to be able to write to a 24bit buffer. Saves conversion later
    surf2Dwrite<uint8_t>( (uint8_t)( (rgba >> 24) & 0xFF), fb_color_surface, x*3, y, cudaBoundaryModeTrap );
    surf2Dwrite<uint8_t>( (uint8_t)( (rgba >> 16) & 0xFF), fb_color_surface, x*3+1, y, cudaBoundaryModeTrap );
    surf2Dwrite<uint8_t>( (uint8_t)( (rgba >> 8) & 0xFF), fb_color_surface, x*3+2, y, cudaBoundaryModeTrap );


    // Write the normal and depth values using a regular 32 float4 texture
    float4 normal_depth;
    normal_depth.w =depth;
    normal_depth.x =frag.normal.x;
    normal_depth.y =frag.normal.y;
    normal_depth.z =frag.normal.z;

    surf2Dwrite<float4>( normal_depth, fb_normal_depth_surface, x * sizeof( float4 ), y, cudaBoundaryModeTrap ); 

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
    float4 normal_depth =surf2Dread<float4>( fb_normal_depth_surface, x * sizeof( float4 ), y, cudaBoundaryModeTrap );
    glm::vec3 normal( normal_depth.x, normal_depth.y, normal_depth.z );
    
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
    origin =origin / -origin.z * normal_depth.w;


    // Obtain a random number to rotate the sampple matrix
    float4 noise =tex1D( ssao_noise, (float)x*y );
    glm::vec3 rvec( noise.x, noise.y, noise.z );

    // Setup the TBN matrix
    glm::vec3 tangent = glm::normalize(rvec - normal * glm::dot(rvec, normal));
    glm::vec3 bitangent = glm::cross(normal, tangent);
    glm::mat3 tbn(tangent, bitangent, normal);

    // Obtain the samples
    for (int i = 0; i < ssao_info.kernelSize; ++i) {
        // get sample position:

        float4 k =tex1D( ssao_kernel, i );
        glm::vec3 kernel_i( k.x, k.y, k.z );
        glm::vec3 sample = tbn * kernel_i;
        sample = sample * ssao_info.radius + origin;

        // project sample position:
        glm::vec4 offset = mat_projection * glm::vec4( origin, 1.f );
        glm::vec2 screenpos = glm::vec2( offset ) / offset.w ;
        screenpos.y =-screenpos.y;
        screenpos = screenpos * 0.5f + glm::vec2(0.5f);
        screenpos *= glm::vec2( framebuffer.width, framebuffer.height );

        // get sample depth:
        float4 normal_depth =surf2Dread<float4>( fb_normal_depth_surface, 
            (int)screenpos.x * sizeof( float4 ), 
            (int)screenpos.y, 
            cudaBoundaryModeClamp );
        float sampleDepth = normal_depth.w;

        // range check & accumulate:
        float rangeCheck= glm::abs(origin.z - sampleDepth) < ssao_info.radius ? 1.0 : 0.0;
        occlusion += (sampleDepth >= sample.z ? 1.0 : 0.0) * rangeCheck;

 //       if( r > 1 ) {
 //           printf( "sample depth %f sample %f %f %f origin %f %f %f screenpos %f %f normal %f %f %f\n", sampleDepth, sample.x, sample.y, sample.z, origin.x, origin.y, origin.z, screenpos.x, screenpos.y, normal.x, normal.y, normal.z );
    //    }
    }

    // Apply the result
    occlusion = 1.f;// - occlusion / ssao_info.kernelSize;


    glm::vec3 color( r, g, b );
    color /= glm::vec3( 255.f );
    color *= occlusion;
    
    surf2Dwrite<uint8_t>( color.r * 255.f, fb_color_surface, x*3, y, cudaBoundaryModeTrap );
    surf2Dwrite<uint8_t>( color.g * 255.f, fb_color_surface, x*3+1, y, cudaBoundaryModeTrap );
    surf2Dwrite<uint8_t>( color.b * 255.f, fb_color_surface, x*3+2, y, cudaBoundaryModeTrap );
}

RaycasterCUDA::RaycasterCUDA( const char* name, Object* parent ) 
    : Renderer( name, parent ) {
    bzero( &d_volume, sizeof( volumeDevice_t ) );
    bzero( &d_framebuffer, sizeof( framebufferDevice_t ) );
    if (!initSSAO() ) {
        fprintf( stderr, "initSSAO(): %s\n", errorString().c_str() ); 
    }
}

RaycasterCUDA::~RaycasterCUDA() {

        if( d_framebuffer.color_data )
            cudaFreeArray( d_framebuffer.color_data );
        if( d_framebuffer.normal_depth_data )
            cudaFreeArray( d_framebuffer.normal_depth_data );
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
    bool realloc_volume = !d_volume.data || ( d_volume.size != glm_ivec3_16( voxelmap.size ) ) || ( d_volume.format != voxelmap.format );

    if( realloc_volume ) { // Reallocate the volume on the device end
        if( d_volume.data )
            RETURN_IF_ERR( cudaFreeArray( d_volume.data ) );

        d_volume.size =glm_ivec3_16( voxelmap.size );
        d_volume.blocks =glm_ivec3_16( voxelmap.blocks );
        d_volume.format =voxelmap.format;

        cudaExtent v_extent;
        cudaChannelFormatDesc v_channelDesc;

        printf( "Allocating texture for voxelmap, blocks=(%d,%d,%d), bytes per block=%d\n", voxelmap.blocks.x, voxelmap.blocks.y, voxelmap.blocks.z, bytesPerBlock( voxelmap.format ) );

        switch( voxelmap.format ) {
            // We differentiate between byte and word block sizes
            case VOXEL_RGB24A1_UINT32:
            case VOXEL_RGB24_8ALPHA1_UINT32:
            case VOXEL_RGBA_UINT32:
                v_extent = make_cudaExtent( d_volume.blocks.x, d_volume.blocks.y, d_volume.blocks.z );
                v_channelDesc = cudaCreateChannelDesc(32,0,0,0,cudaChannelFormatKindUnsigned);
                break;

            case VOXEL_INTENSITY_UINT8:
            case VOXEL_BITMAP_UINT8:
                // By convention, we store blocks per-byte in the z-direction
                v_extent = make_cudaExtent( d_volume.blocks.x, d_volume.blocks.y, d_volume.blocks.z );
                v_channelDesc = cudaCreateChannelDesc(8,0,0,0,cudaChannelFormatKindUnsigned);
                break;

            default:
                return setError( true, "Unhandled voxel format" );
                
        }

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
        copyParams.srcPtr   = make_cudaPitchedPtr( voxelmap.data, v_extent.width * bytesPerBlock(voxelmap.format), v_extent.width, v_extent.height);
        copyParams.dstArray = d_volume.data;
        copyParams.extent   = v_extent;
        copyParams.kind     = cudaMemcpyHostToDevice;
        RETURN_IF_ERR( cudaMemcpy3D(&copyParams) );
    }
    
    // Allocate the framebuffer on the device, if neccesary
    bool realloc_framebuffer = !d_framebuffer.color_data 
        || ( d_framebuffer.width != width ) 
        || ( d_framebuffer.height != height )
        || ( (int)d_framebuffer.format != getFramebuffer()->getPixelFormat() );
    
    d_framebuffer.aaXSamples =getFramebuffer()->getAAXSamples();
    d_framebuffer.aaYSamples =getFramebuffer()->getAAYSamples();

    if( realloc_framebuffer ) { // Reallocate the framebuffer on the device end
        if( d_framebuffer.color_data )
            RETURN_IF_ERR( cudaFreeArray( d_framebuffer.color_data ) );
        if( d_framebuffer.normal_depth_data )
            RETURN_IF_ERR( cudaFreeArray( d_framebuffer.normal_depth_data ) );

        d_framebuffer.width = width;
        d_framebuffer.height = height;
        d_framebuffer.format =(voxowl_pixel_format_t)getFramebuffer()->getPixelFormat();

        int bytes_per_pixel;
        // TODO: this could use a nice function
        if( d_framebuffer.format == VOXOWL_PF_RGB888 )
            bytes_per_pixel =3;
        else
            return false;

        // We allocate one buffer for the color data and optionally another for depth/normal data
        cudaChannelFormatDesc fb_channelDesc = cudaCreateChannelDesc(8,0,0,0,cudaChannelFormatKindUnsigned);
        RETURN_IF_ERR( cudaMallocArray( &(d_framebuffer.color_data), &fb_channelDesc, width * bytes_per_pixel, height, cudaArraySurfaceLoadStore ) );
        RETURN_IF_ERR( cudaBindSurfaceToArray( fb_color_surface, d_framebuffer.color_data ) );
        
        // We use a 32bit 4float type for the normal+depth buffer
        cudaChannelFormatDesc fb_channelDesc2 = cudaCreateChannelDesc(32,32,32,32,cudaChannelFormatKindFloat);
        RETURN_IF_ERR( cudaMallocArray( &(d_framebuffer.normal_depth_data), &fb_channelDesc2, width, height, cudaArraySurfaceLoadStore ) );
        RETURN_IF_ERR( cudaBindSurfaceToArray( fb_normal_depth_surface, d_framebuffer.normal_depth_data ) );
    }

    
    // Setup the raycast parameters based on the matrices from the camera and the 'model'
    // TODO: some kind of caching?
    raycastInfo_t raycast_info;
    getCamera()->setAspect( (float)width/(float)height );
    raycastSetMatrices( &raycast_info, getVolume()->modelMatrix(), getCamera()->getViewMatrix(), getCamera()->getProjMatrix(), width, height );

    // Divide the invidual fragments over N / blocksize blocks
    // Run the raycast kernel on the device
    const dim3 numblocks( width / blocksize.x, height / blocksize.y );
    RETURN_IF_ERR( cudaBindSurfaceToArray( fb_color_surface, d_framebuffer.color_data ) );
    RETURN_IF_ERR( cudaBindSurfaceToArray( fb_normal_depth_surface, d_framebuffer.normal_depth_data ) );
    computeFragment<<<numblocks, blocksize>>>( raycast_info, d_volume, d_framebuffer, ssao_info, getCamera()->getProjMatrix() );
    RETURN_IF_ERR( cudaGetLastError() );

    // Experimental SSAO step
    RETURN_IF_ERR( cudaDeviceSynchronize() );
    computeFragmentSSAO<<<numblocks, blocksize>>>( raycast_info, ssao_info, d_framebuffer, getCamera()->getProjMatrix() );
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


    RETURN_IF_ERR ( cudaMemcpyFromArray( data_ptr, d_framebuffer.color_data, 0, 0, width*height*bpp, cudaMemcpyDeviceToHost ) );
    

    return true;
}

VOXOWL_HOST
bool
RaycasterCUDA::initSSAO() {
    static const int KERNEL_SIZE =512;
    static const int NOISE_SIZE =64;
    //static const float RADIUS =1.f;
    // for VSAO
    static const float RADIUS =20.0f;

    ssao_info.kernelSize =KERNEL_SIZE;
    ssao_info.noiseSize =NOISE_SIZE;
    ssao_info.radius =RADIUS;
    
    glm::vec4 kernel[ssao_info.kernelSize];
    glm::vec4 noise[ssao_info.noiseSize];


    for (int i = 0; i < ssao_info.kernelSize; ++i) {
        float scale = (float)i / (float)ssao_info.kernelSize;
        //scale = glm::mix(0.1f, 1.0f, scale * scale);
        // for VSAO
        scale = glm::mix(0.5f, 1.0f, scale * scale);
        
        kernel[i] = glm::normalize( glm::vec4 (
            random(-1.0f, 1.0f),
            random(-1.0f, 1.0f),
            random(0.0f, 1.0f),
            0.f ) );
//        kernel[i] *= random(0.1f, 1.0f);
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

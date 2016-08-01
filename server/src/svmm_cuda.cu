#include "svmm_cuda.h"
#include "dda_cuda.h"
#include <voxowl_platform.h>
#include <driver_types.h>
#include <cmath>
#include <voxel.h>
#include <glm/gtx/extented_min_max.hpp>

static cudaError_t cuda_error;
#define RETURN_IF_ERR(statement) if( (cuda_error=statement) ) { \
    fprintf( stderr, "Error: '%s' in %s on line %d\n", \
            cudaGetErrorString( cuda_error ), \
            __FILE__, __LINE__ );\
    return cuda_error; }

VOXOWL_HOST
static inline uint32_t 
p2(uint32_t x)
{
    return 1 << (32 - __builtin_clz (x - 1));
}

VOXOWL_HOST
static inline uint32_t 
log2_p2(uint32_t x)
{
    return 32 - __builtin_clz (x - 1);
}

VOXOWL_HOST
glm::ivec3
calcGridSize( uint32_t block_count ) {
    // Wanted: x, y, z such that:
    // x*y*z >= n
    // x and y are powers of two
    
    uint32_t x, y, z;
    uint32_t f =(int32_t)std::cbrt( (float)block_count );
    x =y =p2( f );
    uint32_t g =x*y;
    z = block_count/g;
    if( block_count % g )
        z++;
    return glm::ivec3( x,y,z );
}

/*VOXOWL_DEVICE
svmm_level_t*
getLevel( svmipmapDevice_t* v, int i ) {
    return v->level_ptr+i;
}*/

VOXOWL_HOST_AND_DEVICE
static inline glm::ivec3
gridOffset( svmm_level_t* level, int block_num ) {
    glm::ivec3 grid_offset;
    block_num -=(grid_offset.z =(block_num >> level->grid_z_shift)) << level->grid_z_shift;
    block_num -=(grid_offset.y =(block_num >> level->grid_y_shift)) << level->grid_y_shift;
    grid_offset.x =block_num;
    return grid_offset;
}

VOXOWL_DEVICE
uint64_t
getBlockOffsetTex3D( cudaTextureObject_t texture, 
                     glm::ivec3 subblock_index, 
                     glm::ivec3 block_offset ) {
    uint64_t offset =0;
    int skip =0;
    int n =0;
    // block's relative index in column-major order
    int block_idx =block_offset.z * 4 + block_offset.y * 2 + block_offset.x;

    // TODO: Optimize this, we can do this without loop or voxel-fetch
    for( int k =0; k < 2; k++ )
        for( int j =0; j < 2; j++ )
            for( int i =0; i < 2; i++ ) {
                // The least significant 6 bits are used to
                // store (a part) of the in total 48 bits that
                // contain the offset of the current blockgroup
                uint32_t rgb24a1 =tex3D<uint32_t>( texture, 
                                                   i + subblock_index.x, 
                                                   j + subblock_index.y,
                                                   k + subblock_index.z );
                int bit_offs =6*(k*4+j*2+i); // column-major
                offset |= ((uint64_t)(rgb24a1 & 0x3f)) << bit_offs;
                // Count the blocks we need to skip, only if they're non-empty
                skip += ( !(rgb24a1 & 0x40) && n < block_idx ) ? 1 : 0;
                n++;
            }
    // We just calculated the offset of the block pointed to by the first voxel in
    // this subblock. Now skip another 'block_idx' blocks to reach the exact
    // block we're looking for

    offset +=skip;
    return offset;
}
   
VOXOWL_DEVICE
fragment_t
svmmRaycast( svmipmapDevice_t* v, const ray_t& r ) {
    double tmin, tmax;
    glm::ivec3 size =v->size;
    
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
    const glm::vec3 box_plane( 0, 1, 2 ); // X, Y and Z dividing planes
    glm::vec3 r0 = glm::abs( rayEntry / b.max );
    float largest =max( r0.x, max( r0.y, r0.z ) ); // Largest relative component
    glm::vec3 r1 = glm::floor( r0 / largest ); // Vector with a '1' at the largest component
    int side = glm::clamp( glm::dot( box_plane, r1 ), 0.f, 2.f );
   
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

    // ABOVE IS THE SAME AS voxelmapRaycast()
    //
    svmm_level_t *cur_level =v->level_ptr;
    int level =0;
    glm::vec4 vox;
    glm::ivec3 abs_index, block_index, block_offset, subblock_index, grid_offset, voxel_index;
    uint64_t block_num =0;
    uint32_t vox_raw;
    glm::ivec3 superblock_bounds;
    
    bool first =true;
    while(1) {

        if( index[side] < 0 || index[side] >= size[side] )
            break;
        
        vox = glm::vec4(0);
        cur_level =v->level_ptr;
        level =0;
        block_num =0;

        while(1) {
            // Absolute position in the mipmap level
            abs_index =index / cur_level->mipmap_factor;
            // Relative position in the block
            block_index =abs_index % cur_level->blockwidth;
            // The block is devided in subblocks: calculate the offset within the subblock
            block_offset = glm::ivec3( block_index.x % 2 ? 1 : 0,
                                       block_index.y % 2 ? 1 : 0,
                                       block_index.z % 2 ? 1 : 0 );
            // Calculate the index of the first block of the subblock
            subblock_index =block_index - block_offset;
            // Blocks are ordered linear on disk but as 3D texture on the GPU,
            // this is the 'block grid'. We need to calculate the block's
            // position in the grid.
            grid_offset =gridOffset( cur_level, block_num );

            voxel_index =grid_offset * cur_level->blockwidth + block_index;
            //printf( "(%d,%d,%d) ", voxel_index.x, voxel_index.y, voxel_index.z );
            // Determine the level's format
            switch( cur_level->format ) {
                case VOXEL_RGB24A1_UINT32:
                    vox_raw =tex3D<uint32_t>( cur_level->texture, voxel_index.x, voxel_index.y, voxel_index.z );
                    vox =unpackRGB24A1_UINT32( vox_raw );
                    break;
                case VOXEL_BITMAP_UINT8: {
                    glm::vec4 alpha =voxelTex3D( cur_level->texture, cur_level->format, voxel_index );
                    vox.a =alpha.a;
                    break;
                }
                default:
                    // All other formats only contain color information
                    vox =voxelTex3D_clamp( cur_level->texture, cur_level->format, voxel_index, cur_level->grid_size*cur_level->texels_per_blockwidth );
                    break;
            }

            if( level == v->levels-1 || isTerminal( vox_raw ) ) {
                superblock_bounds =abs_index + step;
                superblock_bounds *= cur_level->mipmap_factor;
                break;
            }

            // Go one level deeper
            
            block_num =getBlockOffsetTex3D( cur_level->texture, 
                                            grid_offset * cur_level->texels_per_blockwidth + subblock_index, 
                                            block_offset );
            level++;
            cur_level++;
        }

        frag.color =blendF2B( vox, frag.color );

        if( vox.a > 0.01f && first ) {
            // We calculate the position in unit-cube space..
            frag.position =frag.position_vs / (float)largest - b.max;
            // ..and the normal of the current 'face' of the voxel
            frag.normal[side] = -step[side];
            first =false;
        }

        if( frag.color.a < 0.1f ) {
            break;
        }

        float step_factor =glm::length( glm::vec3( cur_level->mipmap_factor) );
        float length =glm::length( glm::vec3( (float)cur_level->mipmap_factor * deltaDist ) );
        glm::vec3 l(0);
        while( !glm::any( glm::equal( index, superblock_bounds ) ) ) {
        // Branchless equivalent for
        //for( int i =0; i < 3; i++ ) 
        //    if( boxDist[side] > boxDist[i] )
        //        side =i;*/
        glm::bvec3 b0= glm::lessThan( boxDist, glm::vec3( boxDist.y, boxDist.z, boxDist.x ) );
        glm::bvec3 b1= glm::lessThanEqual( boxDist, glm::vec3( boxDist.z, boxDist.x, boxDist.y ) );
        glm::vec3 mask =glm::ivec3( b0.x && b1.x, b0.y && b1.y, b0.z && b1.z );
        side = glm::dot( box_plane, mask );

        boxDist[side] += deltaDist[side];
        l[side] += step[side];
        index[side] += step[side];
        frag.position_vs[side] += step[side];
        }
    }

    frag.color.a = 1.f - frag.color.a;

    return frag;
}
VOXOWL_HOST 
cudaError_t 
svmmCopyToDevice( svmipmapDevice_t* d_svmm, svmipmap_t* svmm ) {
    char *data_ptr =svmm->data_ptr;
    d_svmm->size =glm_ivec3_32( svmm->header.volume_size );
    
    // Create an array of level objects on device 
    
    printf( "svmmCopyToDevice: creating svmm on device: %d levels, volume size: (%d,%d,%d)\n",
            svmm->header.levels, svmm->header.volume_size.x, svmm->header.volume_size.y, svmm->header.volume_size.z );
    
    d_svmm->levels =svmm->header.levels;
    svmm_level_devptr_t level_ptr;

    RETURN_IF_ERR( cudaMalloc( &level_ptr, d_svmm->levels * sizeof( svmm_level_t ) ) );
    d_svmm->level_ptr =level_ptr;

    
    // Create the actual level objects on the host
   
    data_ptr +=svmm->header.root;
    svmm_level_header_t* lheader =(svmm_level_header_t*)data_ptr;
    
    svmm_level_t level[d_svmm->levels];
    svmm_level_t* cur_level =0;
        
    for( int i =0; i < d_svmm->levels; i++ ) {
        
        cur_level =&level[i];
        cur_level->format =lheader->format;
        cur_level->mipmap_factor =lheader->mipmap_factor;
        cur_level->block_count =lheader->block_count;

        if( i == 0 ) { // Root level
            cur_level->grid_size =glm_ivec3_32( svmm->header.rootsize );
            cur_level->grid_y_shift =cur_level->grid_z_shift =0;
            cur_level->texels_per_blockwidth =0;
            cur_level->blockwidth =glm::max( cur_level->grid_size.x, 
                                             cur_level->grid_size.y,
                                             cur_level->grid_size.z );
        }
        else {
            cur_level->grid_size =calcGridSize( cur_level->block_count );
            cur_level->grid_y_shift =cur_level->grid_z_shift =log2_p2( cur_level->grid_size.y );
            cur_level->grid_z_shift +=cur_level->grid_y_shift;
            cur_level->blockwidth =lheader->blockwidth;
            cur_level->texels_per_blockwidth =
                blockCount( cur_level->format, ivec3_32( cur_level->blockwidth ) ).x;
        }

        printf( "Allocating mipmap level %d: %d blocks, grid size: %dx(%d,%d,%d) shift: %d format: %s mipmap_factor: %d texels_per_blockwidth: %d\n",
                i, cur_level->block_count, cur_level->blockwidth, 
                cur_level->grid_size.x, cur_level->grid_size.y, cur_level->grid_size.z,
                cur_level->grid_y_shift, strVoxelFormat( cur_level->format ), cur_level->mipmap_factor,
                cur_level->texels_per_blockwidth );

        // Create a texture for the calculated grid size and copy the data
        
        glm::ivec3 grid_texels =glm_ivec3_32( blockCount( cur_level->format, 
                                         ivec3_32( cur_level->grid_size * (int)lheader->blockwidth ) ) );

        int bits_per_block =bytesPerBlock( lheader->format ) * 8;
        cudaExtent v_extent = make_cudaExtent( grid_texels.x, 
                                               grid_texels.y, 
                                               grid_texels.z );
        cudaChannelFormatDesc v_channelDesc = cudaCreateChannelDesc( bits_per_block, 
                                                                     0, 0, 0,
                                                                     cudaChannelFormatKindUnsigned);

        RETURN_IF_ERR( cudaMalloc3DArray( &cur_level->data_ptr, &v_channelDesc, v_extent ) );

        // Prepare the parameters to the texture binding
        //
        cudaResourceDesc resDesc;
        memset(&resDesc, 0, sizeof(resDesc));
        resDesc.resType = cudaResourceTypeArray;
        resDesc.res.array.array = (cudaArray_t)cur_level->data_ptr;

        cudaTextureDesc texDesc;
        memset(&texDesc, 0, sizeof(texDesc));
        texDesc.readMode = cudaReadModeElementType;
        texDesc.filterMode = cudaFilterModePoint;

        // Bind the texture object
        
        cur_level->texture =0;
        RETURN_IF_ERR( cudaCreateTextureObject( &cur_level->texture, &resDesc, &texDesc, NULL ) );

        // Copy the block data

        int bytes_per_block =voxelmapSize( cur_level->format, ivec3_32( cur_level->blockwidth ) );
        int bytes_per_row;
        cudaExtent extent;
        
        if( i == 0 ) { // Root level
            extent =v_extent;
            bytes_per_row =v_extent.width * bytesPerBlock( cur_level->format );
        } else {
            extent =make_cudaExtent( cur_level->texels_per_blockwidth, 
                                     cur_level->texels_per_blockwidth, 
                                     cur_level->texels_per_blockwidth );
            bytes_per_row =cur_level->texels_per_blockwidth * bytesPerBlock( cur_level->format );
        }
        
        for( int block =0; block < cur_level->block_count; block++ ) {
            glm::ivec3 grid_pos = gridOffset( cur_level, block ) * cur_level->texels_per_blockwidth;
/*            printf( "Writing block %d to grid position (%d,%d,%d), grid size (%d,%d,%d)\n", 
                    block, grid_pos.x, grid_pos.y, grid_pos.z,
                    grid_texels.x, grid_texels.y, grid_texels.z );*/
            cudaMemcpy3DParms copyParams = {0};
            copyParams.srcPtr   = make_cudaPitchedPtr( data_ptr+sizeof( svmm_level_header_t)+block*bytes_per_block, 
                                                       bytes_per_row, 
                                                       extent.width, 
                                                       extent.height );
            copyParams.dstArray = cur_level->data_ptr;
            copyParams.dstPos.x   = grid_pos.x;
            copyParams.dstPos.y   = grid_pos.y;
            copyParams.dstPos.z   = grid_pos.z;
            copyParams.extent   = extent;
            copyParams.kind     = cudaMemcpyHostToDevice;
            RETURN_IF_ERR( cudaMemcpy3D(&copyParams) );
        }   
        // Prepare for the next level
        data_ptr -= lheader->next; // Jump to the next level is negative
        lheader =(svmm_level_header_t*)data_ptr;
    }

    // Copy the level objects to the device 

    RETURN_IF_ERR( cudaMemcpy( d_svmm->level_ptr, 
                               level, 
                               d_svmm->levels*sizeof( svmm_level_t ), 
                               cudaMemcpyHostToDevice ) );

    return (cudaError_t)0;
}

#include "svmipmap.h"
#include <stdio.h>

#define EQUALITY_MAX_DELTA 0.001f
#define DATA_RESIZE 1024

VOXOWL_HOST
uint32_t
encodeBlock( voxelmap_t *dst, voxelmap_t *src, ivec3_16_t offset, ivec3_16_t size, bool *homogenous ) {
    int count =0;
    glm::dvec4 sum;
    glm::vec4 last;
    if( homogenous ) *homogenous =false;

    for( int z =offset.z; z < size.z + offset.z && z < src->size.z; z++ )
        for( int y =offset.y; y < size.y + offset.y && y < src->size.y; y++ )
            for( int x =offset.x; x < size.x + offset.x && x < src->size.x; x++ ) {
                 glm::vec4 rgba =voxelmapUnpack( src, ivec3_16( x, y, z ) );
                 if( homogenous && *homogenous && count )
                    *homogenous =*homogenous && glm::all( glm::lessThanEqual ( glm::abs( rgba - last ), glm::vec4( EQUALITY_MAX_DELTA ) ) );
                count++;
                sum +=glm::dvec4( rgba ); 
                last =rgba;
                //voxelmapPack( dst, ivec3_16( x - offset.x, y - offset.y, z - offset.z ), rgba );
                *(uint32_t*)voxel( dst, ivec3_16( x - offset.x, y - offset.y, z - offset.z ) )
                    = *(uint32_t*)voxel( src, ivec3_16( x, y, z ) );

            }
    
    // Calculate the average
    glm::vec4 avg =glm::vec4( sum / (double)count );
    // Currently, only 1-bit alpha is supported because we need the other 
    // 7 bits for bookkeeping
    uint32_t rgba;
    packRGB24A1_UINT32( &rgba, avg );
    // Clear the lower 7 bits [0-5]
    //rgba &= ~0x7f;

    // The 6th bit is used to set the terminal condition (leaf node)
    rgba &= ~0x40;
    rgba |= ( *homogenous ? 0x40 : 0x0 );
    return rgba;
}

VOXOWL_HOST
int 
ensureSize( char** ptr, size_t current_size, size_t new_size ) {
    if( current_size < new_size ) {
        *ptr =(char*)realloc( *ptr, new_size + DATA_RESIZE );
        return new_size + DATA_RESIZE;
    }
    return current_size;
}

VOXOWL_HOST
ivec3_16_t
nextMipmapSize( ivec3_16_t size, ivec3_16_t block_size ) {
    ivec3_16_t block_count =ivec3_16( size.x / block_size.x, 
                                      size.y / block_size.y, size.z / block_size.z );

    if( size.x % block_size.x )
        block_count.x++;
    if( size.y % block_size.y )
        block_count.y++;
    if( size.z % block_size.z )
        block_count.z++;
    return block_count;
}

VOXOWL_HOST
voxelmap_t*
encodeLevel( char **data_ptr, size_t *data_size, size_t* data_offset, voxelmap_t *v, bool base_level, unsigned int blockwidth ) {
    // Calculate the block size and number of (super)blocks that are contained within
    // the given volume
    ivec3_16_t block_size =ivec3_16( blockwidth );
    ivec3_16_t block_count =nextMipmapSize( v->size, block_size );

    //ivec3_16_t superblock_size =ivec3_16( blockwidth * 2 );
    ivec3_16_t superblock_count =ivec3_16( block_count.x /2, block_count.y /2, block_count.z /2);

    if( block_count.x % 2 )
        superblock_count.x++;
    if( block_count.y % 2 )
        superblock_count.y++;
    if( block_count.z % 2 )
        superblock_count.z++;

//    ivec3_16_t block_count =ivec3_16( superblock_count.x *2,
 //       superblock_count.y *2, superblock_count.z *2 );

    // Allocate a new voxelmap for the parent level. This is a temporary array
    // that will contain the averaged color information one resolution step
    // lower (i.e. one mipmap level higher)
    //fprintf( stderr, "Temporary voxelmap resolution %d x %d x %d\n", block_count.x, block_count.y, block_count.z );
    //fprintf( stderr, "Superblock count %d x %d x %d\n", superblock_count.x, superblock_count.y, superblock_count.z );
    voxelmap_t *parent =(voxelmap_t*)malloc( sizeof( voxelmap_t ) );
    voxelmapCreate( parent, VOXEL_RGB24A1_UINT32, ivec3_16( superblock_count.x * 2 ) );
    // FIXME: We abuse the size attribute here to pass down the actual mipmap size
    parent->size =block_count;

    // This structure will be used to populate the blocks of the current level
    voxelmap_t block;
    block.format =VOXEL_RGB24A1_UINT32;
    block.size =block_size;
    block.blocks =block_size;
    block.data =0;
    // The size of one block in the current level
    size_t bytes_per_block =voxelmapSize( &block ); 

    // Offset (number of blocks) within the level
    uint64_t level_offset =0;

    for( int z=0; z < superblock_count.z; z++ )
        for( int y=0; y < superblock_count.y; y++ )
            for( int x=0; x < superblock_count.x; x++ ) {
                uint64_t superblock_level_offset =level_offset;
                
                for( int k=0; k < 2; k++ )
                    for( int j=0; j < 2; j++ )
                        for( int i=0; i < 2; i++ ) { 
                            // Prepare to write a new block into the data pointer
                            *data_size =ensureSize( data_ptr, 
                                                    *data_size, 
                                                    *data_offset + bytes_per_block );
                            block.data =(char*)*data_ptr + *data_offset;
                            memset( block.data, 0xff, bytes_per_block ); // DEBUG
                            
                            bool homogenous;
                            // Calculate the offset into the source voxelmap
                            ivec3_16_t offset =ivec3_16( (2 * x + i) * blockwidth, 
                                                         (2 * y + j) * blockwidth, 
                                                         (2 * z + k) * blockwidth);
                            // encodeBlock() will calculate the average and copy the
                            // block data into 'block'
                            // 'avg' now contains packed RGB as well as a 1-bit alpha
                            // and a terminal bit
                            uint32_t avg =encodeBlock( &block, 
                                                       v, 
                                                       offset, 
                                                       block_size, 
                                                       &homogenous );

                            // The least significant 6 bits will be used to
                            // store (a part) of the in total 48 bits that
                            // contain the offset of the current superblock
                            int bit_offs =6*(k*4+j*2+i); // column-major
                            uint64_t bits =superblock_level_offset & (0x3f << bit_offs);
                            avg &= ~0x3f;
                            avg |= (bits >> bit_offs);

                            // Write the average to the parent level temporary
                            *(uint32_t*)voxel( parent, ivec3_16( 2*x+i, 2*y+j, 2*z+k ) ) =avg;

                            // If the calculated block is NOT homogenous, we actually need
                            // to store it. A homogenous block contains only the same values
                            if(/* !homogenous*/true ) {
                                // By incrementing the data offset, we don't overwrite the
                                // block
                                *data_offset +=bytes_per_block;
                                // We also increment the block_offset, i.e.: the amount
                                // actual 'useful' blocks we've written so far
                                level_offset++;
                            }
                }
            }
   return parent; 
}

VOXOWL_HOST 
int 
svmmEncode( svmipmap_t* m,  voxelmap_t* uncompressed, svmm_encode_opts_t opts )
{
    size_t data_size = sizeof( svmm_header_t ) + DATA_RESIZE; 
    char *data_ptr =(char*)malloc( data_size );
    size_t data_offset =sizeof( svmm_header_t );
    size_t noffset =0; // The negative offset that points 'back' to the next level
   
    int levels =1;
    size_t rootoffset =0;
    unsigned int blockwidth =opts.blockwidth;

    voxelmap_t *v =uncompressed;
    
    fprintf( stderr, "Compressing volume %d x %d x %d, blockwidth: %d, rootwidth: %d\n", 
    v->size.x, v->size.y, v->size.z, opts.blockwidth, opts.rootwidth );

    // Write the mipmap levels starting with the highest resolution (level 0)
    while( v->size.x > opts.rootwidth && 
           v->size.y > opts.rootwidth &&
           v->size.z > opts.rootwidth ) {

        // Remember where this level begins
        size_t level_begin =data_offset;

        // If we're not the base level, we add an header to the beginning
        //if( v != uncompressed ) {
            data_offset += sizeof( svmm_level_header_t );
            data_size =ensureSize( &data_ptr, data_size, data_offset );
        //} 
        //else fprintf( stderr, "base" );

        levels++;
        
        // Encode the current mipmap level into blocks
        voxelmap_t *parent =encodeLevel( &data_ptr, 
                                         &data_size, 
                                         &data_offset,
                                         v, 
                                         v == uncompressed, 
                                         blockwidth
                                         );


        // If we're not the base level, cleanup and set the next point to the
        // correct offset (negative/backward)
        svmm_level_header_t *lheader =(svmm_level_header_t*)(data_ptr + level_begin);
        memset( lheader, 0, sizeof( svmm_level_header_t ) );
        lheader->next =noffset;
        lheader->mipmap_size =v->size;
        
        if( v != uncompressed ) {
            //voxelmapFree( v );
            free( v );
        }
        noffset =data_offset - level_begin;
        //fprintf( stderr, "noffset: %d, data offset: %d\n", noffset, level_begin );

        fprintf( stderr, "level #%d: datasize: %d next mipmap level size: %d x %d x %d\n",
            levels, noffset, parent->size.x, parent->size.y, parent->size.z );

        v =parent;
    }

    // Write the top/root level
    // 'v' should now contain the root-level voxelmap
    svmm_level_header_t *rootheader =(svmm_level_header_t*)(data_ptr + data_offset);
    rootoffset =data_offset;
    data_offset += sizeof( svmm_level_header_t );
    
    voxelmap_t rootmap;
    rootmap.format =VOXEL_RGB24A1_UINT32;
    rootmap.size =v->size;
    rootmap.blocks =blockCount( rootmap.format, v->blocks ); // FIXME: 'blocks' is incorrect
    rootmap.data =data_ptr + data_offset;

    data_size =ensureSize( &data_ptr, data_size, data_offset + voxelmapSize( &rootmap ) );
    // Set the offset to the next level
    memset( rootheader, 0, sizeof( svmm_level_header_t ) );
    rootheader->next =noffset; // may be zero if only one level is present
    rootheader->mipmap_size =v->size;
    // Copy the data
    voxelmapSafeCopy( &rootmap, v );
    data_offset += voxelmapSize( &rootmap );

    if( v != uncompressed ) {
//        voxelmapFree( v );
        free( v );
    }

    fprintf( stderr, "rootlevel: %d x %d x%d\n", rootmap.size.x, rootmap.size.y, rootmap.size.z );


    // Resize the buffer to fit exactly
    data_ptr =(char*)realloc( data_ptr, data_offset );
    fprintf( stderr, "Total size: %dK, original size: %dK\n", data_offset/1024, voxelmapSize( uncompressed ) / 1024 );

    // Write the header on top of the buffer
    svmm_header_t *header =(svmm_header_t*)data_ptr;
    header->magic1 =SVMM_MAGIC1;
    header->magic2 =SVMM_MAGIC2;
    header->volume_size =uncompressed->size;
    header->format =VOXEL_RGB24A1_UINT32; // unused for now
    header->data_start =sizeof( svmm_header_t );
    header->data_length =data_offset -  sizeof( svmm_header_t);
    header->levels =levels;
    header->blockwidth =opts.blockwidth;
    header->rootsize =rootmap.blocks;
    header->root =rootoffset;

    m->buffer =data_ptr;
    memcpy( &m->header, header, sizeof( svmm_header_t ) );
    
    return 0;    
}

VOXOWL_HOST 
int 
svmmEncode( svmipmap_t* m,  voxelmap_t* uncompressed ) {
    svmm_encode_opts_t opts;
    opts.blockwidth =4;//SVMM_DEFAULT_BLOCKWIDTH;
    opts.rootwidth =32;
    opts.format =VOXEL_RGBA_UINT32;
    return svmmEncode( m, uncompressed, opts );
}

VOXOWL_HOST 
void 
svmmFree( svmipmap_t* m ) {
    if( m->buffer )
        free( m->buffer );
}

VOXOWL_HOST 
void 
svmmReadHeader( svmm_header_t* header, void *buffer ) {
    memcpy( header, buffer, sizeof( svmm_header_t ) );
}

VOXOWL_HOST 
void 
svmmRead( svmipmap_t* m, void *buffer ) {
    svmmReadHeader( &m->header, buffer );
    m->buffer =buffer;
}

VOXOWL_HOST
void
copySubblock( voxelmap_t *subblock, voxelmap_t *block, ivec3_16_t subblock_offset ) {
    for( int k =0; k < 2; k++ )
        for( int j =0; j < 2; j++ )
            for( int i =0; i < 2; i++ ) {
                ivec3_16_t subpos =ivec3_16( i, j, k );
                ivec3_16_t pos =ivec3_16( subblock_offset.x + i,
                                          subblock_offset.y + j,
                                          subblock_offset.z + k );
                *(uint32_t*)voxel( subblock, subpos ) = *(uint32_t*)voxel( block, pos );
            }
}

VOXOWL_HOST
uint64_t
getBlockOffset( voxelmap_t *subblock, ivec3_16_t block_offset, size_t blocksize ) {
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
                // contain the offset of the current superblock
                uint32_t rgb24a1 =*(uint32_t*)voxel( subblock, ivec3_16( i, j, k ) );
                int bit_offs =6*(k*4+j*2+i); // column-major
                offset |= (uint64_t)(rgb24a1 & 0x3f) << bit_offs;
                // Count the blocks we need to skip, only if they're non-empty
                skip += ( !(rgb24a1 & 0x40) && n < block_idx ) ? 1 : 0;
                n++;
            }
    // We just calculated the offset of the block pointed to by the first voxel in
    // this subblock. Now skip another 'block_idx' blocks to reach the exact
    // block we're looking for
    offset +=skip;
    offset *=blocksize;
    return offset;
}

VOXOWL_HOST 
glm::vec4 
svmmDecodeVoxel( svmipmap_t* m, ivec3_16_t position ) {
    char *data_ptr =(char*)m->buffer;
    svmm_header_t *header =&m->header;
    int level =0;

    data_ptr += header->root;
//    size_t debug_offset =header->root;
    ivec3_16_t block_size =header->rootsize;
    ivec3_16_t mipmap_size =header->rootsize;

    voxelmap_t subblock;
    voxelmapCreate( &subblock, VOXEL_RGB24A1_UINT32, ivec3_16( 2, 2, 2 ) ); 

    voxelmap_t block;
    block.size =header->rootsize;
    block.format =VOXEL_RGB24A1_UINT32;
    block.blocks =header->rootsize;
    block.data =data_ptr + sizeof( svmm_level_header_t );

    uint32_t vxl =0;

    while( 1 ) {
        svmm_level_header_t *lheader =(svmm_level_header_t*)data_ptr;
        mipmap_size =lheader->mipmap_size;
        
        level++;
        int nlevel =header->levels -level;

        // First, calculate the absolute index of the voxel on the current mipmap level
        /*ivec3_16_t abs_index =ivec3_16( (double)position.x / (double)header->volume_size.x * mipmap_size.x,
                                        (double)position.y / (double)header->volume_size.y * mipmap_size.y,
                                        (double)position.z / (double)header->volume_size.z * mipmap_size.z );*/
        ivec3_16_t abs_index =ivec3_16( position.x / pow( header->blockwidth, nlevel ),
                                        position.y / pow( header->blockwidth, nlevel ),
                                        position.z / pow( header->blockwidth, nlevel ) );
        //if( level == header->levels ) {
        //    mipmap_size =header->volume_size;
        //   abs_index =position;
        //}

        // Translate into a relative index into the corresponding block
        ivec3_16_t block_index =ivec3_16( abs_index.x % block_size.x,
                                          abs_index.y % block_size.y,
                                          abs_index.z % block_size.z );
        // The block is devided in subblocks: calculate the offset within the subblock
        ivec3_16_t block_offset = ivec3_16( block_index.x % 2 ? 1 : 0,
                                            block_index.y % 2 ? 1 : 0,
                                            block_index.z % 2 ? 1 : 0 );
        // Finally, calculate the index of the first block of the subblock
        ivec3_16_t subblock_index =block_index;
        subblock_index.x -=block_offset.x;
        subblock_index.y -=block_offset.y;
        subblock_index.z -=block_offset.z;

        // Copy this subblock to a local voxelmap and retrieve the value we want
        copySubblock( &subblock, &block, subblock_index );
       // vxl =*(uint32_t*)voxel( &subblock, block_offset );
       vxl =*(uint32_t*)voxel( &block, block_index );

        // If we're either at level 0 (highest mipmap resolution) or the
        // terminal bit is set, we break the loop
        if( level == header->levels || (vxl & 0x40) )
            break;

        // We continue, fetch the level header to calucate the negative offset
        data_ptr -=lheader->next;
        //debug_offset -=lheader->next;
        //blocksize =lheader->mipmap_size;

        // Calculate the offset within the next level using the subblock
        block_size =ivec3_16( header->blockwidth );
        block.size =block.blocks =block_size;
        uint64_t offset =getBlockOffset( &subblock, block_offset, voxelmapSize( &block ) );

        // Prepare to read this block
        block.data =data_ptr + offset + sizeof( svmm_level_header_t );
        //fprintf( stderr, "negative offset: %d, new data offset %d\n", lheader->next, debug_offset );
        // The next mipmap level will have blockwidth times more voxels along
        // each direction
        mipmap_size.x *=header->blockwidth;
        mipmap_size.y *=header->blockwidth;
        mipmap_size.z *=header->blockwidth;
    }

    voxelmapFree( &subblock );

    glm::vec4 color =unpackRGB24A1_UINT32( vxl );
    return color;
}

VOXOWL_HOST 
bool 
svmmTest( voxelmap_t* uncompressed ) {
    svmipmap_t mipmap;
    svmmEncode( &mipmap, uncompressed );

    int errors =0;

    for( int x=0; x < uncompressed->size.x; x++ )
        for( int y=0; y < uncompressed->size.y; y++ )
            for( int z=0; z < uncompressed->size.z; z++ ) {
                glm::vec4 orig =voxelmapUnpack( uncompressed, ivec3_16( x, y, z ) );
                glm::vec4 enc =svmmDecodeVoxel( &mipmap, ivec3_16( x, y, z ) );
                bool test =glm::all( glm::lessThanEqual ( glm::abs( orig - enc ), glm::vec4( EQUALITY_MAX_DELTA ) ) );
                if( !test ) {
                    if( !errors++ )
                        fprintf( stderr, "Unexpected value. original: (%f %f %f %f), encoded: (%f %f %f, %f), delta: %f, at (%d %d %d)\n",
                        orig.r, orig.g, orig.b, orig.a, enc.r, enc.g, enc.b, enc.a, EQUALITY_MAX_DELTA, x, y, z );
                }
                //DEBUG
                voxelmapPack( uncompressed, ivec3_16( x, y, z ), enc );
            }
    fprintf( stderr, "Compressed size: %d, uncompressed size: %d, ratio: %d%\n",
        mipmap.header.data_length, voxelmapSize( uncompressed ), 
        (int)(((float)mipmap.header.data_length / (float)voxelmapSize( uncompressed ))*100.f) );
    fprintf( stderr, "Test completed. errors: %d\n", errors );
    return errors == 0;
}

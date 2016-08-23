#include "../include/svmipmap.h"
#include <stdio.h>
#include <assert.h>
#include <string.h>

// Unix low-level IO functions
#include <unistd.h>
#include <sys/mman.h>
#include <sys/types.h>
#include <fcntl.h>

#define DATA_RESIZE 8192

VOXOWL_HOST size_t ensureSize( svmipmap_t* m, size_t new_size );
VOXOWL_HOST ivec3_32_t nextMipmapSize( ivec3_32_t size, ivec3_32_t block_size );

VOXOWL_HOST
glm::vec4
encodeBlock( voxelmap_t *dst, 
             voxelmap_t *src, 
             ivec3_32_t offset, 
             ivec3_32_t size, 
             bool *homogeneous, 
             bool base_level, 
             svmm_encode_opts_t *opts ) {
    int count =0;
    glm::vec4 list[size.x*size.y*size.z];
    glm::dvec4 sum;
    if( homogeneous ) *homogeneous =true;
    bool convert =src->format != dst->format;


    for( int z =offset.z; z < size.z + offset.z && z < src->size.z; z++ )
        for( int y =offset.y; y < size.y + offset.y && y < src->size.y; y++ )
            for( int x =offset.x; x < size.x + offset.x && x < src->size.x; x++ ) {
                glm::vec4 rgba =voxelmapUnpack( src, ivec3_32( x, y, z ) );
                list[count++] =rgba;
                sum +=glm::dvec4( rgba );

                /*if( bitsPerVoxel( dst->format ) == 1 )
                    voxelmapPack( dst, ivec3_32( x - offset.x, y - offset.y, z - offset.z ), glm::vec4( rgba.a >= .5f ) );
                else*/ if( convert )
                    voxelmapPack( dst, ivec3_32( x - offset.x, y - offset.y, z - offset.z ), rgba );
                else {
                    uint8_t lower_bits;
                    switch( bitsPerVoxel( dst->format ) ) {
                        case 8: {
                            lower_bits = *(uint8_t*)voxel( src, ivec3_32( x, y, z ) );
                            *(uint32_t*)voxel( dst, ivec3_32( x - offset.x, y - offset.y, z - offset.z ) ) = lower_bits;
                            break;
                        }
                        case 16: {
                            uint16_t u16 = *(uint16_t*)voxel( src, ivec3_32( x, y, z ) );
                            *(uint16_t*)voxel( dst, ivec3_32( x - offset.x, y - offset.y, z - offset.z ) ) = u16;
                            lower_bits =u16 & 0xff;
                            break;
                        }
                        case 32: {
                            uint32_t u32 = *(uint32_t*)voxel( src, ivec3_32( x, y, z ) );
                            *(uint32_t*)voxel( dst, ivec3_32( x - offset.x, y - offset.y, z - offset.z ) ) = u32;
                            lower_bits =u32 & 0xff;
                            break;
                        }
                    }

                    // A block can only be homogeneous if all its voxels are
                    // terminal (except for the baselevel)
                    if( !base_level )
                        *homogeneous = *homogeneous && isTerminal( lower_bits );
                }

            }
    
    // Calculate the average
    glm::vec4 avg =glm::vec4( sum / (double)count );
   
    for( int i =0; i < count; i++ ) {
         // A block is homogeneous if all its voxels differ no more
         // than delta from the average
         if( homogeneous && *homogeneous )
            *homogeneous =*homogeneous && 
                glm::all( glm::lessThanEqual ( glm::abs( list[i] - avg), glm::vec4( opts->delta ) ) );

    }/*for( int z =offset.z; z < size.z + offset.z && z < src->size.z; z++ )
        for( int y =offset.y; y < size.y + offset.y && y < src->size.y; y++ )
            for( int x =offset.x; x < size.x + offset.x && x < src->size.x; x++ ) {
                 glm::vec4 rgba =voxelmapUnpack( src, ivec3_32( x, y, z ) );
                 // A block is homogeneous if all its voxels differ no more
                 // than delta from the average
                 if( homogeneous && *homogeneous && count )
                    *homogeneous =*homogeneous && glm::all( glm::lessThanEqual ( glm::abs( rgba - avg), glm::vec4( opts->delta ) ) );
            }*/
/*    
    // Currently, only 1-bit alpha is supported because we need the other 
    // 7 bits for bookkeeping
    uint32_t rgba;
    packRGB24A1_UINT32( &rgba, avg );

    // The 6th bit is used to set the terminal condition (leaf node)
    setTerminal( &rgba, *homogeneous );*/
    return avg;
}


VOXOWL_HOST
voxelmap_t*
encodeLevel( svmipmap_t * m,
             size_t* data_offset, 
             voxelmap_t *v, 
             bool base_level, 
             unsigned int blockwidth,
             svmm_encode_opts_t opts,
             int *blocks_written,
             svmm_level_header_t *lheader ) {
    // Calculate the block size and number of (super)blocks that are contained within
    // the given volume
    ivec3_32_t block_size =ivec3_32( blockwidth );
    ivec3_32_t block_count =nextMipmapSize( v->size, block_size );

    //ivec3_32_t blockgroup_size =ivec3_32( blockwidth * 2 );
    ivec3_32_t blockgroup_count =ivec3_32( block_count.x /2, block_count.y /2, block_count.z /2);

    // Allocate a new voxelmap for the parent level. This is a temporary array
    // that will contain the averaged color information one resolution step
    // lower (i.e. one mipmap level higher)
    voxelmap_t *parent =(voxelmap_t*)malloc( sizeof( voxelmap_t ) );
    voxelmapCreate( parent, opts.format, block_count );

    // This structure will be used to populate the blocks of the current level
    voxelmap_t block;
    if( base_level && opts.bitmapBaselevel ) {
        if( blockwidth == 2)
                block.format =VOXEL_BITMAP_UINT8;
        else {
            switch( opts.format ) {
                case VOXEL_RGB24A1_UINT32:
                case VOXEL_RGB24A3_UINT32:
                    block.format =VOXEL_RGB24_8ALPHA1_UINT32;
                    break;
                case VOXEL_DENSITY8_UINT16:
                    block.format =VOXEL_DENSITY8_8ALPHA1_UINT16;
                    break;
                case VOXEL_INTENSITY8_UINT16:
                    block.format =VOXEL_INTENSITY8_8ALPHA1_UINT16;
                    break;
            }
        }
    }
    else
        block.format =opts.format;
    lheader->format =block.format;
    block.size =block_size;
    block.blocks =blockCount( block.format, block_size );
    block.data =0;
    // The size of one block in the current level
    size_t bytes_per_block =voxelmapSize( &block ); 

    // Offset (number of blocks) within the level
    uint32_t level_offset =0;

    for( int z=0; z < blockgroup_count.z; z++ ) {
        printf( "Writing Z-Blockgroup %d\n", z );
        for( int y=0; y < blockgroup_count.y; y++ ) {
            for( int x=0; x < blockgroup_count.x; x++ ) {
                uint32_t blockgroup_level_offset =level_offset;
                
                for( int k=0; k < 2; k++ )
                    for( int j=0; j < 2; j++ )
                        for( int i=0; i < 2; i++ ) { 
                            // Prepare to write a new block into the data pointer
                            ensureSize( m, *data_offset + bytes_per_block );
                            block.data =m->data_ptr + *data_offset;

                            if( base_level && opts.bitmapBaselevel )
                                memset( block.data, 0x0, bytes_per_block );
                            
                            bool homogeneous;
                            // Calculate the offset into the source voxelmap
                            ivec3_32_t offset =ivec3_32( (2 * x + i), 
                                                         (2 * y + j), 
                                                         (2 * z + k));
                            // encodeBlock() will calculate the average and copy the
                            // block data into 'block'
                            // 'avg' now contains packed RGB as well as a 1-bit alpha
                            // and a terminal bit
                            glm::vec4 avg =encodeBlock( &block, 
                                                       v, 
                                                       ivec3_mults32( offset, blockwidth ), 
                                                       block_size, 
                                                       &homogeneous,
                                                       base_level, 
                                                       &opts );

                            //memset( block.data, 0x0f, bytes_per_block ); // DEBUG
                            // The least significant 6 bits will be used to
                            // store (a part) of the in total 32 bits that
                            // contain the offset of the current blockgroup
                            int bit_offs =SVMM_OFFSET_BITS * (k*4+j*2+i); // column-major
                            uint32_t bits =blockgroup_level_offset & (SVMM_OFFSET_BITS_MASK << bit_offs);
                            
                            
                            // We way we store the average + offset bits, depends on the level's format
                            
                            voxelmapPack( parent, offset, avg );

/*                            switch( opts.format ) {
                                case VOXEL_RGB24A1_UINT32:
                                    break;
                                case VOXEL_RGB24A3_UINT32:
                                    block.format =VOXEL_RGB24_8ALPHA1_UINT32;
                                    break;
                                case VOXEL_DENSITY8_UINT16:
                                    block.format =VOXEL_DENSITY8_8ALPHA1_UINT16;
                                    break;
                                case VOXEL_INTENSITY8_UINT16:
                                    block.format =VOXEL_INTENSITY8_8ALPHA1_UINT16;
                                    break;
                            }*/

                            uint8_t *lower_bits =(uint8_t*)voxel( parent, offset );

                            (*lower_bits) &= ~SVMM_OFFSET_BITS_MASK; 
                            (*lower_bits) |= (uint8_t)(bits >> bit_offs);
                            setTerminal( lower_bits, homogeneous );

                            // Write the average to the parent level temporary

                            // If the calculated block is NOT homogeneous, we actually need
                            // to store it. A homogeneous block contains only the same values
                            if( !homogeneous ) {
                                // By incrementing the data offset, we don't overwrite the
                                // block
                                *data_offset +=bytes_per_block;
                                // We also increment the level_offset, i.e.: the amount
                                // actual 'useful' blocks we've written so far
                                level_offset++;
                                (*blocks_written)++;
                            }
                }
            }
        }
    }
   return parent; 
}

VOXOWL_HOST 
ssize_t 
svmmEncode( svmipmap_t* m,  voxelmap_t* uncompressed, svmm_encode_opts_t opts )
{
    // Only accepts formats that reserve the lower 8 bits for structural information
    switch( opts.format ) {
        case VOXEL_RGB24A1_UINT32:
        case VOXEL_RGB24A3_UINT32:
        case VOXEL_INTENSITY8_UINT16:
        case VOXEL_DENSITY8_UINT16:
            break;
        default:
            return -1;
    }

    if( !m->is_mmapped ) {
        m->data_size = sizeof( svmm_header_t ) + DATA_RESIZE; 
        m->data_ptr =(char*)malloc( m->data_size );
    }

    size_t data_offset =sizeof( svmm_header_t );
    size_t noffset =0; // The negative offset that points 'back' to the next level
   
    // Number of levels, there's always at least one
    int levels =1;
    // Positive offset into the highest mipmap level (the root)
    size_t rootoffset =0;
    // Factor by which the current mipmap level is divided
    // Level = 0 is fullsize, so mipmap_factor = 1
    size_t mipmap_factor =1;
    unsigned int blockwidth =opts.blockwidth;

    voxelmap_t *v =uncompressed;
    
    fprintf( stdout, "Compressing volume %d x %d x %d, blockwidth: %d, rootwidth: %d, delta: %f\n", 
    v->size.x, v->size.y, v->size.z, opts.blockwidth, opts.rootwidth, opts.delta );

    // Write the mipmap levels starting with the highest resolution (level 0)
    while( v->size.x > opts.rootwidth && 
           v->size.y > opts.rootwidth &&
           v->size.z > opts.rootwidth ) {

        // Remember where this level begins
        size_t level_begin =data_offset;

        // Prepare the level header
        data_offset += sizeof( svmm_level_header_t );
        ensureSize( m, data_offset );
        
        svmm_level_header_t *lheader =(svmm_level_header_t*)(m->data_ptr + level_begin);
        memset( lheader, 0, sizeof( svmm_level_header_t ) );

        int block_count =0;
        // Encode the current mipmap level into blocks
        voxelmap_t *parent =encodeLevel( m, 
                                         &data_offset,
                                         v, 
                                         v == uncompressed, 
                                         blockwidth,
                                         opts,
                                         &block_count,
                                         lheader
                                         );


        // If we're not the base level, cleanup and set the next point to the
        // correct offset (negative/backward)
        lheader =(svmm_level_header_t*)(m->data_ptr + level_begin); // Reset the pointer, its address may have changed
        lheader->next =noffset;
        lheader->blockwidth =blockwidth;
        lheader->mipmap_factor =mipmap_factor;
        lheader->block_count =block_count;
        
        if( v != uncompressed ) {
            //voxelmapFree( v );
            free( v );
        }
        noffset =data_offset - level_begin;
        //fprintf( stderr, "noffset: %d, data offset: %d\n", noffset, level_begin );

        fprintf( stdout, "level #%d: datasize: %zu format: %d non-homogeneous blocks %d (%d%) parent mipmap level size: %d x %d x %d\n",
            levels-1, noffset, lheader->format, 
            block_count, (int)((float)block_count / (float)(parent->size.x*parent->size.y*parent->size.z) * 100),
            parent->size.x, parent->size.y, parent->size.z );

        levels++;
        mipmap_factor *= blockwidth;
        v =parent;
    }

    // Write the top/root level
    // 'v' should now contain the root-level voxelmap
    svmm_level_header_t *rootheader =(svmm_level_header_t*)(m->data_ptr + data_offset);
    rootoffset =data_offset;
    data_offset += sizeof( svmm_level_header_t );
    
    voxelmap_t rootmap;
    rootmap.format =opts.format;
    rootmap.size =v->size;
    rootmap.blocks =blockCount( rootmap.format, v->size );
    rootmap.data =m->data_ptr + data_offset;

    ensureSize( m, data_offset + voxelmapSize( &rootmap ) );
    // Set the offset to the next level
    memset( rootheader, 0, sizeof( svmm_level_header_t ) );
    rootheader->next =noffset; // may be zero if only one level is present
    rootheader->format =rootmap.format;
    rootheader->mipmap_factor =mipmap_factor;
    rootheader->blockwidth =1;//opts.rootwidth;
    rootheader->block_count =1;
    // Copy the data
    voxelmapSafeCopy( &rootmap, v );
    data_offset += voxelmapSize( &rootmap );

    if( v != uncompressed ) {
//        voxelmapFree( v );
        free( v );
    }

    fprintf( stdout, "rootlevel: %d x %d x%d, format %d\n", rootmap.size.x, rootmap.size.y, rootmap.size.z, v->format );


    // Resize the buffer to fit exactly
    m->data_size =data_offset;
    if( !m->is_mmapped )
        m->data_ptr =(char*)realloc( m->data_ptr, m->data_size );
    else
        ftruncate( m->fd, m->data_size );

    // Write the header on top of the buffer
    svmm_header_t *header =(svmm_header_t*)m->data_ptr;
    header->magic1 =SVMM_MAGIC1;
    header->magic2 =SVMM_MAGIC2;
    header->volume_size =uncompressed->size;
    header->format =opts.format; // unused for now
    header->data_start =sizeof( svmm_header_t );
    header->data_length =m->data_size -  sizeof( svmm_header_t);
    header->levels =levels;
    header->blockwidth =opts.blockwidth;
    header->rootsize =rootmap.blocks;
    header->root =rootoffset;

    memcpy( &m->header, header, sizeof( svmm_header_t ) );
    
    return m->data_size;    
}

VOXOWL_HOST 
ssize_t
svmmEncode( svmipmap_t* m,  voxelmap_t* uncompressed, int quality ) {
    svmm_encode_opts_t opts;
    svmmSetOpts( &opts, uncompressed, quality );
    return svmmEncode( m, uncompressed, opts );
}

VOXOWL_HOST 
ssize_t
svmmEncodeFile( const char* filename, voxelmap_t* uncompressed, svmm_encode_opts_t opts ) {

    int fd =open( filename, O_CREAT | O_TRUNC | O_RDWR, S_IRUSR | S_IWUSR );
    if( fd == -1 )
        return -1;

    // We reserve twice the uncompressed size in virtual memory
    // and DATA_RESIZE bytes initially in the file.

    size_t data_size =DATA_RESIZE + sizeof( svmm_header_t );
    size_t vm_max =voxelmapSize( uncompressed ) * 2;

    ftruncate( fd, data_size );

    caddr_t addr =(caddr_t)mmap( (caddr_t)0, vm_max, PROT_READ|PROT_WRITE, MAP_FILE | MAP_SHARED, fd, 0 );

    if( addr == (caddr_t)-1 ) {
        close( fd );
        return -1;
    }

    // Success, create a svmipmap object and pass it to the encode function
    
    svmipmap_t svmm;
    svmm.is_mmapped =true;
    svmm.data_ptr =(char*)addr;
    svmm.data_size =data_size;
    svmm.fd =fd;

    int r =svmmEncode( &svmm, uncompressed, opts );

    // Sync and cleanup
    
    if( msync( addr, /*svmm.data_size*/ vm_max, MS_SYNC ) == -1 )
        return -1;
    if( munmap( addr, /*svmm.data_size*/ vm_max ) == -1 )
        return -1;
    close( fd );

    return r;
}

/* Attempts to set optimal settings for a given voxelmap automatically
   The quality parameter ranges 1-100 */
VOXOWL_HOST 
void 
svmmSetOpts( svmm_encode_opts_t *opts, 
             voxelmap_t* uncompressed, 
             int quality ) {
    opts->blockwidth =4;
    opts->rootwidth =24;
    switch( uncompressed->format ) {
        case VOXEL_RGBA_UINT32:
        case VOXEL_RGB24A3_UINT32:
            opts->format =VOXEL_RGB24A3_UINT32;
            break;
        case VOXEL_RGB24_8ALPHA1_UINT32:
        case VOXEL_RGB24A1_UINT32:
            opts->format =VOXEL_RGB24A1_UINT32;
            break;
        case VOXEL_INTENSITY8_UINT16:
        case VOXEL_INTENSITY8_8ALPHA1_UINT16:
        case VOXEL_INTENSITY_UINT8:
        case VOXEL_BITMAP_UINT8:
            opts->format =VOXEL_INTENSITY8_UINT16;
            break;
        case VOXEL_DENSITY8_UINT16:
        case VOXEL_DENSITY8_8ALPHA1_UINT16:
        case VOXEL_DENSITY_UINT8:
            opts->format =VOXEL_DENSITY8_UINT16;
            break;
    }
    opts->delta =0.5f - (float)quality / 200.f;
    opts->bitmapBaselevel =true;
}

VOXOWL_HOST 
int
svmmFree( svmipmap_t* m ) {
    if( !m->is_mmapped ) {
        if( m->data_ptr )
            free( m->data_ptr );
    } else {
        if( msync( m->data_ptr, m->data_size, MS_SYNC ) == -1 )
            return -1;
        if( munmap( m->data_ptr, m->data_size ) == -1 )
            return -1;
        close( m->fd );
    }
    return 0;
}

VOXOWL_HOST 
int 
svmmOpenMapped( svmipmap_t* svmm, const char *filename ) {
    int fd =open( filename, O_RDONLY );
    if( fd == -1 )
        return -1;

    if( read( fd, (void*)&svmm->header, sizeof( svmm_header_t ) ) == -1 ) {
        close( fd );
        return -1;
    }

    if( svmm->header.magic1 != SVMM_MAGIC1 || 
        svmm->header.magic2 != SVMM_MAGIC2 ) {
        close( fd );
        return -1;
    }

    lseek( fd, 0, SEEK_SET );
    size_t len =svmm->header.data_start + svmm->header.data_length;

    svmm->data_ptr =(char*)mmap( (caddr_t)0, len, PROT_READ, MAP_SHARED, fd, 0 );

    if( svmm->data_ptr == (char*)-1 ) {
        close( fd );
        return -1;
    }

    svmm->fd =fd;
    svmm->data_size =len;
    return 0;

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
    m->data_ptr =(char*)buffer;
}

VOXOWL_HOST
uint64_t
getBlockOffset( voxelmap_t *block, ivec3_32_t subblock_index, ivec3_32_t block_offset, size_t blocksize ) {
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
                ivec3_32_t abs_index =ivec3_32( i + subblock_index.x, 
                                                j + subblock_index.y, 
                                                k + subblock_index.z );
                
                uint8_t lower_bits =*(uint8_t*)voxel( block, abs_index );
                int bit_offs =SVMM_OFFSET_BITS * (k*4+j*2+i); // column-major
                offset |= ((uint64_t)(lower_bits & SVMM_OFFSET_BITS_MASK)) << bit_offs;
                
                // Count the blocks we need to skip, only if they're non-empty
                skip += ( !(lower_bits & SVMM_TERMINAL_BIT_MASK) && n < block_idx ) ? 1 : 0;
                n++;
            }
    // We just calculated the offset of the block pointed to by the first voxel in
    // this subblock. Now skip another 'block_idx' blocks to reach the exact
    // block we're looking for

    offset +=(uint64_t)skip;
    offset *=blocksize;
    return offset;
}

VOXOWL_HOST 
glm::vec4 
svmmDecodeVoxel( svmipmap_t* m, ivec3_32_t position ) {
    char *data_ptr =(char*)m->data_ptr;
    svmm_header_t *header =&m->header;
    int level =header->levels;;

    data_ptr += header->root;
    
    // Size of a block in bytes
    size_t block_size =0;
    // Width in any direction
    size_t block_width =header->blockwidth;
    size_t mipmap_factor =0;

    glm::vec4 color;
    svmm_level_header_t *lheader =(svmm_level_header_t*)data_ptr;
    
    voxelmap_t block;
    block.size =header->rootsize;
    block.format =lheader->format;
    block.blocks =blockCount( lheader->format, header->rootsize );
    block.data =data_ptr + sizeof( svmm_level_header_t );

    while( 1 ) {
       
        level--;
        mipmap_factor = lheader->mipmap_factor; 

        // First, calculate the absolute index of the voxel on the current mipmap level
        ivec3_32_t abs_index =ivec3_32( position.x / mipmap_factor,
                                        position.y / mipmap_factor,
                                        position.z / mipmap_factor );

        // Translate into a relative index into the corresponding block
        ivec3_32_t block_index =ivec3_32( abs_index.x % block.size.x,
                                          abs_index.y % block.size.y,
                                          abs_index.z % block.size.z );
        // The block is devided in subblocks: calculate the offset within the subblock
        ivec3_32_t block_offset = ivec3_32( block_index.x % 2 ? 1 : 0,
                                            block_index.y % 2 ? 1 : 0,
                                            block_index.z % 2 ? 1 : 0 );
        // Finally, calculate the index of the first block of the subblock
        ivec3_32_t subblock_index =block_index; 
        subblock_index.x -=block_offset.x;
        subblock_index.y -=block_offset.y;
        subblock_index.z -=block_offset.z;

        // Retrieve the value we want
        uint8_t lower_bits =0;
        switch( block.format ) {
            case VOXEL_BITMAP_UINT8: {
                // We use parent's color value
                glm::vec4 alpha =voxelmapUnpack( &block, block_index );
                color.a =alpha.a;
                break;
            }
            default:
                color =voxelmapUnpack( &block, block_index );
                lower_bits =*(uint8_t*)voxel( &block, block_index );
                break;
        }

        // If we're either at level 0 (highest mipmap resolution) or the
        // terminal bit is set, we break the loop
        if( level == 0 || isTerminal( lower_bits ) )
            break;

        // We continue, fetch the level header to calucate the negative offset
        data_ptr -=lheader->next;
        // Get the new header
        lheader =(svmm_level_header_t*)data_ptr;

        // Calculate the offset within the next level using the subblock
        block_size =(pow( block_width, 3 ) / voxelsPerBlock( lheader->format ) ) * bytesPerBlock( lheader->format );
        uint64_t offset =getBlockOffset( &block, subblock_index, block_offset, block_size );

//        fprintf( stderr, "Level %d: block { format: %d, size: %d bytes }\n",
//        level-1, lheader->format, block_size );

        // Prepare to read this block
        block.format =lheader->format;
        block.size =ivec3_32( block_width );
        block.blocks =blockCount( block.format, block.size );
        block.data =data_ptr + offset + sizeof( svmm_level_header_t );
        
        // The next mipmap level will have blockwidth times more voxels along
        // each direction
    }

    return color;
}

VOXOWL_HOST 
int 
svmmDecode( voxelmap_t* v, svmipmap_t* svmm ) {

    for( int x=0; x < v->size.x; x++ )
        for( int y=0; y < v->size.y; y++ )
            for( int z=0; z < v->size.z; z++ ) {
                glm::vec4 enc =svmmDecodeVoxel( svmm, ivec3_32( x, y, z ) );
                voxelmapPack( v, ivec3_32( x, y, z ), enc );
            }
    return 0;
}

VOXOWL_HOST 
bool 
svmmTest( voxelmap_t* uncompressed, int quality ) {
    svmipmap_t mipmap;
    mipmap.is_mmapped =false;
    svmm_encode_opts_t opts;
    svmmSetOpts( &opts, uncompressed, quality );
    svmmEncode( &mipmap, uncompressed, opts );

    int errors =0;

    for( int x=0; x < uncompressed->size.x; x++ )
        for( int y=0; y < uncompressed->size.y; y++ )
            for( int z=0; z < uncompressed->size.z; z++ ) {
                //fprintf( stderr, "@ %d, %d, %d: ", x, y, z );
                glm::vec4 orig =voxelmapUnpack( uncompressed, ivec3_32( x, y, z ) );
                glm::vec4 enc =svmmDecodeVoxel( &mipmap, ivec3_32( x, y, z ) );
                bool test =glm::all( glm::lessThanEqual ( glm::abs( orig - enc ), glm::vec4( opts.delta ) ) );
                if( !test ) {
                    if( !errors++ )
                        fprintf( stderr, "Unexpected value. original: (%f %f %f %f), encoded: (%f %f %f, %f), delta: %f, at (%d %d %d)\n",
                        orig.r, orig.g, orig.b, orig.a, enc.r, enc.g, enc.b, enc.a, opts.delta, x, y, z );
                }
                //DEBUG
                voxelmapPack( uncompressed, ivec3_32( x, y, z ), enc );
            }
    fprintf( stdout, "Compressed size: %d, uncompressed size: %d, ratio: %d%\n",
        mipmap.header.data_length, voxelmapSize( uncompressed ), 
        (int)(((float)mipmap.header.data_length / (float)voxelmapSize( uncompressed ))*100.f) );
    fprintf( stdout, "Test completed. errors: %d\n", errors );
    return errors == 0;
}

VOXOWL_HOST
size_t 
ensureSize( svmipmap_t *m, size_t new_size ) {
    if( m->data_size < new_size ) {
        new_size +=(size_t)DATA_RESIZE;

        if( !m->is_mmapped )
            m->data_ptr =(char*)realloc( m->data_ptr, new_size );
        else
            ftruncate( m->fd, new_size );
        m->data_size =new_size;
        return new_size;
    }
    return m->data_size;
}

VOXOWL_HOST
ivec3_32_t
nextMipmapSize( ivec3_32_t size, ivec3_32_t block_size ) {
    ivec3_32_t block_count =ivec3_32( size.x / block_size.x, 
                                      size.y / block_size.y, size.z / block_size.z );

    if( size.x % block_size.x )
        block_count.x++;
    if( size.y % block_size.y )
        block_count.y++;
    if( size.z % block_size.z )
        block_count.z++;
    
    if( block_count.x & 0x1 )
        block_count.x++;
    if( block_count.y & 0x1 )
        block_count.y++;
    if( block_count.z & 0x1 )
        block_count.z++;
    return block_count;
}

VOXOWL_HOST_AND_DEVICE
bool 
isTerminal( uint8_t lower_bits ) {
    return (lower_bits & SVMM_TERMINAL_BIT_MASK);
}

VOXOWL_HOST_AND_DEVICE 
void 
setTerminal( uint8_t* lower_bits, bool terminal ) {
    *lower_bits &= ~SVMM_TERMINAL_BIT_MASK;
    *lower_bits |= (terminal ? SVMM_TERMINAL_BIT_MASK : 0x0);
}

/*VOXOWL_HOST_AND_DEVICE 
bool 
isStub( uint32_t rgb24a1 ) {
    return (rgb24a1 & SVMM_STUB_BIT_MASK);
}

VOXOWL_HOST_AND_DEVICE 
void 
setStub( uint32_t *rgb24a1, bool stub ) {
    *rgb24a1 &= ~SVMM_STUB_BIT_MASK;
    *rgb24a1 |= ((int)stub * SVMM_STUB_BIT_MASK);
}*/

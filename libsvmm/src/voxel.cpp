#include "../include/voxel.h"
//#include "../include/platform.h"
#include <stdio.h>
#include <string.h>

// Source: http://graphics.stanford.edu/~seander/bithacks.html#CountBitsSetTable
static const unsigned char BitsSetTable256[256] = 
{
#   define B2(n) n,     n+1,     n+1,     n+2
#   define B4(n) B2(n), B2(n+1), B2(n+1), B2(n+2)
#   define B6(n) B4(n), B4(n+1), B4(n+1), B4(n+2)
        B6(0), B6(1), B6(1), B6(2)
};

/* Initialize volume and allocate its buffer */
SVMM_HOST
void 
voxelmapCreate( voxelmap_t *v, voxel_format_t f, ivec3_32_t size ) {
    v->format =f;
    v->size =size;
    v->blocks =blockCount( f, size );
    v->scale =ivec3_32( 1 );
    v->data =malloc( voxelmapSize( v ) );
}

SVMM_HOST
void 
voxelmapCreate( voxelmap_t *v, voxel_format_t f, uint32_t size_x, uint32_t size_y, uint32_t size_z ) {
    voxelmapCreate( v, f, ivec3_32( size_x, size_y, size_z ) );
}

/* Free a volume's buffer */
void 
voxelmapFree( voxelmap_t *v ) {
    if( v->data ) free( v->data );
    v->data =NULL;
}

/* Copies src to dst and checks bounds. Only voxelmaps of equal size are copied.
   If the maps have the same format, the copy is a simple call to memcpy(), otherwise
   the data is converted to the destination type first */
SVMM_HOST 
bool 
voxelmapSafeCopy( voxelmap_t* dst, voxelmap_t* src ) {
    if( dst->size.x != src->size.x ||
        dst->size.y != src->size.y ||
        dst->size.z != src->size.z )
        return false;

    if( dst->format == src->format ) {
        memcpy( dst->data, src->data, voxelmapSize( dst ) );
    } else {
        for( int z =0; z < dst->size.z; z++ )
            for( int y =0; y < dst->size.y; y++ )
                for( int x =0; x < dst->size.x; x++ ) {
                    voxelmapPack( dst, ivec3_32( x, y, z ),
                        voxelmapUnpack( src, ivec3_32( x, y, z ) ) );
                }
    }
    return true;
}

/* Return the size of one voxel in BITS */
SVMM_HOST_AND_DEVICE
size_t 
bitsPerVoxel( voxel_format_t f ) {
    size_t s;
    switch( f ) {
        case VOXEL_RGBA_UINT32:
            s =32;
            break;
        case VOXEL_INTENSITY_UINT8:
        case VOXEL_DENSITY_UINT8:
            s =8;
            break;
        case VOXEL_BITMAP_UINT8:
            s =1;
            break;
        case VOXEL_RGB24_8ALPHA1_UINT32:
            s =4;
            break;
        case VOXEL_RGB24A1_UINT32:
        case VOXEL_RGB24A3_UINT32:
            s =32;
            break;
        case VOXEL_INTENSITY8_UINT16:
        case VOXEL_DENSITY8_UINT16:
            s =16;
            break;
        case VOXEL_INTENSITY8_8ALPHA1_UINT16:
        case VOXEL_DENSITY8_8ALPHA1_UINT16:
            s =2;
            break;
        default:
            s =0;
            break;
    }
    return s;
}

/* Returns the size of the smallest accessible block in bytes */
SVMM_HOST_AND_DEVICE
size_t 
bytesPerBlock( voxel_format_t f ) {
    size_t s;
    switch( f ) {
        case VOXEL_RGBA_UINT32:
            s =4;
            break;
        case VOXEL_INTENSITY_UINT8:
        case VOXEL_DENSITY_UINT8:
            s =1;
            break;
        case VOXEL_BITMAP_UINT8:
            s =1;
            break;
        case VOXEL_RGB24_8ALPHA1_UINT32:
            s =4;
            break;
        case VOXEL_RGB24A1_UINT32:
        case VOXEL_RGB24A3_UINT32:
            s =4;
            break;
        case VOXEL_INTENSITY8_UINT16:
        case VOXEL_DENSITY8_UINT16:
            s =2;
            break;
        case VOXEL_INTENSITY8_8ALPHA1_UINT16:
        case VOXEL_DENSITY8_8ALPHA1_UINT16:
            s =2;
            break;
        default:
            s =1;
            break;
    }
    return s;
}

/* Returns the number of voxels that are stored in one elementary block */
SVMM_HOST_AND_DEVICE
size_t 
voxelsPerBlock( voxel_format_t f ) {
    size_t s;
    switch( f ) {
        case VOXEL_RGBA_UINT32:
            s =1;
            break;
        case VOXEL_INTENSITY_UINT8:
        case VOXEL_DENSITY_UINT8:
            s =1;
            break;
        case VOXEL_BITMAP_UINT8:
            s =8;
            break;
        case VOXEL_RGB24_8ALPHA1_UINT32:
            s =8;
            break;
        case VOXEL_RGB24A1_UINT32:
        case VOXEL_RGB24A3_UINT32:
            s =1;
            break;
        case VOXEL_INTENSITY8_UINT16:
        case VOXEL_DENSITY8_UINT16:
            s =1;
            break;
        case VOXEL_INTENSITY8_8ALPHA1_UINT16:
        case VOXEL_DENSITY8_8ALPHA1_UINT16:
            s =8;
            break;
        default:
            s =1;
            break;
    }
    return s;
}

/* If the given format has cubical blocks, returns the width of one block along each axis.
   For example, if blockWidth() = 2, f stores blocks of 2x2x2 voxels */
SVMM_HOST_AND_DEVICE unsigned int blockWidth( voxel_format_t f ) {
    switch( f ) {
        case VOXEL_BITMAP_UINT8:
        case VOXEL_RGB24_8ALPHA1_UINT32:
        case VOXEL_INTENSITY8_8ALPHA1_UINT16:
        case VOXEL_DENSITY8_8ALPHA1_UINT16:
            return 2;
        default:
            break;
    }
    return 1;
}

/* Returns the number of blocks that are required to store a volume of size, given format f */
SVMM_HOST_AND_DEVICE 
ivec3_32_t 
blockCount( voxel_format_t f, ivec3_32_t size ) {
    ivec3_32_t blocks;

    // blockcount = ceil( size / blockwidth(f) )
    // Optimized out, blockwidth can only be 1 or 2

    switch( f ) {
        case VOXEL_INTENSITY8_8ALPHA1_UINT16:
        case VOXEL_DENSITY8_8ALPHA1_UINT16:
        case VOXEL_BITMAP_UINT8:
        case VOXEL_RGB24_8ALPHA1_UINT32:
            // Blockwidth is 2
            blocks =ivec3_sar32( size ); // Arithmetic shift right by one
            // Round up to nearest multiple of two
            if( size.x & 0x1 )
                blocks.x++;
            if( size.y & 0x1 )
                blocks.y++;
            if( size.z & 0x1 )
                blocks.z++;
            break;
        default:
            blocks =size;
            break;

    }

    return blocks;
}

/* Returns the indices of the block that contains the voxel in position, given format f */
SVMM_HOST_AND_DEVICE 
ivec3_32_t 
blockPosition( voxel_format_t f, ivec3_32_t position ) {
    ivec3_32_t block = position;
    // block = floor( size / blockwidth(f) )
    // Optimized out, blockwidth can only be 1 or 2

    switch( f ) {
        case VOXEL_INTENSITY8_8ALPHA1_UINT16:
        case VOXEL_DENSITY8_8ALPHA1_UINT16:
        case VOXEL_BITMAP_UINT8:
        case VOXEL_RGB24_8ALPHA1_UINT32:
            // Blockwidth is 2
            block =ivec3_sar32( position ); // Arithmetic shift right by one
            break;
        default:
            block =position;
            break;

    }
    return block;
}

/* Returns the 1D offset within a block to the voxel in `position`, given format f 
   Column-major (z) indexing is utilized */
SVMM_HOST_AND_DEVICE 
unsigned int 
blockOffset( voxel_format_t f, ivec3_32_t position ) {
    
    switch( f ) {
        case VOXEL_INTENSITY8_8ALPHA1_UINT16:
        case VOXEL_DENSITY8_8ALPHA1_UINT16:
        case VOXEL_BITMAP_UINT8:
        case VOXEL_RGB24_8ALPHA1_UINT32: {
            // Blockwidth is 2
            unsigned int x_offs = position.x & 0x1;
            unsigned int y_offs = position.y & 0x1;
            unsigned int z_offs = position.z & 0x1;
            return 4 * z_offs + 2 * y_offs + x_offs;
        }
        default:
            break;
    }
    return 0;
}

/* Return the size of a volume's data in bytes */
SVMM_HOST_AND_DEVICE
size_t 
voxelmapSize( voxelmap_t *v ) {
//    double bytes = (double)(bitsPerVoxel( v->format ) * v->size.x * v->size.y * v->size.z) / 8;
//    return (size_t)ceil( bytes );
    return (size_t)v->blocks.x * (size_t)v->blocks.y * (size_t)v->blocks.z * (size_t)bytesPerBlock( v->format );
}

SVMM_HOST_AND_DEVICE 
size_t 
voxelmapSize( voxel_format_t f, ivec3_32_t size, ivec3_32_t scale ) {
    ivec3_32_t blocks =blockCount( f, size );
    return (size_t)blocks.x * (size_t)blocks.y * (size_t)blocks.z * (size_t)bytesPerBlock( f );
}

/* Access a block in an array based volume by coordinates. Returns a pointer to the first element */
SVMM_HOST_AND_DEVICE
void* 
voxel( voxelmap_t* v, uint32_t x, uint32_t y, uint32_t z ) {
    size_t bytes_per_block =bytesPerBlock( v->format );
    ivec3_32_t block =blockPosition( v->format, ivec3_32( x, y, z ) );
    //size_t offset =( v->blocks.z * v->blocks.y * block.x + v->blocks.z * block.y + block.z ) ;
    // CUDA's textures use column-major order
    size_t offset =( (size_t)v->blocks.x * (size_t)v->blocks.y * (size_t)block.z 
            + (size_t)v->blocks.x * (size_t)block.y 
            + (size_t)block.x ) ; 
    return (void*)((char*)v->data + offset * bytes_per_block );
}

SVMM_HOST_AND_DEVICE
void* 
voxel( voxelmap_t* v, ivec3_32_t position ) {
    return voxel( v, position.x, position.y, position.z );
}


/* Fills a given volume by copying a value to every position.  */
SVMM_HOST_AND_DEVICE
void 
voxelmapFill( voxelmap_t* v, void *value ) {

    uint32_t block_size =bytesPerBlock( v->format );
    size_t byte_count =voxelmapSize( v );
    for( size_t i =0; i < byte_count; i+= block_size ) {
        memcpy( (char*)v->data + i, value, block_size );
    }
    
}

/* Pack an rgba-4float value into an arbitrarily formatted voxelmap */
SVMM_HOST 
void 
voxelmapPack( voxelmap_t* v, ivec3_32_t position, glm::vec4 rgba ) {
   void *block_ptr =voxel( v, position );
   ivec3_32_t block =blockPosition( v->format, position );
   switch( v->format ) {
        case VOXEL_RGBA_UINT32:
            packRGBA_UINT32( (uint32_t*)block_ptr, rgba );
            break;
        case VOXEL_INTENSITY_UINT8:
            packINTENSITY_UINT8( (uint8_t*)block_ptr, intensityRGBA_linear( rgba, true ) );
            break;
        case VOXEL_DENSITY_UINT8:
            rgba.a = rgba.a < .001f ? 0.f : 1.f;
            rgba *= rgba.a;
            packINTENSITY_UINT8( (uint8_t*)block_ptr, intensityRGBA_linear( rgba, false ) );
            break;
        case VOXEL_BITMAP_UINT8: {
            unsigned bit_offs =blockOffset( v->format, position );
            packBIT_UINT8( (uint8_t*)block_ptr, bit_offs, thresholdRGBA_linear( rgba, true ) );
            break;
        }
        case VOXEL_RGB24_8ALPHA1_UINT32: {
            unsigned bit_offs =blockOffset( v->format, position );
            packRGBA_RGB24_8ALPHA1_UINT32( (uint32_t*)block_ptr, bit_offs, rgba );
            break;
        }
        case VOXEL_INTENSITY8_8ALPHA1_UINT16: {
            unsigned bit_offs =blockOffset( v->format, position );
            packRGBA_INTENSITY8_8ALPHA1_UINT16( (uint16_t*)block_ptr, bit_offs, rgba );
            break;
        }
        case VOXEL_DENSITY8_8ALPHA1_UINT16: {
            unsigned bit_offs =blockOffset( v->format, position );
            rgba.a = rgba.a < .001f ? 0.f : 1.f;
            packRGBA_INTENSITY8_8ALPHA1_UINT16( (uint16_t*)block_ptr, bit_offs, rgba );
            break;
        }
        case VOXEL_RGB24A1_UINT32:
            packRGB24A1_UINT32( (uint32_t*)block_ptr, rgba );
            break;
        case VOXEL_RGB24A3_UINT32:
            packRGB24A3_UINT32( (uint32_t*)block_ptr, rgba );
            break;
        case VOXEL_INTENSITY8_UINT16:
            packINTENSITY8_UINT16( (uint16_t*)block_ptr, intensityRGBA_linear( rgba, true ) );
            break;
        case VOXEL_DENSITY8_UINT16:
            packINTENSITY8_UINT16( (uint16_t*)block_ptr, intensityRGBA_linear( rgba, false ) );
            break;
   }
}

/* Unpack an rgba-4float value from an arbitrarily formatted  voxelmap */
SVMM_HOST 
glm::vec4 
voxelmapUnpack( voxelmap_t* v, ivec3_32_t position ) {
   void *block_ptr =voxel( v, position );
   glm::vec4 rgba;
   switch( v->format ) {
        case VOXEL_RGBA_UINT32:
            rgba =unpackRGBA_UINT32( *((uint32_t*)block_ptr) );
            break;
        case VOXEL_INTENSITY_UINT8:
            rgba =glm::vec4( unpackINTENSITY_UINT8( *((uint8_t*)block_ptr) ) );
            rgba.a =rgba.r != 0.f;
            break;
        case VOXEL_DENSITY_UINT8:
            rgba =glm::vec4( unpackINTENSITY_UINT8( *((uint8_t*)block_ptr) ) );
            break;
        case VOXEL_BITMAP_UINT8: {
            unsigned bit_offs =blockOffset( v->format, position );
            rgba =glm::vec4( (int)unpackBIT_UINT8( *((uint8_t*)block_ptr), bit_offs ) );
            break;
        }
        case VOXEL_RGB24_8ALPHA1_UINT32: {
            unsigned bit_offs =blockOffset( v->format, position );
            rgba =unpackRGBA_RGB24_8ALPHA1_UINT32( *((uint32_t*)block_ptr), bit_offs );
            break;
        }
        case VOXEL_INTENSITY8_8ALPHA1_UINT16: {
            unsigned bit_offs =blockOffset( v->format, position );
            rgba =unpackRGBA_INTENSITY8_8ALPHA1_UINT16( *((uint16_t*)block_ptr), bit_offs );
            break;
        }
        case VOXEL_DENSITY8_8ALPHA1_UINT16: {
            unsigned bit_offs =blockOffset( v->format, position );
            rgba =unpackRGBA_INTENSITY8_8ALPHA1_UINT16( *((uint16_t*)block_ptr), bit_offs );
            rgba = rgba * rgba.a;
            rgba.a =rgba.r;
            break;
        }
        case VOXEL_RGB24A1_UINT32:
            rgba =unpackRGB24A1_UINT32( *((uint32_t*)block_ptr) );
            break;
        case VOXEL_RGB24A3_UINT32:
            rgba =unpackRGB24A3_UINT32( *((uint32_t*)block_ptr) );
            break;
        case VOXEL_INTENSITY8_UINT16:
            rgba =glm::vec4( unpackINTENSITY8_UINT16( *((uint16_t*)block_ptr ) ) );
            rgba.a =rgba.r != 0.f;
            break;
        case VOXEL_DENSITY8_UINT16:
            rgba =glm::vec4( unpackINTENSITY8_UINT16( *((uint16_t*)block_ptr ) ) );
            break;
        
   }
   return rgba;
}


SVMM_HOST_AND_DEVICE
void
packRGBA_UINT32( uint32_t* rgba, const glm::vec4 &v ) {
    *rgba =
        ((uint32_t)(v.r * 255) << 24) |
        ((uint32_t)(v.g * 255) << 16) |
        ((uint32_t)(v.b * 255) << 8) |
        (uint32_t)(v.a * 255);
}

SVMM_HOST_AND_DEVICE
glm::vec4
unpackRGBA_UINT32( uint32_t rgba ) {
    return glm::vec4(
        (float)((rgba >> 24) & 0xff) / 255.f,
        (float)((rgba >> 16) & 0xff) / 255.f,
        (float)((rgba >> 8) & 0xff) / 255.f,
        (float)(rgba & 0xff) / 255.f );
}

SVMM_HOST_AND_DEVICE
glm::vec4
unpackABGR_UINT32( uint32_t abgr ) {
    return glm::vec4(
        (float)(abgr & 0xff) / 255.f,
        (float)((abgr >> 8) & 0xff) / 255.f,
        (float)((abgr >> 16) & 0xff) / 255.f,
        (float)((abgr >> 24) & 0xff) / 255.f );
}

/* Unpack an rgba-4float value from four uint8s */
SVMM_HOST_AND_DEVICE 
glm::vec4 
unpackRGBA_4UINT8( uint8_t r, uint8_t g, uint8_t b, uint8_t a ) {
    return glm::vec4(
        (float)(r) / 255.f,
        (float)(g) / 255.f,
        (float)(b) / 255.f,
        (float)(a) / 255.f );
}

/* Pack an grayscale intensity to an uint8 */
SVMM_HOST_AND_DEVICE 
void 
packINTENSITY_UINT8( uint8_t* dst, float intensity ) {
    // Trivial, but anyway
    *dst =(uint8_t)(intensity * 255.f);
}

/* Unpack an grayscale intensity from an uint8 */
SVMM_HOST_AND_DEVICE 
float 
unpackINTENSITY_UINT8( uint8_t intensity ) {
    return intensity / 255.f;
}

/* Pack a bitmap (array of bool) into a single uint8 */
SVMM_HOST_AND_DEVICE 
void 
packBITMAP_UINT8( uint8_t* dst, bool src[8] ) {
    *dst =0x0;
    uint8_t mask =0x1;
    for( int i =0; i < 8; i++ ) {
        *dst |= src[i] * mask;
        mask = mask << 1;
    }
}

/* Unpack a bitmap (array of bool) from an uint8 */
SVMM_HOST_AND_DEVICE 
void 
unpackBITMAP_UINT8( bool dst[8], uint8_t src ) {
    uint8_t mask =0x1;
    for( int i =0; i < 8; i++ ) {
        dst[i] = src & mask;
        mask = mask << 1;
    }
}

/* Pack a single bit from a bitmap into an uint8. offset gives the offset in the uint8 where the LSB is zero */
SVMM_HOST_AND_DEVICE 
void 
packBIT_UINT8( uint8_t* dst, int offset, bool b ) {
    uint8_t mask = ((uint8_t)0x1 << offset);
    if( b )
        *dst |= mask;
    else
        *dst &= ~mask;
}

/* Unpack a single bit from an uint8. offset gives the offset in the uint8 where the LSB is zero */
SVMM_HOST_AND_DEVICE 
bool 
unpackBIT_UINT8( uint8_t src, int offset ) {
    return src & ((uint8_t)0x1 << offset);
}

/* Pack a RGBA-4float value into a rgb24_8alpha1_uint32, where offset determines 
   the bit number [0,7] of the alpha channel starting from the LSB
   The given RGB value will be averaged against any of the existing elements with alpha=1
   The A value will be applied a threshold at 0.5 */
SVMM_HOST 
void 
packRGBA_RGB24_8ALPHA1_UINT32( uint32_t *rgb24_8alpha1, int offset, glm::vec4 rgba ) {
    uint8_t lower_bits = (uint8_t)*rgb24_8alpha1;
    int count =BitsSetTable256[lower_bits];
    glm::vec3 color(0);
    if( count ) {
        color = glm::vec3 (
            (float)(((*rgb24_8alpha1) >> 24) & 0xff) / 255.f,
            (float)(((*rgb24_8alpha1) >> 16) & 0xff) / 255.f,
            (float)(((*rgb24_8alpha1) >> 8) & 0xff) / 255.f );
    }
    if( rgba.a >= 0.5f ) {
        color *= (float)count;
        color += glm::vec3( rgba );
        color /= (float)(count+1);
    }

    packBIT_UINT8( &lower_bits, offset, rgba.a >= 0.5f ); 
    *rgb24_8alpha1 =(uint32_t)lower_bits | 
        ((uint32_t)(color.r * 255.f) << 24) |
        ((uint32_t)(color.g * 255.f) << 16) |
        ((uint32_t)(color.b * 255.f) << 8);
    
}

/* Unpack a RGBA-4float from a rgb24_8alpha_uint32, where offset determines the bit number [0,7] of the alpha channel */
SVMM_HOST_AND_DEVICE 
glm::vec4 
unpackRGBA_RGB24_8ALPHA1_UINT32( uint32_t rgb24_8alpha1, int offset ) {
    return glm::vec4 (
        (float)((rgb24_8alpha1 >> 24) & 0xff) / 255.f,
        (float)((rgb24_8alpha1 >> 16) & 0xff) / 255.f,
        (float)((rgb24_8alpha1 >> 8) & 0xff) / 255.f,
        (float)unpackBIT_UINT8( rgb24_8alpha1, offset ) );
}

/* Pack one RGB-3float and a alpha bitmap [0,7] into on uint32 using rgb24_8alpha1 encoding */
SVMM_HOST_AND_DEVICE 
void 
packRGB24_8ALPHA1_UINT32( uint32_t *dst, glm::vec3 rgb, bool alpha[8] ) {
    uint8_t lower_bits;
    packBITMAP_UINT8( &lower_bits, alpha );
    *dst =(uint8_t)lower_bits | 
        ((uint32_t)(rgb.r * 255.f) << 24) |
        ((uint32_t)(rgb.g * 255.f) << 16) |
        ((uint32_t)(rgb.b * 255.f) << 8);
}


/* Unpack one RGB-3float and a alpha bitmap [0,7] from one uint32 using rgb24_8alpha1 encoding */
SVMM_HOST_AND_DEVICE 
glm::vec3 
unpackRGB24_8ALPHA1_UINT32( bool alpha[8], uint16_t rgb24_8alpha1 ) {
    uint8_t lower_bits = rgb24_8alpha1;
    unpackBITMAP_UINT8( alpha, lower_bits );
    return glm::vec3 (
        (float)((rgb24_8alpha1 >> 24) & 0xff) / 255.f,
        (float)((rgb24_8alpha1 >> 16) & 0xff) / 255.f,
        (float)((rgb24_8alpha1 >> 8) & 0xff) / 255.f );
}

/*! Pack a RGBA-4float value into a intensity8_8alpha1_uint16, where offset determines 
   the bit number [0,7] of the alpha channel starting from the LSB. Color is discarded.
   The given RGB value will be averaged against any of the existing elements with alpha=1
   The A value will be applied a threshold at 0.5 */
SVMM_HOST 
void 
packRGBA_INTENSITY8_8ALPHA1_UINT16( uint16_t *intensity8_8alpha1, int offset, glm::vec4 rgba ) {
    uint8_t lower_bits = (uint8_t)*intensity8_8alpha1;
    int count =BitsSetTable256[lower_bits];
    float intensity =0;
    if( count ) {
        intensity =(float)((*intensity8_8alpha1 >> 8) & 0xff) / 255.f;
    }
    if( rgba.a >= 0.5f ) {
        intensity *= (float)count;
        intensity += intensityRGBA_linear( rgba, false );
        intensity /= (float)(count+1);
    }

    packBIT_UINT8( &lower_bits, offset, rgba.a >= 0.5f ); 
    *intensity8_8alpha1 =(uint8_t)lower_bits | ((uint16_t)(intensity * 255.f) << 8);
}

/* Unpack a RGBA-4float from a intensity8_8alpha1_uint16, where offset determines the bit number [0,7] of the alpha channel */
SVMM_HOST_AND_DEVICE 
glm::vec4 
unpackRGBA_INTENSITY8_8ALPHA1_UINT16( uint16_t intensity8_8alpha1, int offset ) {
    float intensity =(float)((intensity8_8alpha1 >> 8) & 0xff) / 255.f;
    return glm::vec4 ( intensity, intensity, intensity, (float)unpackBIT_UINT8( intensity8_8alpha1, offset ) );
}

/*! Pack one intensity float and an alpha bitmap [0,7] into one uint16 using intensity8_8alpha1 encoding */
SVMM_HOST_AND_DEVICE 
void 
packINTENSITY8_8ALPHA1_UINT16( uint16_t *dst, float intensity, bool alpha[8] ) {
    uint8_t lower_bits;
    packBITMAP_UINT8( &lower_bits, alpha );
    *dst =(uint8_t)lower_bits | ((uint32_t)(intensity * 255.f) << 8);

}

/*! Unpack one intensity float and a alpha bitmap [0,7] from one uint16 using intensity8_8alpha1 encoding */
SVMM_HOST_AND_DEVICE 
float 
unpackINTENSITY8_8ALPHA1_UINT16( bool alpha[8], uint16_t intensity8_8alpha1 ) {
    uint8_t lower_bits = intensity8_8alpha1;
    unpackBITMAP_UINT8( alpha, lower_bits );
    return (float)((intensity8_8alpha1 >> 8) & 0xff) / 255.f;
}

/* Pack a RGBA-4float into a rgb24a1. The RGB components are packed in 8 bpc in the MSB's
   A threshold is applied to the alpha value ( >= 0.5 ) and is packed into the 7th bit. 
   Bits 0-6 (from LSB) are untouched */
SVMM_HOST_AND_DEVICE 
void 
packRGB24A1_UINT32( uint32_t *dst, glm::vec4 rgba ) {
    // Clear bits [7-31]
    *dst &= 0x7F;
    *dst = *dst | 
        ((uint32_t)(rgba.r * 255) << 24) |
        ((uint32_t)(rgba.g * 255) << 16) |
        ((uint32_t)(rgba.b * 255) << 8) |
        (uint32_t)0x80 * (int)(rgba.a >= .1f);
}

/* Unpack a RGBA-4float from a rgb24a1 type. Bit 7 contains the one-bit alpha. Bits 0-6 (from LSB) are untoched. */
SVMM_HOST_AND_DEVICE 
glm::vec4 
unpackRGB24A1_UINT32( uint32_t rgb24a1 ) {
    return glm::vec4(
        (float)((rgb24a1 >> 24) & 0xff) / 255.f,
        (float)((rgb24a1 >> 16) & 0xff) / 255.f,
        (float)((rgb24a1 >> 8) & 0xff) / 255.f,
        (float)((rgb24a1 & 0x80) != 0) );
}

/*! Pack a RGBA-4float into a rgb24a3. The RGB components are packed in 8 bpc in the MSB's
   The alpha is converted to 3 bits which are stored in bits 7-5 
   Bits 0-4 (from LSB) are untouched */
SVMM_HOST_AND_DEVICE 
void 
packRGB24A3_UINT32( uint32_t *dst, glm::vec4 rgba ) {
    // Clear bits [5-31]
    *dst &= 0x1F;
    *dst = *dst | 
        ((uint32_t)(rgba.r * 255) << 24) |
        ((uint32_t)(rgba.g * 255) << 16) |
        ((uint32_t)(rgba.b * 255) << 8) |
        (uint32_t)(rgba.a * 7) << 5;

}

/*! Unpack a RGBA-4float from a rgb24a3 type. Bits 7-5 contain the 3-bit alpha. Bits 0-4 (from LSB) are untouched. */
SVMM_HOST_AND_DEVICE 
glm::vec4 
unpackRGB24A3_UINT32( uint32_t rgb24a3 ) {
    return glm::vec4(
        (float)((rgb24a3 >> 24) & 0xff) / 255.f,
        (float)((rgb24a3 >> 16) & 0xff) / 255.f,
        (float)((rgb24a3 >> 8) & 0xff) / 255.f,
        (float)((rgb24a3 >> 5) & 0x7) / 7.f ) ;

}

/*! Pack an 8-bit grayscale intensity to an uint16 */
SVMM_HOST_AND_DEVICE 
void 
packINTENSITY8_UINT16( uint16_t* dst, float intensity ) {
    *dst &= 0xff;
    *dst = *dst | ((uint16_t)(intensity * 255)) << 8;
}

/*! Unpack an 8-bit grayscale intensity from an uint16 */
SVMM_HOST_AND_DEVICE 
float 
unpackINTENSITY8_UINT16( uint16_t intensity ) {
    return (float)((intensity >> 8) & 0xff) / 255.f;
}

/*
 * Misc functions
 */


/* Calculate the perceived intensity of an RGBA quadruple. Values are assumed to be linear */
SVMM_HOST_AND_DEVICE 
float 
intensityRGBA_linear( glm::vec4 rgba, bool multiply_alpha ) {
    // We use CIE 1931 weights and alpha as absolute weight using the following formula
    // Y = A( 0.2126R + 0.7152G + 0.0722B )
    return ( (multiply_alpha ? rgba.a : 1.f) * ( 0.2126f * rgba.r + 0.7152f * rgba.g + 0.0722 * rgba.b ) );
}

/* Calculate the perceived intensity of an RGBA quadruple. Values are assumed to be linear */
SVMM_HOST_AND_DEVICE 
uint8_t 
intensityRGBA_UINT32_linear( uint32_t rgba, bool multiply_alpha ) {
    return intensityRGBA_linear( unpackRGBA_UINT32( rgba ), multiply_alpha ) / 255.f;
}

/* Calculate the perceived intensity of an RGBA quadruple by fast, inaccurate conversion. Values are assumed to be linear */
SVMM_HOST_AND_DEVICE 
uint8_t 
intensityRGBA_UINT32_fastlinear( uint32_t rgba ) {
    // We apply an approximation Y =0.25R + 0.5G + 0.25B - (1-A) by bitshifting

    return 
        (uint8_t)((rgba & 0xff000000) >> 24+2) +
        (uint8_t)((rgba & 0x00ff0000) >> 16+1) +
        (uint8_t)((rgba & 0x0000ff00) >> 8+2) -
        (255 - (uint8_t)(rgba & 0xff));
}

/* Calculate if a RGBA quadruple is 'black' (false) or 'white' depending on the given treshold [0,1] */
SVMM_HOST_AND_DEVICE 
bool 
thresholdRGBA_linear( glm::vec4 rgba, float threshold ) {
    return intensityRGBA_linear( rgba, true ) >= threshold;
}

/* Calculate if a RGBA quadruple is 'black' (false) or 'white' depending on the given treshold [0,1] */
SVMM_HOST_AND_DEVICE 
bool 
thresholdRGBA_UINT32_linear( uint32_t rgba, float threshold ) {
    return intensityRGBA_UINT32_linear( rgba, true ) >= (255.f * threshold);
}

/* Calculate if a RGBA quadruple is 'black' (false) or 'white' using fast, inaccurate conversion, depending on the given treshold [0,255] */
SVMM_HOST_AND_DEVICE 
bool 
thresholdRGBA_UINT32_fastlinear( uint32_t rgba, uint8_t threshold  ) {
    return intensityRGBA_UINT32_fastlinear( rgba ) >= threshold;
}

SVMM_HOST_AND_DEVICE
glm::vec4
blendF2B( glm::vec4 src, glm::vec4 dst ) {
    glm::vec4 c;
    c.r = dst.a*(src.r * src.a) + dst.r;
    c.g = dst.a*(src.g * src.a) + dst.g;
    c.b = dst.a*(src.b * src.a) + dst.b;
    c.a = (1.f - src.a) * dst.a;
    return c;
}

SVMM_HOST 
const char* 
strVoxelFormat( voxel_format_t f ) {
    switch( f ) {
        case VOXEL_RGBA_UINT32:
            return "RGBA_UINT32";
        case VOXEL_INTENSITY_UINT8:
            return "INTENSITY_UINT8";
        case VOXEL_DENSITY_UINT8:
            return "DENSITY_UINT8";
        case VOXEL_BITMAP_UINT8:
            return "BITMAP_UINT8";
        case VOXEL_RGB24_8ALPHA1_UINT32:
            return "RGB24_8ALPHA1_UINT32";
        case VOXEL_RGB24A1_UINT32:
            return "RGB24A1_UINT32";
        case VOXEL_RGB24A3_UINT32:
            return "RGB24A3_UINT32";
        case VOXEL_INTENSITY8_UINT16:
            return "INTENSITY8_UINT16";
        case VOXEL_DENSITY8_UINT16:
            return "DTENSITY8_UINT16";
        case VOXEL_INTENSITY8_8ALPHA1_UINT16:
            return "INTENSITY8_8ALPHA1_UINT16";
        case VOXEL_DENSITY8_8ALPHA1_UINT16:
            return "DTENSITY8_8ALPHA1_UINT16";
    }
    return "";
}

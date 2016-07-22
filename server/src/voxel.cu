#include "voxel.h"
#include <stdio.h>

/* Initialize volume and allocate its buffer */
VOXOWL_HOST
void 
voxelmapCreate( voxelmap_t *v, voxel_format_t f, ivec3_16_t size ) {
    v->format =f;
    v->size =size;
    v->blocks =blockCount( f, size );
    v->data =malloc( voxelmapSize( v ) );
}

VOXOWL_HOST
void 
voxelmapCreate( voxelmap_t *v, voxel_format_t f, uint32_t size_x, uint32_t size_y, uint32_t size_z ) {
    voxelmapCreate( v, f, ivec3_16( size_x, size_y, size_z ) );
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
VOXOWL_HOST 
bool 
voxelmapSafeCopy( voxelmap_t* dst, voxelmap_t* src ) {
    if( dst->size.x != src->size.x ||
        dst->size.y != src->size.y ||
        dst->size.z != src->size.z )
        return false;

    if( dst->format == src->format ) {
        memcpy( dst->data, src->data, voxelmapSize( dst ) );
    } else {
        for( int x =0; x < dst->size.x; x++ )
            for( int y =0; y < dst->size.y; y++ )
                for( int z =0; z < dst->size.z; z++ ) {
                    voxelmapPack( dst, ivec3_16( x, y, z ),
                        voxelmapUnpack( src, ivec3_16( x, y, z ) ) );
                }
    }
    return true;
}

/* Return the size of one voxel in BITS */
VOXOWL_HOST_AND_DEVICE
size_t 
bitsPerVoxel( voxel_format_t f ) {
    size_t s;
    switch( f ) {
        case VOXEL_RGBA_UINT32:
            s =32;
            break;
        case VOXEL_INTENSITY_UINT8:
            s =8;
            break;
        case VOXEL_BITMAP_UINT8:
            s =1;
            break;
        case VOXEL_RGB24_8ALPHA1_UINT32:
            s =4;
            break;
        case VOXEL_RGB24A1_UINT32:
            s =32;
            break;
        default:
            s =0;
            break;
    }
    return s;
}

/* Returns the size of the smallest accessible block in bytes */
VOXOWL_HOST_AND_DEVICE
size_t 
bytesPerBlock( voxel_format_t f ) {
    size_t s;
    switch( f ) {
        case VOXEL_RGBA_UINT32:
            s =4;
            break;
        case VOXEL_INTENSITY_UINT8:
            s =1;
            break;
        case VOXEL_BITMAP_UINT8:
            s =1;
            break;
        case VOXEL_RGB24_8ALPHA1_UINT32:
            s =4;
            break;
        case VOXEL_RGB24A1_UINT32:
            s =4;
            break;
        default:
            s =1;
            break;
    }
    return s;
}

/* Returns the number of voxels that are stored in one elementary block */
VOXOWL_HOST_AND_DEVICE
size_t 
voxelsPerBlock( voxel_format_t f ) {
    size_t s;
    switch( f ) {
        case VOXEL_RGBA_UINT32:
            s =1;
            break;
        case VOXEL_INTENSITY_UINT8:
            s =1;
            break;
        case VOXEL_BITMAP_UINT8:
            s =8;
            break;
        case VOXEL_RGB24_8ALPHA1_UINT32:
            s =8;
            break;
        case VOXEL_RGB24A1_UINT32:
            s =1;
            break;
        default:
            s =1;
            break;
    }
    return s;
}

/* If the given format has cubical blocks, returns the width of one block along each axis.
   For example, if blockWidth() = 2, f stores blocks of 2x2x2 voxels */
VOXOWL_HOST_AND_DEVICE unsigned int blockWidth( voxel_format_t f ) {
    switch( f ) {
        case VOXEL_BITMAP_UINT8:
        case VOXEL_RGB24_8ALPHA1_UINT32:
            return 2;
        default:
            break;
    }
    return 1;
}

/* Returns the number of blocks that are required to store a volume of size, given format f */
VOXOWL_HOST_AND_DEVICE 
ivec3_16_t 
blockCount( voxel_format_t f, ivec3_16_t size ) {
    ivec3_16_t blocks;

    // blockcount = ceil( size / blockwidth(f) )

    unsigned int blockwidth =blockWidth( f );
    blocks.x =size.x / blockwidth;
    blocks.y =size.y / blockwidth;
    blocks.z =size.z / blockwidth;

    if( size.x % blockwidth )
        blocks.x++;
    if( size.y % blockwidth )
        blocks.y++;
    if( size.z % blockwidth )
        blocks.z++;

    return blocks;

}

/* Returns the indices of the block that contains the voxel in position, given format f */
VOXOWL_HOST_AND_DEVICE 
ivec3_16_t 
blockPosition( voxel_format_t f, ivec3_16_t position ) {
    ivec3_16_t block = position;
    unsigned int blockwidth =blockWidth( f );
    block.x =position.x / blockwidth;
    block.y =position.y / blockwidth;
    block.z =position.z / blockwidth;
    return block;
}

/* Returns the 1D offset within a block to the voxel in `position`, given format f 
   Column-major (z) indexing is utilized */
VOXOWL_HOST_AND_DEVICE 
unsigned int 
blockOffset( voxel_format_t f, ivec3_16_t position ) {
    unsigned int blockwidth =blockWidth( f );
    unsigned int x_offs = position.x % blockwidth;
    unsigned int y_offs = position.y % blockwidth;
    unsigned int z_offs = position.z % blockwidth;
    return blockwidth * blockwidth * z_offs + blockwidth * y_offs + x_offs;
}

/* Return the size of a volume's data in bytes */
VOXOWL_HOST_AND_DEVICE
size_t 
voxelmapSize( voxelmap_t *v ) {
//    double bytes = (double)(bitsPerVoxel( v->format ) * v->size.x * v->size.y * v->size.z) / 8;
//    return (size_t)ceil( bytes );
    return v->blocks.x * v->blocks.y * v->blocks.z * bytesPerBlock( v->format );
}

/* Access a block in an array based volume by coordinates. Returns a pointer to the first element */
VOXOWL_HOST_AND_DEVICE
void* 
voxel( voxelmap_t* v, uint32_t x, uint32_t y, uint32_t z ) {
    size_t bytes_per_block =bytesPerBlock( v->format );
    ivec3_16_t block =blockPosition( v->format, ivec3_16( x, y, z ) );
    //size_t offset =( v->blocks.z * v->blocks.y * block.x + v->blocks.z * block.y + block.z ) ;
    // CUDA's textures use column-major order
    size_t offset =( v->blocks.x * v->blocks.y * block.z + v->blocks.x * block.y + block.x ) ; 
    return (void*)((char*)v->data + offset * bytes_per_block );
}

VOXOWL_HOST_AND_DEVICE
void* 
voxel( voxelmap_t* v, ivec3_16_t position ) {
    return voxel( v, position.x, position.y, position.z );
}


/* Fills a given volume by copying a value to every position.  */
VOXOWL_HOST_AND_DEVICE
void 
voxelmapFill( voxelmap_t* v, void *value ) {

    uint32_t block_size =bytesPerBlock( v->format );
    size_t byte_count =voxelmapSize( v );
    for( size_t i =0; i < byte_count; i+= block_size ) {
        memcpy( (char*)v->data + i, value, block_size );
    }
    
}

/* Pack an rgba-4float value into an arbitrarily formatted voxelmap */
VOXOWL_HOST_AND_DEVICE 
void 
voxelmapPack( voxelmap_t* v, ivec3_16_t position, glm::vec4 rgba ) {
   void *block_ptr =voxel( v, position );
   ivec3_16_t block =blockPosition( v->format, position );
   switch( v->format ) {
        case VOXEL_RGBA_UINT32:
            packRGBA_UINT32( (uint32_t*)block_ptr, rgba );
            break;
        case VOXEL_INTENSITY_UINT8:
            packINTENSITY_UINT8( (uint8_t*)block_ptr, intensityRGBA_linear( rgba ) );
            break;
        case VOXEL_BITMAP_UINT8: {
            unsigned bit_offs =blockOffset( v->format, position );
            packBIT_UINT8( (uint8_t*)block_ptr, bit_offs, thresholdRGBA_linear( rgba ) );
            break;
        }
        case VOXEL_RGB24_8ALPHA1_UINT32: {
            unsigned bit_offs =blockOffset( v->format, position );
            packRGBA_RGB24_8ALPHA1_UINT32( (uint32_t*)block_ptr, bit_offs, rgba );
            break;
        }
        case VOXEL_RGB24A1_UINT32:
            packRGB24A1_UINT32( (uint32_t*)block_ptr, rgba );
            break;

   }
}

/* Unpack an rgba-4float value from an arbitrarily formatted  voxelmap */
VOXOWL_HOST_AND_DEVICE 
glm::vec4 
voxelmapUnpack( voxelmap_t* v, ivec3_16_t position ) {
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
        case VOXEL_BITMAP_UINT8: {
            unsigned bit_offs =blockOffset( v->format, position );
            rgba =glm::vec4( (int)unpackBIT_UINT8( *((uint8_t*)block_ptr), bit_offs ) );
            break;
        }
        case VOXEL_RGB24_8ALPHA1_UINT32: {
            size_t bit_offs =position.z % voxelsPerBlock( v->format );
            rgba =unpackRGBA_RGB24_8ALPHA1_UINT32( *((uint32_t*)block_ptr), bit_offs );
            break;
        }
        case VOXEL_RGB24A1_UINT32:
            rgba =unpackRGB24A1_UINT32( *((uint32_t*)block_ptr) );
            break;
        
   }
   return rgba;
}


VOXOWL_HOST_AND_DEVICE
void
packRGBA_UINT32( uint32_t* rgba, const glm::vec4 &v ) {
    *rgba =
        ((uint32_t)(v.r * 255) << 24) |
        ((uint32_t)(v.g * 255) << 16) |
        ((uint32_t)(v.b * 255) << 8) |
        (uint32_t)(v.a * 255);
}

VOXOWL_HOST_AND_DEVICE
glm::vec4
unpackRGBA_UINT32( uint32_t rgba ) {
    return glm::vec4(
        (float)((rgba >> 24) & 0xff) / 255.f,
        (float)((rgba >> 16) & 0xff) / 255.f,
        (float)((rgba >> 8) & 0xff) / 255.f,
        (float)(rgba & 0xff) / 255.f );
}

/* Unpack an rgba-4float value from four uint8s */
VOXOWL_HOST_AND_DEVICE 
glm::vec4 
unpackRGBA_4UINT8( uint8_t r, uint8_t g, uint8_t b, uint8_t a ) {
    return glm::vec4(
        (float)(r) / 255.f,
        (float)(g) / 255.f,
        (float)(b) / 255.f,
        (float)(a) / 255.f );
}

/* Pack an grayscale intensity to an uint8 */
VOXOWL_HOST_AND_DEVICE 
void 
packINTENSITY_UINT8( uint8_t* dst, float intensity ) {
    // Trivial, but anyway
    *dst =(uint8_t)(intensity * 255.f);
}

/* Unpack an grayscale intensity from an uint8 */
VOXOWL_HOST_AND_DEVICE 
float 
unpackINTENSITY_UINT8( uint8_t intensity ) {
    return intensity / 255.f;
}

/* Pack a bitmap (array of bool) into a single uint8 */
VOXOWL_HOST_AND_DEVICE 
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
VOXOWL_HOST_AND_DEVICE 
void 
unpackBITMAP_UINT8( bool dst[8], uint8_t src ) {
    uint8_t mask =0x1;
    for( int i =0; i < 8; i++ ) {
        dst[i] = src & mask;
        mask = mask << 1;
    }
}

/* Pack a single bit from a bitmap into an uint8. offset gives the offset in the uint8 where the LSB is zero */
VOXOWL_HOST_AND_DEVICE 
void 
packBIT_UINT8( uint8_t* dst, int offset, bool b ) {
    uint8_t mask = ((uint8_t)0x1 << offset);
    if( b )
        *dst |= mask;
    else
        *dst &= ~mask;
}

/* Unpack a single bit from an uint8. offset gives the offset in the uint8 where the LSB is zero */
VOXOWL_HOST_AND_DEVICE 
bool 
unpackBIT_UINT8( uint8_t src, int offset ) {
    return src & ((uint8_t)0x1 << offset);
}

/* Pack a RGBA-4float value into a rgb24_8alpha1_uint32, where offset determines 
   the bit number [0,7] of the alpha channel starting from the LSB
   The given RGB value will be averaged against any of the existing elements with alpha=1
   The A value will be applied a threshold at 0.5 */
VOXOWL_HOST_AND_DEVICE 
void 
packRGBA_RGB24_8ALPHA1_UINT32( uint32_t *rgb24_8alpha1, int offset, glm::vec4 rgba ) {
    uint8_t lower_bits = (uint8_t)*rgb24_8alpha1;
    packBIT_UINT8( &lower_bits, offset, rgba.a >= 0.5 ); 
    *rgb24_8alpha1 =(uint8_t)lower_bits | 
        ((uint32_t)(rgba.r * 255.f) << 24) |
        ((uint32_t)(rgba.g * 255.f) << 16) |
        ((uint32_t)(rgba.b * 255.f) << 8);
    
}

/* Unpack a RGBA-4float from a rgb24_8alpha_uint32, where offset determines the bit number [0,7] of the alpha channel */
VOXOWL_HOST_AND_DEVICE 
glm::vec4 
unpackRGBA_RGB24_8ALPHA1_UINT32( uint32_t rgb24_8alpha1, int offset ) {
    return glm::vec4 (
        (float)((rgb24_8alpha1 >> 24) & 0xff) / 255.f,
        (float)((rgb24_8alpha1 >> 16) & 0xff) / 255.f,
        (float)((rgb24_8alpha1 >> 8) & 0xff) / 255.f,
        (float)unpackBIT_UINT8( rgb24_8alpha1, offset ) );
}

/* Pack one RGB-3float and a alpha bitmap [0,7] into on uint32 using rgb24_8alpha1 encoding */
VOXOWL_HOST_AND_DEVICE 
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
VOXOWL_HOST_AND_DEVICE 
glm::vec3 
unpackRGB24_8ALPHA1_UINT32( bool alpha[8], uint32_t rgb24_8alpha1 ) {
    uint8_t lower_bits = rgb24_8alpha1;
    unpackBITMAP_UINT8( alpha, lower_bits );
    return glm::vec3 (
        (float)((rgb24_8alpha1 >> 24) & 0xff) / 255.f,
        (float)((rgb24_8alpha1 >> 16) & 0xff) / 255.f,
        (float)((rgb24_8alpha1 >> 8) & 0xff) / 255.f );
}

/* Pack a RGBA-4float into a rgb24a1. The RGB components are packed in 8 bpc in the MSB's
   A threshold is applied to the alpha value ( >= 0.5 ) and is packed into the 7th bit. 
   Bits 0-6 (from LSB) are untouched */
VOXOWL_HOST_AND_DEVICE 
void 
packRGB24A1_UINT32( uint32_t *dst, glm::vec4 rgba ) {
    // Clear bits [7-31]
    *dst &= 0x7F;
    *dst = *dst | 
        ((uint32_t)(rgba.r * 255) << 24) |
        ((uint32_t)(rgba.g * 255) << 16) |
        ((uint32_t)(rgba.b * 255) << 8) |
        (uint32_t)0x80 * (int)(rgba.a >= .5f);
}

/* Unpack a RGBA-4float from a rgb24a1 type. Bit 7 contains the one-bit alpha. Bits 0-6 (from LSB) are untoched. */
VOXOWL_HOST_AND_DEVICE 
glm::vec4 
unpackRGB24A1_UINT32( uint32_t rgb24a1 ) {
    return glm::vec4(
        (float)((rgb24a1 >> 24) & 0xff) / 255.f,
        (float)((rgb24a1 >> 16) & 0xff) / 255.f,
        (float)((rgb24a1 >> 8) & 0xff) / 255.f,
        (float)((rgb24a1 & 0x80) != 0) );
}

/*
 * Misc functions
 */


/* Calculate the perceived intensity of an RGBA quadruple. Values are assumed to be linear */
VOXOWL_HOST_AND_DEVICE 
float 
intensityRGBA_linear( glm::vec4 rgba ) {
    // We use CIE 1931 weights and alpha as absolute weight using the following formula
    // Y = A( 0.2126R + 0.7152G + 0.0722B )
    return ( rgba.a * ( 0.2126f * rgba.r + 0.7152f * rgba.g + 0.0722 * rgba.b ) );
}

/* Calculate the perceived intensity of an RGBA quadruple. Values are assumed to be linear */
VOXOWL_HOST_AND_DEVICE 
uint8_t 
intensityRGBA_UINT32_linear( uint32_t rgba ) {
    return intensityRGBA_linear( unpackRGBA_UINT32( rgba ) ) / 255.f;
}

/* Calculate the perceived intensity of an RGBA quadruple by fast, inaccurate conversion. Values are assumed to be linear */
VOXOWL_HOST_AND_DEVICE 
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
VOXOWL_HOST_AND_DEVICE 
bool 
thresholdRGBA_linear( glm::vec4 rgba, float threshold ) {
    return intensityRGBA_linear( rgba ) >= threshold;
}

/* Calculate if a RGBA quadruple is 'black' (false) or 'white' depending on the given treshold [0,1] */
VOXOWL_HOST_AND_DEVICE 
bool 
thresholdRGBA_UINT32_linear( uint32_t rgba, float threshold ) {
    return intensityRGBA_UINT32_linear( rgba ) >= (255.f * threshold);
}

/* Calculate if a RGBA quadruple is 'black' (false) or 'white' using fast, inaccurate conversion, depending on the given treshold [0,255] */
VOXOWL_HOST_AND_DEVICE 
bool 
thresholdRGBA_UINT32_fastlinear( uint32_t rgba, uint8_t threshold  ) {
    return intensityRGBA_UINT32_fastlinear( rgba ) >= threshold;
}

VOXOWL_HOST_AND_DEVICE
glm::vec4
blendF2B( glm::vec4 src, glm::vec4 dst ) {
    glm::vec4 c;
    c.r = dst.a*(src.r * src.a) + dst.r;
    c.g = dst.a*(src.g * src.a) + dst.g;
    c.b = dst.a*(src.b * src.a) + dst.b;
    c.a = (1.f - src.a) * dst.a;
    return c;
}

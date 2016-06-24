#pragma once
#define GLM_FORCE_CXX98

/* This file describes the elemtary data type of the voxel and voxelmap
   A voxelmap is (in this context) defined as a contiguous three-dimensional array of voxels */

#include "platform.h"
#include <inttypes.h>
#include "glm/mat4x4.hpp"
#include "glm/vec3.hpp"

typedef enum {
    /* Widest possible voxel type, RGBA-8888 stored as an unsigned 32 bits integer */
    VOXEL_RGBA_UINT32,
    /* One-dimensional voxel type, 0x0 is transparent, stored as one 8 bits integer */
    VOXEL_INTENSITY_UINT8,
    /* Binary voxel type, block of 8 voxels stored in one uint8 in the z-direction. Z-size will be rounded to multiple of 8*/
    VOXEL_BITMAP_UINT8,
    /* Stores 8 voxels with a shared RGB value in the 24 most significant bits 
       and 1 bit alpha for each voxel in the lower bits. 4 bytes blocksize */
    VOXEL_RGB24_8ALPHA1_UINT32,
    /* Stores RGB values in the 24 most significant bits and a one bit alpha at bit 7. The lower 7 bits are untouched. */
    VOXEL_RGB24A1_UINT32
}
 voxel_format_t;

typedef struct {
    uint16_t x;
    uint16_t y;
    uint16_t z;
} ivec3_16_t;

VOXOWL_HOST_AND_DEVICE inline ivec3_16_t ivec3_16( uint16_t x, uint16_t y, uint16_t z ) { ivec3_16_t v = { x, y, z }; return v; }
VOXOWL_HOST_AND_DEVICE inline ivec3_16_t ivec3_16( uint16_t n ) { ivec3_16_t v = { n, n, n }; return v; }
VOXOWL_HOST_AND_DEVICE inline ivec3_16_t ivec3_16( glm::ivec3 gv ) { ivec3_16_t v = { (uint16_t)gv.x, (uint16_t)gv.y, (uint16_t)gv.z }; return v; }
VOXOWL_HOST_AND_DEVICE inline glm::ivec3 glm_ivec3_16( ivec3_16_t v ) { return glm::ivec3( v.x, v.y, v.z ); }

typedef struct {
    ivec3_16_t size;
    ivec3_16_t blocks;
    voxel_format_t format;
    void *data;
} voxelmap_t;

/* Initialize volume and allocate its buffer */
VOXOWL_HOST void voxelmapCreate( voxelmap_t *, voxel_format_t, ivec3_16_t size );
VOXOWL_HOST void voxelmapCreate( voxelmap_t *, voxel_format_t, uint32_t size_x, uint32_t size_y, uint32_t size_z );

/* Free a volume's buffer */
void voxelmapFree( voxelmap_t * );

/* Copies src to dst and checks bounds. Only voxelmaps of equal size are copied.
   If the maps have the same format, the copy is a simple call to memcpy(), otherwise
   the data is converted to the destination type first */
VOXOWL_HOST bool voxelmapSafeCopy( voxelmap_t* dst, voxelmap_t* src );

/* Return the size of one voxel in BITS */
VOXOWL_HOST_AND_DEVICE size_t bitsPerVoxel( voxel_format_t f );

/* Returns the size of the smallest accessible block in bytes */
VOXOWL_HOST_AND_DEVICE size_t bytesPerBlock( voxel_format_t f );

/* Returns the number of voxels that are stored in one elementary block */
VOXOWL_HOST_AND_DEVICE size_t voxelsPerBlock( voxel_format_t f );

/* Returns the number of blocks that are required to store a volume of size, given format f */
VOXOWL_HOST_AND_DEVICE ivec3_16_t blockCount( voxel_format_t f, ivec3_16_t size );

/* Returns the indices of the block that contains the voxel in position, given format f */
VOXOWL_HOST_AND_DEVICE ivec3_16_t blockPosition( voxel_format_t f, ivec3_16_t position );

/* Return the size of a volume's data in bytes */
VOXOWL_HOST_AND_DEVICE size_t voxelmapSize( voxelmap_t * );

/* Access an array based volume by coordinates. Returns a pointer to the block containing the element */
VOXOWL_HOST_AND_DEVICE void* voxel( voxelmap_t*, ivec3_16_t position );
VOXOWL_HOST_AND_DEVICE void* voxel( voxelmap_t*, uint32_t x, uint32_t y, uint32_t z );

/* Fills a given volume by copying a value to every position. 
   The value pointer is read for voxelSize( format ) bytes */
VOXOWL_HOST_AND_DEVICE void voxelmapFill( voxelmap_t*, void *value );

/* 
    Voxel pack/unpack functions are used to load/store channel values to different voxel formats 
*/

/* Pack an rgba-4float value into an arbitrarily formatted  voxelmap */
VOXOWL_HOST_AND_DEVICE void voxelmapPack( voxelmap_t*, ivec3_16_t position, glm::vec4 rgba );

/* Unpack an rgba-4float value from an arbitrarily formatted  voxelmap */
VOXOWL_HOST_AND_DEVICE glm::vec4 voxelmapUnpack( voxelmap_t*, ivec3_16_t position );

/* Pack an rgba-4float value to an uint32 */
VOXOWL_HOST_AND_DEVICE void packRGBA_UINT32( uint32_t* rgba, const glm::vec4& v );

/* Unpack an rgba-4float value from an uint32 */
VOXOWL_HOST_AND_DEVICE glm::vec4 unpackRGBA_UINT32( uint32_t rgba );

/* Unpack an rgba-4float value from four uint8s */
VOXOWL_HOST_AND_DEVICE glm::vec4 unpackRGBA_4UINT8( uint8_t r, uint8_t g, uint8_t b, uint8_t a );

/* Pack an grayscale intensity to an uint8 */
VOXOWL_HOST_AND_DEVICE void packINTENSITY_UINT8( uint8_t* dst, float intensity );

/* Unpack an grayscale intensity from an uint8 */
VOXOWL_HOST_AND_DEVICE float unpackINTENSITY_UINT8( uint8_t intensity );

/* Pack a bitmap (array of bool) into a single uint8 */
VOXOWL_HOST_AND_DEVICE void packBITMAP_UINT8( uint8_t* dst, bool src[8] );

/* Unpack a bitmap (array of bool) from an uint8 */
VOXOWL_HOST_AND_DEVICE void unpackBITMAP_UINT8( bool dst[8], uint8_t src );

/* Pack a single bit from a bitmap into an uint8. offset gives the offset in the uint8 where the LSB is zero */
VOXOWL_HOST_AND_DEVICE void packBIT_UINT8( uint8_t* dst, int offset, bool b );

/* Unpack a single bit from an uint8. offset gives the offset in the uint8 where the LSB is zero */
VOXOWL_HOST_AND_DEVICE bool unpackBIT_UINT8( uint8_t src, int offset );

/* Pack a RGBA-4float value into a rgb24_8alpha1_uint32, where offset determines 
   the bit number [0,7] of the alpha channel starting from the LSB
   The given RGB value will be averaged against any of the existing elements with alpha=1
   The A value will be applied a threshold at 0.5 */
VOXOWL_HOST_AND_DEVICE void packRGBA_RGB24_8ALPHA1_UINT32( uint32_t *rgb24_8alpha1, int offset, glm::vec4 rgba );

/* Unpack a RGBA-4float from a rgb24_8alpha_uint32, where offset determines the bit number [0,7] of the alpha channel */
VOXOWL_HOST_AND_DEVICE glm::vec4 unpackRGBA_RGB24_8ALPHA1_UINT32( uint32_t rgb24_8alpha1, int offset );

/* Pack one RGB-3float and a alpha bitmap [0,7] into on uint32 using rgb24_8alpha1 encoding */
VOXOWL_HOST_AND_DEVICE void packRGB24_8ALPHA1_UINT32( uint32_t *dst, glm::vec3 rgb, bool alpha[8] );

/* Unpack one RGB-3float and a alpha bitmap [0,7] from one uint32 using rgb24_8alpha1 encoding */
VOXOWL_HOST_AND_DEVICE glm::vec3 unpackRGB24_8ALPHA1_UINT32( bool alpha[8], uint32_t rgb24_8alpha1 );

/* Pack a RGBA-4float into a rgb24a1. The RGB components are packed in 8 bpc in the MSB's
   A threshold is applied to the alpha value ( >= 0.5 ) and is packed into the 7th bit. 
   Bits 0-6 (from LSB) are untouched */
VOXOWL_HOST_AND_DEVICE void packRGB24A1_UINT32( uint32_t *dst, glm::vec4 rgba );

/* Unpack a RGBA-4float from a rgb24a1 type. Bit 7 contains the one-bit alpha. Bits 0-6 (from LSB) are untoched. */
VOXOWL_HOST_AND_DEVICE glm::vec4 unpackRGB24A1_UINT32( uint32_t rgb24a1 );

/*
 * Misc functions
 */

/* Calculate the perceived intensity of an RGBA quadruple. Values are assumed to be linear */
VOXOWL_HOST_AND_DEVICE float intensityRGBA_linear( glm::vec4 rgba );

/* Calculate the perceived intensity of an RGBA quadruple. Values are assumed to be linear */
VOXOWL_HOST_AND_DEVICE uint8_t intensityRGBA_UINT32_linear( uint32_t rgba );

/* Calculate the perceived intensity of an RGBA quadruple by fast, inaccurate conversion. Values are assumed to be linear */
VOXOWL_HOST_AND_DEVICE uint8_t intensityRGBA_UINT32_fastlinear( uint32_t rgba );

/* Calculate if a RGBA quadruple is 'black' (false) or 'white' depending on the given treshold [0,1] */
VOXOWL_HOST_AND_DEVICE bool thresholdRGBA_linear( glm::vec4 rgba, float threshold=0.5 );

/* Calculate if a RGBA quadruple is 'black' (false) or 'white' depending on the given treshold [0,1] */
VOXOWL_HOST_AND_DEVICE bool thresholdRGBA_UINT32_linear( uint32_t rgba, float threshold=0.5 );

/* Calculate if a RGBA quadruple is 'black' (false) or 'white' using fast, inaccurate conversion, depending on the given treshold [0,255] */
VOXOWL_HOST_AND_DEVICE bool thresholdRGBA_UINT32_fastlinear( uint32_t rgba, uint8_t threshold =127 );

/* Return the result of 'front-to-back-blending' src (back) to dst (front) in rgba-4float */
VOXOWL_HOST_AND_DEVICE glm::vec4 blendF2B( glm::vec4 src, glm::vec4 dst );


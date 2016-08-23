#pragma once
#include "voxowl_platform.h"
#include "voxel.h"

#define SVMM_DEFAULT_BLOCKWIDTH 4
#define SVMM_MAGIC1 'V'
#define SVMM_MAGIC2 's'

#define SVMM_TERMINAL_BIT_MASK 0x10
//#define SVMM_STUB_BIT_MASK 0x20
#define SVMM_OFFSET_BITS 4
#define SVMM_OFFSET_BITS_MASK 0xf
#define SVMM_SUBBLOCK_WIDTH 2

typedef struct {
    uint64_t next;
    uint32_t mipmap_factor;
    uint32_t block_count;
    voxel_format_t format;
    uint8_t blockwidth;

} svmm_level_header_t;

typedef struct {
    uint8_t magic1;
    uint8_t magic2;
    ivec3_32_t volume_size;
    voxel_format_t format;
    uint64_t data_start;
    uint64_t data_length;
    uint8_t levels;
    uint8_t blockwidth;
    ivec3_32_t rootsize;
    uint64_t root;

} svmm_header_t;

typedef struct {
    svmm_header_t header;
    char *data_ptr;
    size_t data_size;
    int fd;
    bool is_mmapped;

} svmipmap_t;

typedef struct {
    unsigned int blockwidth;
    unsigned int rootwidth;
    float delta;
    bool bitmapBaselevel;
    voxel_format_t format;
} svmm_encode_opts_t;

// Encode functions

VOXOWL_HOST ssize_t svmmEncode( svmipmap_t*,  voxelmap_t* uncompressed, svmm_encode_opts_t opts );
VOXOWL_HOST ssize_t svmmEncode( svmipmap_t*,  voxelmap_t* uncompressed, int quality );
VOXOWL_HOST ssize_t svmmEncodeFile( const char* filename, voxelmap_t* uncompressed, svmm_encode_opts_t opts );

/*! Attempts to set optimal settings for a given voxelmap automatically
   The quality parameter ranges 1-100 */
VOXOWL_HOST void svmmSetOpts( svmm_encode_opts_t *opts, 
                              voxelmap_t* uncompressed, 
                              int quality );

VOXOWL_HOST int svmmFree( svmipmap_t* );

// Decode functions

/*! Open the file at `filename`, consisting of a svmm header and a data section.
 * The returned svmipmap object can be used to decode its contents through memory-mapped i/o */
VOXOWL_HOST int svmmOpenMapped( svmipmap_t*, const char *filename );

VOXOWL_HOST void svmmReadHeader( svmm_header_t*, void *buffer );
VOXOWL_HOST void svmmRead( svmipmap_t*, void *buffer );

VOXOWL_HOST glm::vec4 svmmDecodeVoxel( svmipmap_t* m, ivec3_32_t position );

VOXOWL_HOST int svmmDecode( voxelmap_t* v, svmipmap_t* svmm );
VOXOWL_HOST bool svmmTest( voxelmap_t* uncompressed, int quality );

// Utility functions

VOXOWL_HOST_AND_DEVICE bool isTerminal( uint8_t lower_bits );
VOXOWL_HOST_AND_DEVICE void setTerminal( uint8_t* lower_bits, bool terminal );
//VOXOWL_HOST_AND_DEVICE bool isStub( uint32_t rgb24a1 );
//VOXOWL_HOST_AND_DEVICE void setStub( uint32_t *rgb24a1, bool stub );

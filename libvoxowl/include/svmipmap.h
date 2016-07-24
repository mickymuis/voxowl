#pragma once
#include "voxowl_platform.h"
#include "voxel.h"

#define SVMM_DEFAULT_BLOCKWIDTH 4
#define SVMM_MAGIC1 'V'
#define SVMM_MAGIC2 's'

#define SVMM_TERMINAL_BIT_MASK 0x40
#define SVMM_STUB_BIT_MASK 0x20
#define SVMM_OFFSET_BITS 5
#define SVMM_OFFSET_BITS_MASK 0x1f
#define SVMM_SUBBLOCK_WIDTH 2

typedef struct {
    uint32_t next;
    uint32_t mipmap_factor;
    voxel_format_t format;
    uint8_t blockwidth;

} svmm_level_header_t;

typedef struct {
    uint8_t magic1;
    uint8_t magic2;
    ivec3_16_t volume_size;
    voxel_format_t format;
    uint64_t data_start;
    uint64_t data_length;
    uint8_t levels;
    uint8_t blockwidth;
    ivec3_16_t rootsize;
    uint64_t root;

} svmm_header_t;

typedef struct {
    svmm_header_t header;
    void *buffer;

} svmipmap_t;

/*typedef enum {
    SVMM_OCTREE =2,
    SVMM_BLOCK64 =4
} svmm_blockmode_t;*/

typedef struct {
    unsigned int blockwidth;
    unsigned int rootwidth;
    float delta;
    bool bitmapBaselevel;
    voxel_format_t format;
} svmm_encode_opts_t;

// Encode functions

VOXOWL_HOST int svmmEncode( svmipmap_t*,  voxelmap_t* uncompressed, svmm_encode_opts_t opts );
VOXOWL_HOST int svmmEncode( svmipmap_t*,  voxelmap_t* uncompressed, int quality );

/* Attempts to set optimal settings for a given voxelmap automatically
   The quality parameter ranges 1-100 */
VOXOWL_HOST void svmmSetOpts( svmm_encode_opts_t *opts, 
                              voxelmap_t* uncompressed, 
                              int quality );

VOXOWL_HOST void svmmFree( svmipmap_t* );

// Decode functions

VOXOWL_HOST void svmmReadHeader( svmm_header_t*, void *buffer );
VOXOWL_HOST void svmmRead( svmipmap_t*, void *buffer );

VOXOWL_HOST glm::vec4 svmmDecodeVoxel( svmipmap_t* m, ivec3_16_t position );

VOXOWL_HOST bool svmmTest( voxelmap_t* uncompressed, int quality );

// Utility functions

VOXOWL_HOST bool isTerminal( uint32_t rgb24a1 );
VOXOWL_HOST void setTerminal( uint32_t* rgb24a1, bool terminal );
VOXOWL_HOST bool isStub( uint32_t rgb24a1 );
VOXOWL_HOST void setStub( uint32_t *rgb24a1, bool stub );

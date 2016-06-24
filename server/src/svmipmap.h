#pragma once
#include "platform.h"
#include "voxel.h"

#define SVMM_DEFAULT_BLOCKWIDTH 4
#define SVMM_MAGIC1 'S'
#define SVMM_MAGIC2 'v'

typedef struct {
    uint32_t next;
    ivec3_16_t mipmap_size;

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
    voxel_format_t format;
} svmm_encode_opts_t;

VOXOWL_HOST int svmmEncode( svmipmap_t*,  voxelmap_t* uncompressed, svmm_encode_opts_t opts );
VOXOWL_HOST int svmmEncode( svmipmap_t*,  voxelmap_t* uncompressed );

VOXOWL_HOST void svmmFree( svmipmap_t* );

VOXOWL_HOST void svmmReadHeader( svmm_header_t*, void *buffer );
VOXOWL_HOST void svmmRead( svmipmap_t*, void *buffer );

VOXOWL_HOST glm::vec4 svmmDecodeVoxel( svmipmap_t* m, ivec3_16_t position );

VOXOWL_HOST bool svmmTest( voxelmap_t* uncompressed );

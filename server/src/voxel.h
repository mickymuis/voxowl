#pragma once

/* This file describes the elemtary data type of the voxel and voxelmap
   A voxelmap is (in this context) defined as a contiguous three-dimensional array of voxels */

#include "platform.h"
#include <inttypes.h>
#include "glm/mat4x4.hpp"
#include "glm/vec3.hpp"

typedef enum {
    /* Widest possible voxel type, RGBA stored as an unsigned 32 bits integer */
    VOXEL_RGBA_UINT32
} voxel_format_t;

typedef struct {
    glm::ivec3 size;
    voxel_format_t format;
    void *data;
} voxelmap_t;

/* Initialize volume and allocate its buffer */
void voxelmapCreate( voxelmap_t *, voxel_format_t, glm::ivec3 size );
void voxelmapCreate( voxelmap_t *, voxel_format_t, uint32_t size_x, uint32_t size_y, uint32_t size_z );

/* Free a volume's buffer */
void voxelmapFree( voxelmap_t * );

/* Return the size of one voxel in bytes */
VOXOWL_HOST_AND_DEVICE size_t voxelSize( voxel_format_t );

/* Return the size of a volume's data in bytes */
VOXOWL_HOST_AND_DEVICE size_t voxelmapSize( voxelmap_t * );

/* Access an array based volume by coordinates. Returns a pointer to the first element */
VOXOWL_HOST_AND_DEVICE void* voxel( voxelmap_t*, glm::ivec3 position );
VOXOWL_HOST_AND_DEVICE void* voxel( voxelmap_t*, uint32_t x, uint32_t y, uint32_t z );

/* Fills a given volume by copying a value to every position. 
   The value pointer is read for voxelSize( format ) bytes */
VOXOWL_HOST_AND_DEVICE void voxelmapFill( voxelmap_t*, void *value );

/* 
    Voxel pack/unpack functions are used to load/store channel values to different voxel formats 
*/

/* Pack an rgba-4float value into an arbitrarily formatted  voxelmap */
VOXOWL_HOST_AND_DEVICE void voxelmapPack( voxelmap_t*, glm::ivec3 position, glm::vec4 rgba );

/* Unpack an rgba-4float value from an arbitrarily formatted  voxelmap */
VOXOWL_HOST_AND_DEVICE glm::vec4 voxelmapUnpack( voxelmap_t*, glm::ivec3 position );

/* Pack an rgba-4float value to an uint32 */
VOXOWL_HOST_AND_DEVICE void packRGBA_UINT32( uint32_t* rgba, const glm::vec4& v );

/* Unpack an rgba-4float value from an uint32 */
VOXOWL_HOST_AND_DEVICE glm::vec4 unpackRGBA_UINT32( uint32_t rgba );

/* Return the result of 'front-to-back-blending' src (back) to dst (front) in rgba-4float */
VOXOWL_HOST_AND_DEVICE glm::vec4 blendF2B( glm::vec4 src, glm::vec4 dst );


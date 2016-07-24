#pragma once
#include <voxowl_platform.h>
#include "voxel.h"

/* voxelmap.h
 *
 * Implements reading and writing of memory-mapped voxelmaps 
 * */

#define VOXELMAP_MAGIC1 'V'
#define VOXELMAP_MAGIC2 'm'

typedef struct {
    uint8_t             magic1;
    uint8_t             magic2;
    uint8_t             version;
    ivec3_16_t          volume_size;
    ivec3_16_t          block_count;
    ivec3_16_t          scale;
    voxel_format_t      format;
    uint64_t            data_start;
    uint64_t            data_length;
} voxelmap_header_t;

typedef struct {
    caddr_t mapped_addr;
    size_t mapped_length;
    int fd;
    voxelmap_header_t header;
} voxelmap_mapped_t;

/* mmap() an existing voxelmap file consisting of a header and a data section 
   'vm' contains the address of the mapped memory region and file descriptor */
int voxelmapOpenMapped( voxelmap_mapped_t *vm, const char *filename );
int voxelmapCreateMapped( voxelmap_mapped_t *vm, 
                          const char* filename, 
                          voxel_format_t format, 
                          ivec3_16_t size, 
                          ivec3_16_t scale =ivec3_16(1) );

void voxelmapFromMapped( voxelmap_t *, voxelmap_mapped_t *mapped );

void voxelmapUnmap( voxelmap_mapped_t* );

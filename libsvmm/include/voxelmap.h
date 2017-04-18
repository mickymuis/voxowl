#pragma once
#include "platform.h"
#include "voxel.h"

/** Implements reading and writing of memory-mapped voxelmaps.
  * The functions `voxelmapOpenMapped()` and `voxelmapCreateMapped()`
  * can be used to open or create a memory mapped voxelmap on disk.
  * To access the data using the voxelmap functions provided in `voxel.h`,
  * `voxelmapFromMapped()` is utilized to generate a voxelmap object
  * that is linked to the mapped memory region.
  */

#define VOXELMAP_MAGIC1 'V'
#define VOXELMAP_MAGIC2 'm'

typedef struct {
    uint8_t             magic1;
    uint8_t             magic2;
    uint8_t             version;
    ivec3_32_t          volume_size;
    ivec3_32_t          block_count;
    ivec3_32_t          scale;
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

/*! Memory-map an existing voxelmap file on disk, consisting of a header and a data section.
   `vm` is populated with the address of the mapped memory region and file descriptor 
    Returns 0 if the operation was succesful */
int voxelmapOpenMapped( voxelmap_mapped_t *vm, const char *filename );

/*! Creates a new memory mapped voxelmap on disk.
    The parameters of this function are identical to `voxelmapCreate()`, with
    the exception of `filename` */
int voxelmapCreateMapped( voxelmap_mapped_t *vm, 
                          const char* filename, 
                          voxel_format_t format, 
                          ivec3_32_t size, 
                          ivec3_32_t scale =ivec3_32(1) );

/*! Populates a regular voxelmap object from a memory mapped reference. 
   Multiple object can be made and all will be invalid after `voxemapUnmap()` is called. */
void voxelmapFromMapped( voxelmap_t *, voxelmap_mapped_t *mapped );

/*! Unmap the memory from a memory mapped voxelmap and close its file */
int voxelmapUnmap( voxelmap_mapped_t* );


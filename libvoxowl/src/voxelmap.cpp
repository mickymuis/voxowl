#include "voxelmap.h"

#include <string.h>
#include <stdio.h>
// Unix low-level IO functions
#include <unistd.h>
#include <sys/mman.h>
#include <sys/types.h>
#include <fcntl.h>

/*! Memory-map an existing voxelmap file on disk, consisting of a header and a data section.
    `vm` is populated with the address of the mapped memory region and file descriptor 
    Returns 0 if the operation was succesful */
int 
voxelmapOpenMapped( voxelmap_mapped_t* vm, const char* filename ) {
    int fd =open( filename, O_RDWR );
    if( fd == -1 )
        return -1;

    if( read( fd, (void*)&vm->header, sizeof( voxelmap_header_t ) ) == -1 ) {
        close( fd );
        return -1;
    }

    if( vm->header.magic1 != VOXELMAP_MAGIC1 || 
        vm->header.magic2 != VOXELMAP_MAGIC2 ) {
        close( fd );
        return -1;
    }

    lseek( fd, 0, SEEK_SET );
    size_t len =vm->header.data_start + vm->header.data_length;

    vm->mapped_addr =(caddr_t)mmap( (caddr_t)0, len, PROT_READ|PROT_WRITE, MAP_SHARED, fd, 0 );

    if( vm->mapped_addr == (caddr_t)-1 ) {
        close( fd );
        return -1;
    }

    vm->fd =fd;
    vm->mapped_length =len;
    return 0;
}

/*! Creates a new memory mapped voxelmap on disk.
    The parameters of this function are identical to `voxelmapCreate()`, with
    the exception of `filename` */
int 
voxelmapCreateMapped( voxelmap_mapped_t* vm, 
                      const char* filename, 
                      voxel_format_t format, 
                      ivec3_32_t size, 
                      ivec3_32_t scale ) {
    int fd =open( filename, O_CREAT | O_TRUNC | O_RDWR, S_IRUSR | S_IWUSR );
    if( fd == -1 )
        return -1;

    // Prepare a header
    voxelmap_header_t header;
    memset( &header, 0, sizeof( voxelmap_header_t ) );

    header.magic1 =VOXELMAP_MAGIC1;
    header.magic2 =VOXELMAP_MAGIC2;
    header.volume_size =size;
    header.scale =scale;
    header.format =format;
    header.block_count =blockCount( format, size );
    header.data_length =voxelmapSize( format, size, scale );
    header.data_start =sizeof( voxelmap_header_t ); // data starts directly after header

    size_t len =header.data_start + header.data_length;

    // Create the memory map
    /*lseek( fd, len+1, SEEK_SET );
    write( fd, "", 1 );
    lseek( fd, 0, SEEK_SET );*/
    ftruncate( fd, len );
    vm->mapped_addr =(caddr_t)mmap( (caddr_t)0, len, PROT_READ|PROT_WRITE, MAP_SHARED, fd, 0 );

    if( vm->mapped_addr == (caddr_t)-1 ) {
        close( fd );
        return -1;
    }

    // Success, populate the object and the file
    
    vm->header =header;
    vm->fd =fd;
    vm->mapped_length =len;

    memcpy( (char*)vm->mapped_addr, (char*)&header, sizeof( voxelmap_header_t ) );

    return 0;
}

/*! Populates a regular voxelmap object from a memory mapped reference. 
   Multiple object can be made and all will be invalid after `voxemapUnmap()` is called. */
void 
voxelmapFromMapped( voxelmap_t* v, voxelmap_mapped_t *vm ) {
    char *data_ptr =(char*)vm->mapped_addr;
    data_ptr +=vm->header.data_start;

    v->data =data_ptr;
    v->size =vm->header.volume_size;
    v->format =vm->header.format;
    v->blocks =vm->header.block_count;
    v->scale =vm->header.scale;
}

/*! Unmap the memory from a memory mapped voxelmap and close its file */
int 
voxelmapUnmap( voxelmap_mapped_t* vm ) {
    if( msync( vm->mapped_addr, vm->mapped_length, MS_SYNC ) == -1 )
        perror( "" );
    if( munmap( vm->mapped_addr, vm->mapped_length ) == -1 ) {
        perror( "" );
       return -1;
    }

   close( vm->fd );
   return 0;
}

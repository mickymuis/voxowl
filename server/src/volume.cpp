#include "volume.h"

Volume::Volume( const char *name, Object* parent ) 
    : Actor( name, parent ){}

Volume::~Volume() {}

//

VolumeVoxelmap::VolumeVoxelmap( const char *name, Object* parent )
    : Volume( name, parent ), voxelmap( 0 ) {}
VolumeVoxelmap::~VolumeVoxelmap() {}

void 
VolumeVoxelmap::setVoxelmap( voxelmap_t *v ) {
    voxelmap =v;
    newConfiguration();
}

voxelmap_t* 
VolumeVoxelmap::getVoxelmap() {
    return voxelmap;
}

void* 
VolumeVoxelmap::data_ptr() {
    return (void*)voxelmap;
}

ivec3_32_t 
VolumeVoxelmap::size() const {
    if( voxelmap )
        return voxelmap->size;
    return ivec3_32(0);
}

//

VolumeSVMipmap::VolumeSVMipmap( const char *name, Object* parent )
    : Volume( name, parent ), svmm( 0 ) {}
VolumeSVMipmap::~VolumeSVMipmap() {}

void 
VolumeSVMipmap::setSVMipmap( svmipmap_t *s ) {
    svmm =s;
    newConfiguration();
}

svmipmap_t* 
VolumeSVMipmap::getSVMipmap() {
    return svmm;
}

void* 
VolumeSVMipmap::data_ptr() {
    return (void*)svmm;
}

ivec3_32_t 
VolumeSVMipmap::size() const {
    if( svmm )
        return svmm->header.volume_size;
    return ivec3_32(0);
}

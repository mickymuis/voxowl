#pragma once

#include "volume.h"
#include <voxel.h>
#include <svmipmap.h>

class VolumeStorageDetail {
    public:
        virtual void* data_ptr() =0;
        virtual ivec3_32_t size() const =0;

};

class VolumeVoxelmapStorageDetail : public VolumeStorageDetail {
    public:
        inline VolumeVoxelmapStorageDetail() : voxelmap(0) {}
        inline void setVoxelmap( voxelmap_t* v ) { voxelmap =v; }
        inline voxelmap_t* getVoxelmap() { return voxelmap; }
        
        inline void* data_ptr();
        inline ivec3_32_t size() const;

    protected:
        voxelmap_t* voxelmap;

};

class VolumeSVMipmapStorageDetail : public VolumeStorageDetail {
    public:
        inline VolumeSVMipmapStorageDetail() : svmm(0) {}
        inline void setSVMipmap( svmipmap_t *v ) { svmm =v; }
        inline svmipmap_t* getSVMipmap() { return svmm; }
        
        inline void* data_ptr();
        inline ivec3_32_t size() const;

    protected:
        svmipmap_t* svmm;

};

class VolumeVoxelmap : public Volume {
    public:
        inline VolumeVoxelmap( const char *name, Object* parent =0 );
        inline ~VolumeVoxelmap();

        inline VolumeStorageDetail* storageDetail() { return &m_detail; }
        inline void setVoxelmap( voxelmap_t * );
        
        inline ivec3_32_t size() const { return m_detail.size(); }
    
    private:
        VolumeVoxelmapStorageDetail m_detail;
};

class VolumeSVMipmap : public Volume {
    public:
        inline VolumeSVMipmap( const char *name, Object* parent =0 );
        inline ~VolumeSVMipmap();

        inline VolumeStorageDetail* storageDetail() { return &m_detail; } 
        inline void setSVMipmap( svmipmap_t * );
        
        inline ivec3_32_t size() const { return m_detail.size(); }

    private:
        VolumeSVMipmapStorageDetail m_detail;

};

// Implementation of inline functions

VolumeVoxelmap::VolumeVoxelmap( const char *name, Object* parent )
    : Volume( name, parent ) {}
VolumeVoxelmap::~VolumeVoxelmap() {}

void 
VolumeVoxelmap::setVoxelmap( voxelmap_t *v ) {
    m_detail.setVoxelmap( v );
    newConfiguration();
}

void* 
VolumeVoxelmapStorageDetail::data_ptr() {
    return (void*)voxelmap;
}

ivec3_32_t 
VolumeVoxelmapStorageDetail::size() const {
    if( voxelmap )
        return voxelmap->size;
    return ivec3_32(0);
}

//

VolumeSVMipmap::VolumeSVMipmap( const char *name, Object* parent )
    : Volume( name, parent ) {}
VolumeSVMipmap::~VolumeSVMipmap() {}

void 
VolumeSVMipmap::setSVMipmap( svmipmap_t *s ) {
    m_detail.setSVMipmap( s );
    newConfiguration();
}

void* 
VolumeSVMipmapStorageDetail::data_ptr() {
    return (void*)svmm;
}

ivec3_32_t 
VolumeSVMipmapStorageDetail::size() const {
    if( svmm )
        return svmm->header.volume_size;
    return ivec3_32(0);
};

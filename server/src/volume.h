#pragma once

#include "actor.h"
#include <voxel.h>
#include <svmipmap.h>

class Volume : public Actor {
    public:
        Volume( const char *name, Object* parent =0 );
        ~Volume();

        virtual void* data_ptr() =0;
        virtual ivec3_32_t size() const =0;

};

class VolumeVoxelmap : public Volume {
    public:
        VolumeVoxelmap( const char *name, Object* parent =0 );
        ~VolumeVoxelmap();

        void setVoxelmap( voxelmap_t * );
        voxelmap_t* getVoxelmap();

        void* data_ptr();
        ivec3_32_t size() const;

    protected:
        voxelmap_t* voxelmap;
};

class VolumeSVMipmap : public Volume {
    public:
        VolumeSVMipmap( const char *name, Object* parent =0 );
        ~VolumeSVMipmap();

        void setSVMipmap( svmipmap_t * );
        svmipmap_t* getSVMipmap();

        void* data_ptr();
        ivec3_32_t size() const;

    protected:
        svmipmap_t* svmm;
};

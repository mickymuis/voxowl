#pragma once

#include "actor.h"
#include "voxel.h"

class Volume : public Actor {
    public:
        Volume( const char *name, Object* parent =0 );
        ~Volume();

        // TODO: needs to be restructured 

        virtual voxelmap_t& data() =0;

/*        uint32_t sizeX() const { return size_x; }
        uint32_t sizeY() const { return size_y; }
        uint32_t sizeZ() const { return size_z; }

        void *data() { return data_ptr; }
        size_t dataSize() { return data_size; }

        int format() { return data_format; }
    
    protected:
        uint32_t size_x, size_y, size_z;
        void *data_ptr;
        size_t data_size;
        int data_format;*/

};

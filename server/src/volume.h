#pragma once

#include "actor.h"
#include <ivec3_32.h>

class VolumeStorageDetail;

class Volume : public Actor {
    public:

        inline Volume( const char *name, Object* parent =0 );
        inline ~Volume();

        virtual VolumeStorageDetail* storageDetail() =0;

        virtual ivec3_32_t size() const =0;
};

//

Volume::Volume( const char *name, Object* parent ) 
    : Actor( name, parent ){}

Volume::~Volume() {}


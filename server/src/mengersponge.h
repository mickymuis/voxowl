#pragma once

#include "volume.h"
#include "volume_detail.h"

class MengerSponge : public VolumeVoxelmap {
    public:
        enum MODE {
            COLORS_RGBA
        };
        MengerSponge( const char *name, Object *parent );
        ~MengerSponge();

        void setDepth( int );
        void setMode( int );

    private:
        void makeSponge();

        int depth;
        int mode;

        voxelmap_t volume;
        svmipmap_t volume_svmm;
};

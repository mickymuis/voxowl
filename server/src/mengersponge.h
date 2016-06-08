#pragma once
#define GLM_FORCE_CXX98

#include "volume.h"

class MengerSponge : public Volume {
    public:
        enum MODE {
            COLORS_RGBA
        };
        MengerSponge( const char *name, Object *parent );
        ~MengerSponge();

        void setDepth( int );
        void setMode( int );

        voxelmap_t& data() { return volume; }

    private:
        void makeSponge();

        int depth;
        int mode;

        voxelmap_t volume;
};

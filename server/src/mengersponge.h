#pragma once
//#define GLM_FORCE_CXX98

#include "volume.h"

class MengerSponge : public VolumeSVMipmap {
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

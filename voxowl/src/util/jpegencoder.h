#pragma once

#include "../core/platform.h"
#ifdef VOXOWL_USE_TURBOJPEG

/**
 * Implements a simple JPEG encoder using libturbojpeg 
 */
class JPEGEncoder {
    public:
        JPEGEncoder( int quality =75, int channels =3 );
        ~JPEGEncoder();

        void setQuality( int q ) { quality =q; }
        int getQuality() const { return quality; }

        void setChannels( int c ) { channels =c; }
        int getChannels() const { return channels; }

        void setSize( unsigned int w, unsigned int h ) { width =w; height =h; }
        unsigned int getWidth() const { return width; }
        unsigned int getHeight() const { return height; }

        bool encode( unsigned char* uncompressed );
        
        const unsigned char* compressed() const { return jpeg; }
        long unsigned int size() const { return jpegSize; }

    private:
        bool jpegEncode( unsigned char* data );
        void jpegCleanup();

        unsigned char* jpeg;
        long unsigned int jpegSize, internalSize;
        int quality;
        int channels;
        unsigned int width, height;
};

#endif

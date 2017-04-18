
#include "jpegencoder.h"

#ifdef VOXOWL_USE_TURBOJPEG
#include <turbojpeg.h>
#include <iostream>

JPEGEncoder::JPEGEncoder( int quality, int channels )
    : jpeg( 0 ), jpegSize( 0 ), internalSize( 0 ), quality( quality ), channels( channels ) {}

JPEGEncoder::~JPEGEncoder() {
    jpegCleanup();
}

bool
JPEGEncoder::encode( unsigned char* uncompressed ) {
    if( !jpegEncode( uncompressed ) ) {
        std::cerr << "JPEGEncoder::encode(): " << tjGetErrorStr() << std::endl;
        return false;
    }
    return true;
}

/* The following code is based on a snippet posted on Stackoverflow by user 'Theolodis' */

bool
JPEGEncoder::jpegEncode( unsigned char* data ) {
    long unsigned int size = jpegSize;
    int err =0;

    tjhandle jpegCompressor = tjInitCompress();

    err =tjCompress2(jpegCompressor, data, width, 0, height, TJPF_RGB,
                      &jpeg, &size, TJSAMP_444, quality,
                                TJFLAG_FASTDCT);

    tjDestroy(jpegCompressor);

    internalSize = internalSize >= size ? internalSize : size;
    jpegSize = size;

    return err == 0;
}

void 
JPEGEncoder::jpegCleanup() {
    if( jpeg ) {
        tjFree( jpeg );
        internalSize =jpegSize =0;
    }
}

#endif

#include "framebuffer.h"
#include <cstdlib>
#include <fstream>
#include <stdexcept>

//
// Begin of Framebuffer implementation
// 


Framebuffer::Framebuffer( const char* name, Object* parent )
    : RenderTarget( name, parent ), buffer( nullptr ) {
}

Framebuffer::~Framebuffer() {
    destroyBuffer( buffer );
}

void* 
Framebuffer::allocateBuffer( size_t framesize ) {
    // framesize is already in bytes...
    void *buf =malloc( framesize );
    if( !buf ) throw std::runtime_error( "Framebuffer: could not allocate a buffer" );
    return buf;
}

void 
Framebuffer::destroyBuffer( void* buf ) {
    // in this particular case, free() will suffice
    free( buf );
}

size_t
Framebuffer::calculateFrameSize() const {
    return width*height*bytesPerPixel();
}

void 
Framebuffer::reinitialize() {

    size_t s =calculateFrameSize();
    
    if( buffer != nullptr ) {
        if( s == frame_size )
            return;
        else
            destroyBuffer( buffer );
    }

    buffer =allocateBuffer( s );
    frame_size =s;
}


bool 
Framebuffer::synchronize() {
    // Framebuffer is a raw buffer in memory, nothing to synchronize
    return true;
}

// 
// Begin of Compressed Framebuffer implementation
//

CompressedFramebuffer::CompressedFramebuffer( const char* name, Object* parent ) 
    : Framebuffer( name, parent ) {
        setMode( PIXMAP );
}

CompressedFramebuffer::~CompressedFramebuffer() {}

void 
CompressedFramebuffer::setMode( Mode m ) {
    // Fore now...
    mode =m;
}

void 
CompressedFramebuffer::reinitialize() {
    Framebuffer::reinitialize();
}

bool 
CompressedFramebuffer::synchronize() {
    Framebuffer::synchronize();
    
    if( channels() != 3 )
            return setError( true, "Unsupported pixel format" );
    
    switch( mode ) {
#ifdef VOXOWL_USE_TURBOJPEG
        case JPEG:
            jpeg.setChannels( channels() );
            jpeg.setSize( getWidth(), getHeight() );
            if( !jpeg.encode( (unsigned char*)getBuffer() ) )
                return setError( true, "JPEG-encoder failed" );
            break;
#endif
        default:
            break;
    }
    return true;
}

const void* 
CompressedFramebuffer::getCompressedBuffer() {
    switch( mode ) {
#ifdef VOXOWL_USE_TURBOJPEG
        case JPEG:
            return jpeg.compressed();
            break;
#endif
        default:
            break;
    }
    return getBuffer();
}

size_t 
CompressedFramebuffer::compressedBufferSize() {
    switch( mode ) {
#ifdef VOXOWL_USE_TURBOJPEG
        case JPEG:
            return jpeg.size();
            break;
#endif
        default:
            break;
    }
    return getFrameSize();

}



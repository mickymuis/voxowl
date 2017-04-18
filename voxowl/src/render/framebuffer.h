#pragma once

//#include <mutex>
#include <inttypes.h>
#include <string>

#include "../core/platform.h"
#include "../core/object.h"
#include "../core/voxowl.h"
#include "rendertarget.h"
#include "../util/jpegencoder.h"

/**
 * Framebuffer implements an in-memory render target (framebuffer) in host memory.
 */
class Framebuffer : public RenderTarget {
    public:

        Framebuffer( const char* name, Object* parent =0 );
        ~Framebuffer();

        size_t getFrameSize() const { return frame_size; }

        virtual void reinitialize();
        virtual bool synchronize();

        void *getBuffer() { return buffer; }

    protected:
        void* buffer;
        size_t frame_size;
        int mode;

        virtual void* allocateBuffer( size_t framesize );
        virtual void destroyBuffer( void* );
        size_t calculateFrameSize() const;
};

/*
 * Implements an additional compression layer of the regular (raw) framebuffer class.
 */
class CompressedFramebuffer : public Framebuffer {
    public:
        enum Mode {
            PIXMAP = VOXOWL_FBMODE_PIXMAP
#ifdef VOXOWL_USE_TURBOJPEG
            ,JPEG = VOXOWL_FBMODE_JPEG
#endif
        };
        CompressedFramebuffer( const char* name, Object* parent =0 );
        ~CompressedFramebuffer();

        void setMode( Mode );
        Mode getMode() const { return mode; }

        virtual void reinitialize() override;
        virtual bool synchronize() override;

        const void* getCompressedBuffer();
        size_t compressedBufferSize();

    private:
        Mode mode;
#ifdef VOXOWL_USE_TURBOJPEG
        JPEGEncoder jpeg;
#endif
};


#pragma once

//#include <mutex>
#include <inttypes.h>
#include <string>

#include "platform.h"
#include "object.h"
#include "voxowl.h"

#define FRAMEBUFFER_DEFAULT_WIDTH 800
#define FRAMEBUFFER_DEFAULT_HEIGHT 600

class Framebuffer : public Object {
    public:
        enum PIXEL_FORMAT {
            PF_NONE = VOXOWL_PF_NONE,
            PF_RGB888 = VOXOWL_PF_RGB888
        };
        enum MODE {
            MODE_NONE = VOXOWL_FBMODE_NONE,
            MODE_PIXMAP = VOXOWL_FBMODE_PIXMAP
#ifdef VOXOWL_USE_TURBOJPEG
            ,MODE_JPEG = VOXOWL_FBMODE_JPEG
#endif
        };

        enum TARGET {
            TARGET_NONE = VOXOWL_TARGET_NONE,
            TARGET_FILE = VOXOWL_TARGET_FILE,
            TARGET_REMOTE = VOXOWL_TARGET_REMOTE
        };

        Framebuffer( const char* name, Object* parent =0 );
        ~Framebuffer();

        void setTarget( int );
        int getTarget() const { return target; }

        void setPixelFormat( int );
        int getPixelFormat() const { return pixel_format; }

        void setMode( int );
        int getMode() const { return mode; }

        void setSize( int width, int height );
        int getWidth() const { return width; }
        int getHeight() const { return height; }

        void setAASamples( int xsize, int ysize );
        int getAAXSamples() const { return aa_xsamples; }
        int getAAYSamples() const { return aa_ysamples; }

        uint32_t getFrameSize() const { return frame_size; }
        std::string errorString() const { return err_str; }

        void reinitialize();
        bool read();
        bool write();

        void *data() { return frame_begin; }

        virtual Variant callMeta( const std::string& method, const Variant::list& args );
        virtual bool setMeta( const std::string& property, const Variant& value );
        virtual Variant getMeta( const std::string& property ) const;

    private:
        void* header_begin;
        void* frame_begin;
        uint32_t frame_size;
        int width, height;
        int pixel_format;
        int mode;
        int target;
        int aa_xsamples, aa_ysamples;
        std::string err_str;

        bool setError( bool, const std::string& );
        uint32_t calculateFrameSize() const;

#ifdef VOXOWL_USE_TURBOJPEG
        class jpeg_t {
            public:
               jpeg_t() : compressedImage(0), size(0) {}
                unsigned char *compressedImage;
                long unsigned int size;
        } jpeg;
        bool jpegEncode();
        void jpegCleanup();
#endif

        //std::mutex update_lock;
};

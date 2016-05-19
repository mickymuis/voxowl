#pragma once

#include <mutex>
#include <cinttypes>

#include "object.h"

class Framebuffer : public Object {
    public:
        enum PIXEL_FORMAT {
            PF_NONE,
            PF_RGBA
        };
        enum MODE {
            MODE_NONE,
            MODE_PIXMAP
        };

        Framebuffer( const char* name, Object* parent =0 );
        ~Framebuffer();

        void setPixelFormat( PIXEL_FORMAT );
        int getPixelFormat() const { return pixel_format; }

        void setMode( MODE );
        int getMode() const { return mode; }

        void setSize( int width, int height );
        int getWidth() const { return width; }
        int getHeight() const { return height; }

        uint32_t getFrameSize() const;

        void reinitialize();

        virtual Variant callMeta( const std::string& method, const Variant::list& args );

    private:
        int width, height;
        int pixel_format;
        int mode;
        int target;
};

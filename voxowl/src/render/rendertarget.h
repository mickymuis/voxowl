#pragma once

#include "../core/object.h"
#include "../core/voxowl.h"
#include "../core/platform.h"

/**
 * Abstract class that represents a data sink to the rendering process.
 * A render target can be, for example, a framebuffer or an OpenGL texture.
 */
class RenderTarget : public Object {
    public:
        enum PIXEL_FORMAT {
            RGB888 = VOXOWL_PF_RGB888
        };
        RenderTarget( const char* name, Object* parent =0 );
        ~RenderTarget();

        void setPixelFormat( int );
        int getPixelFormat() const { return pixel_format; }
        int channels() const;
        int bytesPerPixel() const;

        void setSize( int w, int h );
        int getWidth() const { return width; }
        int getHeight() const { return height; }

        void setAASamples( int xsize, int ysize );
        int getAAXSamples() const { return aa_xsamples; }
        int getAAYSamples() const { return aa_ysamples; }

        void setClearColor( glm::vec4 c ) { clear_color =c; }
        glm::vec4 getClearColor() const { return clear_color; }

        std::string errorString() const { return err_str; }

        virtual bool synchronize() =0;
        virtual void reinitialize() =0;

        virtual Variant callMeta( const std::string& method, const Variant::list& args ) override;
        virtual bool setMeta( const std::string& property, const Variant& value ) override;
        virtual Variant getMeta( const std::string& property ) const override;

    protected:
        int width, height;
        int pixel_format;
        int aa_xsamples, aa_ysamples;
        glm::vec4 clear_color;
        std::string err_str;

        bool setError( bool, const std::string& );
};

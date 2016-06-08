#pragma once

#include <voxowl.h>
#include "object.h"

class Camera;
class Framebuffer;
class Volume;

class Renderer : public Object {
    public:
        Renderer( const char *name, Object *parent =0 );
        ~Renderer();

        void setCamera( Camera* );
        Camera *getCamera() const;

        void setFramebuffer( Framebuffer* );
        Framebuffer* getFramebuffer() const;

        void setVolume( Volume* );
        Volume* getVolume() const;

        bool render();

        virtual bool beginRender() =0;
        virtual bool synchronize() =0;

        virtual Variant callMeta( const std::string& method, const Variant::list& args );
        virtual bool setMeta( const std::string& property, const Variant& value );
        virtual Variant getMeta( const std::string& property ) const;
    
        std::string errorString() const { return err_str; }

    protected:
        std::string err_str;
        bool setError( bool, const std::string& );
        Camera *camera;
        Framebuffer *fb;
        Volume *vol;


};

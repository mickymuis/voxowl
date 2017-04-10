#pragma once

#include <voxowl.h>
#include "object.h"

class Camera;
class RenderTarget;
class Volume;

class Renderer : public Object {
    public:
        typedef enum {
            FEATURE_AA =0x1,
            FEATURE_SSAO =0x2,
            FEATURE_SSNA =0x4,
            FEATURE_LIGHTING =0x8
        } Feature;
        Renderer( const char *name, Object *parent =0 );
        ~Renderer();

        void setCamera( Camera* );
        Camera *getCamera() const;

        void setTarget( RenderTarget* );
        RenderTarget* getTarget() const;

        void setVolume( Volume* );
        Volume* getVolume() const;

        void setFeature( Feature, bool );
        void enable( Feature );
        void disable( Feature );
        bool isEnabled( Feature ) const;

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
        RenderTarget *target;
        Volume *vol;
        int features;
};

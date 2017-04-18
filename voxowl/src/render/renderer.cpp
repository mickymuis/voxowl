#include "renderer.h"
#include "framebuffer.h"
#include "volume.h"
#include "camera.h"
#include <typeinfo>

Renderer::Renderer( const char *name, Object *parent )
    : Object( name, parent ) {
    target =nullptr;
    vol =nullptr;
    camera =nullptr;
    features =0;
    addProperty( "camera" );
    addProperty( "target" );
    addProperty( "volume" );
    addProperty( "featureAA" );
    addProperty( "featureSSAO" );
    addProperty( "featureSSNA" );
    addProperty( "featureLighting" );
    addMethod( "render" );
}

Renderer::~Renderer() {

}

void 
Renderer::setCamera( Camera* c ) { camera =c; }
Camera*
Renderer::getCamera() const { return camera; }

void 
Renderer::setTarget( RenderTarget* t ) { target =t; }

RenderTarget* 
Renderer::getTarget() const { return target; }

void
Renderer::setVolume( Volume* v ) { vol =v; }
Volume* 
Renderer::getVolume() const { return vol; }

void
Renderer::setFeature( Feature f, bool b ) {
    if( b )
        enable( f );
    else
        disable( f );
}

void 
Renderer::enable( Feature f ) {
    features |= (int)f;
}

void 
Renderer::disable( Feature f ) {
    features &= ~(int)f;
}

bool 
Renderer::isEnabled( Feature f ) const {
    return (features & (int)f);
}

bool 
Renderer::render() {
    if( !beginRender() ) return false;
    return synchronize();
}

Variant 
Renderer::callMeta( const std::string& method, const Variant::list& args ) {
    if( method == "render" ) {
        if( !render() ) 
            return errorString();
        return Variant();
    }

    return Object::callMeta( method, args );
}

bool 
Renderer::setMeta( const std::string& property, const Variant& value ) {
    if( property == "target" ) {
        Object *obj =value.toObject();
        try {
            if( typeid(*obj) == typeid(RenderTarget) ) {
                //std::lock_guard<std::mutex> lock ( write_lock );
                setTarget( dynamic_cast<RenderTarget*>(obj) );
                return true;
            }
            else
                setTarget( 0 );
        } catch( std::bad_typeid& e ) { setTarget( 0 ); }
        return false;
    } else if( property == "camera" ) {
        Object *obj =value.toObject();
        try {
            if( typeid(*obj) == typeid(Camera) ) {
                //std::lock_guard<std::mutex> lock ( write_lock );
                setCamera( dynamic_cast<Camera*>(obj) );
                return true;
            }
            else
                setCamera( 0 );
        } catch( std::bad_typeid& e ) { setCamera( 0 ); }
        return false;
    } else if( property == "volume" ) {
        Volume *vol;
        try {
            vol = dynamic_cast<Volume*>( value.toObject() );

            if( vol ) {
                //std::lock_guard<std::mutex> lock ( write_lock );
                setVolume( vol );
                return true;
            }
            else
                setVolume( 0 );
        } catch( std::bad_typeid& e ) { setVolume( 0 ); }
        return false;
    } else if( property == "featureAA" ) {
        setFeature( FEATURE_AA, value.toBool() );
        return true;
    } else if( property == "featureSSAO" ) {
        setFeature( FEATURE_SSAO, value.toBool() );
        return true;
    } else if( property == "featureSSNA" ) {
        setFeature( FEATURE_SSNA, value.toBool() );
        return true;
    } else if( property == "featureLighting" ) {
        setFeature( FEATURE_LIGHTING, value.toBool() );
        return true;
    }

    return Object::setMeta( property, value );
}

Variant 
Renderer::getMeta( const std::string& property ) const {
    if( property == "target" )
        return getTarget();
    else if( property == "camera" )
        return getCamera();
    else if( property == "volume" )
        return getVolume();
    else if( property == "featureAA" )
        return isEnabled( FEATURE_AA );
    else if( property == "featureSSAO" )
        return isEnabled( FEATURE_SSAO );
    else if( property == "featureSSNA" )
        return isEnabled( FEATURE_SSNA );
    else if( property == "featureLighting" )
        return isEnabled( FEATURE_LIGHTING );
    return Object::getMeta( property );
}

bool 
Renderer::setError( bool err, const std::string& str ) {
    if( err )
        err_str =str;
    else
        err_str ="";
    return !err;
}

#include "rendertarget.h"

RenderTarget::RenderTarget( const char* name, Object* parent )
    : Object( name, parent ) {
    pixel_format =RGB888;
    width = VOXOWL_DEFAULT_WIDTH;
    height = VOXOWL_DEFAULT_HEIGHT;
    aa_xsamples =aa_ysamples =1;

    addMethod( "reinitialize" );
    addMethod( "synchronize" );
    addProperty( "width" );
    addProperty( "height" );
    addProperty( "pixelFormat" );
}

RenderTarget::~RenderTarget() {
}

int 
RenderTarget::channels() const {
    int r =0;
    switch( pixel_format ) {
        case RGB888:
            r =3;
            break;
    }
    return r;
}

int 
RenderTarget::bytesPerPixel() const {
    int r =0;
    switch( pixel_format ) {
        case RGB888:
            r =3;
            break;
    }
    return r;
}

void 
RenderTarget::setPixelFormat( int f ) {
    pixel_format =f;
    reinitialize();
}
        
void 
RenderTarget::setSize( int w, int h ) {
    width =w;
    height =h;
    reinitialize();
}

void 
RenderTarget::setAASamples( int xsize, int ysize ) {
    aa_xsamples =xsize;
    aa_ysamples =ysize;
    reinitialize();
}

Variant 
RenderTarget::callMeta( const std::string& method, const Variant::list& args ) {
    if( method == "reinitialize" ) {
        reinitialize();
        return Variant();
    }
    else if( method == "synchronize" ) {
        synchronize();
        return errorString();
    }
    return Object::callMeta( method, args );
}

bool 
RenderTarget::setMeta( const std::string& property, const Variant& value ) {
    if( property == "pixelFormat" ) {
        setPixelFormat( value.toInt() );
    } else if( property == "width" ) {
        setSize( value.toInt(), height );
    } else if( property == "height" ) {
        setSize( width, value.toInt() );
    } else {
        return Object::setMeta( property, value );
    }

    return true;
}

Variant 
RenderTarget::getMeta( const std::string& property ) const {
    if( property == "pixelFormat" ) {
        return pixel_format;
    } else if( property == "width" ) {
        return width;
    } else if( property == "height" ) {
        return height;
    } 

    return Object::getMeta( property );
}

bool 
RenderTarget::setError( bool err, const std::string& str ) {
    if( err )
        err_str =str;
    else
        err_str ="";
    return !err;
}

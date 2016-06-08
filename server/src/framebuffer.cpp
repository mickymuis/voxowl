#include "framebuffer.h"
#include "server.h"
#include "packetbuffer.h"
#include "voxowl.h"

Framebuffer::Framebuffer( const char* name, Object* parent )
    : Object( name, parent ) {
    mode = target = pixel_format =0;
    width = FRAMEBUFFER_DEFAULT_WIDTH;
    height = FRAMEBUFFER_DEFAULT_HEIGHT;
    header_begin =0;
    frame_begin =0;
    frame_size =0;
    aa_xsamples =aa_ysamples =1;

    addMethod( "getFrameSize" );
    addMethod( "reinitialize" );
    addMethod( "read" );
    addMethod( "write" );
    addProperty( "width" );
    addProperty( "height" );
    addProperty( "target" );
    addProperty( "pixelFormat" );
    addProperty( "mode" );
}

Framebuffer::~Framebuffer() {
    if( header_begin )
        free( header_begin );
}

void 
Framebuffer::setTarget( int t ){
    //std::lock_guard<std::mutex> lock( update_lock );
    target =t;
}

void 
Framebuffer::setPixelFormat( int pf ){
    //std::lock_guard<std::mutex> lock( update_lock );
    pixel_format =pf;
}

void 
Framebuffer::setMode( int m ) {
    //std::lock_guard<std::mutex> lock( update_lock );
    mode =m;
}

void 
Framebuffer::setSize( int w, int h ) {
    //std::lock_guard<std::mutex> lock( update_lock );
    width =w;
    height =h;
}

uint32_t
Framebuffer::calculateFrameSize() const {
    uint32_t size;
    uint32_t pixel_size;
    switch( pixel_format ) {
        case PF_RGB888: pixel_size =3; break;
        default: pixel_size =0; break;
    }

    switch( mode ) {
        case MODE_PIXMAP: size = width*height*pixel_size; break;
        default: size =0; break;
    }
    return size;
}

void 
Framebuffer::setAASamples( int xsize, int ysize ) {
    //std::lock_guard<std::mutex> lock( update_lock );
    aa_xsamples =xsize;
    aa_ysamples =ysize;
}

void 
Framebuffer::reinitialize() {
    //std::lock_guard<std::mutex> lock( update_lock );

    if( header_begin )
        free( header_begin );

    frame_size =calculateFrameSize();

    /* We allocate a contiguous buffer for the header + frame */

    header_begin =malloc( frame_size + sizeof( struct voxowl_frame_header_t ) );
    frame_begin =header_begin + sizeof( struct voxowl_frame_header_t );

    /* Initialize the header, we need it for network transmission */
    struct voxowl_frame_header_t* header =(struct voxowl_frame_header_t*)header_begin;
    header->magic = VOXOWL_FRAME_MAGIC;
    header->pixel_format = pixel_format;
    header->fb_mode = mode;
    header->width = width;
    header->height = height;
    header->frame_size = frame_size;
}

bool 
Framebuffer::read() {
    //std::lock_guard<std::mutex> lock( update_lock );

}

bool 
Framebuffer::write() {
   // std::lock_guard<std::mutex> lock( update_lock );

    switch( target ) {
        case TARGET_FILE:
            return setError( true, "Not implemented" );
            break;
        case TARGET_REMOTE: {
            Server *server = Server::active();
            if( !server )
                return setError( true, "Framebuffer: server not running" );
            Connection *c =server->getDataConnection();
            if( !c )
                return setError( true, "Framebuffer: no data connection" );

            Packet packet;
            packet.connection =c;
            packet.direction =Packet::SEND;
            packet.mode =Packet::DATA;
            packet.payload =header_begin;
            packet.size =frame_size + sizeof( struct voxowl_frame_header_t );
            packet.own_payload =false;

            c->pbuffer->enqueue( packet );
            break;
        }

        default: break;
    }
    return true;
}

Variant 
Framebuffer::callMeta( const std::string& method, const Variant::list& args ) {
    if( method == "getFrameSize" )
        return getFrameSize();
    else if( method == "reinitialize" ) {
        reinitialize();
        return Variant();
    }
    else if( method == "read" ) {
        read();
        return errorString();
    }
    else if( method == "write" ) {
        write();
        return errorString();
    }
    return Object::callMeta( method, args );
}

bool 
Framebuffer::setMeta( const std::string& property, const Variant& value ) {
    if( property == "target" ) {
        setTarget( value.toInt() );
    } else if( property == "mode" ) {
        setMode( value.toInt() );
    } else if( property == "pixelFormat" ) {
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
Framebuffer::getMeta( const std::string& property ) const {
    if( property == "target" ) {
        return target;
    } else if( property == "mode" ) {
        return mode;
    } else if( property == "pixelFormat" ) {
        return pixel_format;
    } else if( property == "width" ) {
        return width;
    } else if( property == "height" ) {
        return height;
    } 

    return Object::getMeta( property );
}

bool 
Framebuffer::setError( bool err, const std::string& str ) {
    if( err )
        err_str =str;
    else
        err_str ="";
    return !err;
}


#include "fbrendertarget.h"
#include "server.h"
#include "packetbuffer.h"
#include "voxowl.h"
#include "bmp.h"
#include <fstream>
#include <vector>

//
// Implementation of FBImageTarget
//

FBImageTarget::FBImageTarget( const char* name, Object* parent ) 
    : CompressedFramebuffer( name, parent ) {
}

FBImageTarget::~FBImageTarget() {}

bool
FBImageTarget::synchronize() {
    if( !CompressedFramebuffer::synchronize() ) return false;

    if( path.empty() )
        return setError( true, "No path specified" );

    if( channels() != 3 )        
        return setError( true, "Unsupported pixel format" );
    

    std::ofstream file;
    file.open( path, std::ofstream::trunc ); 
    if( !file.is_open() )
        return setError( true, "Specified file could not be opened for writing: " + path );

    switch( getMode() ) {
        case PIXMAP: { // We use BMP in this case
            std::vector<uint8_t> output;
            size_t output_size =bitmap_encode_multichannel_8bit( (const uint8_t*)getBuffer(), getWidth(), getHeight(), channels(), output );
            file.write((const char*)&output[0], output_size);
            break;
        }
#ifdef VOXOWL_USE_TURBOJPEG
        case JPEG: { // Just write the contents of the JPEG buffer directly
            if( compressedBufferSize() ) {
                file.write( (const char*)getCompressedBuffer(), compressedBufferSize() );
            }
            break;
        }
#endif
    }

    file.close();
    return true;
}

//
// Implementation of FBRemoteTarget
// 

FBRemoteTarget::FBRemoteTarget( const char* name, Object* parent )
    : CompressedFramebuffer( name, parent ), header_begin( nullptr ) {}

FBRemoteTarget::~FBRemoteTarget() {
    if( header_begin ) free( header_begin );
}

void
FBRemoteTarget::reinitialize() {
    CompressedFramebuffer::reinitialize();

    if( header_begin )
        free( header_begin );

    header_begin =malloc( sizeof( struct voxowl_frame_header_t ) );

    /* Initialize the header, we need it for network transmission */
    struct voxowl_frame_header_t* header =(struct voxowl_frame_header_t*)header_begin;
    header->magic = VOXOWL_FRAME_MAGIC;
    header->pixel_format = (int)getPixelFormat();
    header->fb_mode = (int)getMode();
    header->width = getWidth();
    header->height = getHeight();
}

bool
FBRemoteTarget::synchronize() {
    if( !CompressedFramebuffer::synchronize() )
        return false;

    Server *server = Server::active();
    if( !server )
        return setError( true, "Server not running" );
    Connection *c =server->getDataConnection();
    if( !c )
        return setError( true, "No data connection" );

    Packet packet;
    packet.connection =c;
    packet.direction =Packet::SEND;
    packet.mode =Packet::DATA;
    packet.own_payload =false;
    // Need to split into two packets

    // Update the frame_size field to the accurate size of the jpeg
    struct voxowl_frame_header_t* header =(struct voxowl_frame_header_t*)header_begin;
    header->frame_size = compressedBufferSize();

    // Send only the header
    packet.payload =header_begin;
    packet.size =sizeof( struct voxowl_frame_header_t );

    c->pbuffer->enqueue( packet );

    // Send the image data as a separate packet
    Packet image =packet;
    image.payload =(void*)getCompressedBuffer();
    image.size =compressedBufferSize();

    c->pbuffer->enqueue( image );

    return true;
}

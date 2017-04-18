#include "server.h"
#include <iostream>
#include <stdint.h>
#include <string.h>
#include <stdlib.h>
#include <thread>
#include <chrono>
#include <errno.h>
#include <list>
#include <sstream>

#include "core/platform.h"
#include "network/server.h"
#include "core/types.h"
#include "core/parser.h"
#include "network/packetbuffer.h"
#include "render/fbrendertarget.h"
#include "render/camera.h"
#include "cuda/raycast_cuda.h"
#include "util/mengersponge.h"
#include "render/volume_loader.h"

#define MODULE_KEY "server"
#define MODULE_DESCRIPTION "Run the network render server"

Module::Descriptor 
ServerModule::getDescriptor() {
    Descriptor d;
    d.key =MODULE_KEY;
    d.description =MODULE_DESCRIPTION;
    std::stringstream str;
    str << "\t --port \t\t Specify listening (TCP) port for control connections (default: " << VOXOWL_DEFAULT_PORT << ")" << std::endl
        << "\t -l" << std::endl
        << "\t --load <file> \t\t Preload <file> on startup of the server." << std::endl
        << "\t -w" << std::endl
        << "\t --width \t\t Specify the width of the framebuffer." << std::endl
        << "\t -h" << std::endl
        << "\t --height \t\t Specify the height of the framebuffer." << std::endl
        << "\t --menger <n> \t\t Load the built-in Menger-sponge volume with depth <n>, for testing purposes." << std::endl;
    d.helpText =str.str();
    return d;
}

static Parser parser;

static void 
incoming_packet( const Packet& p_recv ) {

    if( p_recv.mode == Packet::CHAR ) {
        std::string str( (char*)p_recv.payload, p_recv.size );
            
        bool last =false, error;
        Variant v;
        Statement* s;

        s =parser.parse( parser.tokenize( str ) );
        if( s ) {
            error =parser.evaluate( v, last, s );
            //std::cerr << v << std::endl;

            if( last && p_recv.connection )
                p_recv.connection->closeDeferred();
            else if( p_recv.connection && p_recv.connection->pbuffer ) {
                Packet packet;

                packet.connection =p_recv.connection;
                packet.direction =Packet::SEND;
                packet.mode =Packet::CHAR;
                std::string buffer = v.toString();
                packet.size =buffer.length()+1;
                packet.payload =(char*)malloc( (buffer.length()+1) * sizeof( char ) );
                memcpy( packet.payload, buffer.c_str(), packet.size * sizeof( char ) );
                packet.own_payload =true;
                p_recv.connection->pbuffer->enqueue( packet );
            }

            delete s;
        }
    }
}

static void 
outgoing_packet( const Packet& p_send ) {
    if( !p_send.connection || !p_send.connection->pbuffer )
        return;

    /* Delegate this packet to the queue of the outgoing connection */
    p_send.connection->pbuffer->enqueue( p_send );
}

int 
ServerModule::exec( const arglist_t& args ) {

    Variant::list unnamed =unnamedArgs( args );

    /* Setup the environment */
    Server server( "server", environment );
    PacketBuffer pbuffer;

    FBRemoteTarget fb( "framebuffer", environment );
    fb.setSize( 1024, 768 );
    fb.setMode( CompressedFramebuffer::JPEG );
    fb.setPixelFormat( RenderTarget::RGB888 );
    fb.setAASamples( 1, 1 );
    fb.setClearColor( glm::vec4( 1.f, 1.f, 1.f, 1.f ) );
    //fb.setClearColor( glm::vec4( .85f, .84f, .75f, 1.f ) );
    fb.reinitialize();

    Camera camera( "camera", environment );
    
  /*  MengerSponge sponge( "mengersponge", environment );
    sponge.setDepth( 7 );*/

    /* Optionally, pre-load a volume from disk (--load or -l option) */
    VolumeLoader loader( "volumeloader", environment );
    std::string path;
    if( args.count( "load" ) > 0 )
        path =args.at("load").toString();
    else if( args.count( "l" ) > 0 )
        path =args.at("l").toString();
    else if( unnamed.size() > 0 ) {
        path =unnamed.front().toString();
        unnamed.erase( unnamed.begin() );
    }

    if( !path.empty() ) {
        if( !loader.open( path ) )
            std::cerr << "Warning: could not load `" << path << "' (" 
                << loader.errorString() << ")" << std::endl;
    }

#ifdef VOXOWL_USE_CUDA
    RaycasterCUDA renderer( "renderer", environment );
    renderer.setTarget( &fb );
    renderer.setCamera( &camera );
    renderer.setVolume( &loader );
#else
    std::cerr << "Warning: currently, only the CUDA raycaster is supported and Voxowl is not compiled with CUDA support" << std::endl;
#endif

    /*renderer.beginRender();
    renderer.synchronize();

        VolumeVoxelmapStorageDetail* detail =dynamic_cast<VolumeVoxelmapStorageDetail*>(loader.storageDetail());
        if( !detail ) {
            fprintf( stderr, "File not a voxelmap\n" );
            return -1;
        }
        voxelmap_t* voxelmap =detail->getVoxelmap();
        VolumeSVMipmap volume( "volume", environment );
        
        svmipmap_t svmm;
        bzero( &svmm, sizeof( svmipmap_t ) );
        svmm_encode_opts_t opts;
        svmmSetOpts( &opts, voxelmap, 50 );

        svmmEncode( &svmm, voxelmap, opts );
        volume.setSVMipmap( &svmm );
        renderer.setVolume( &volume );*/

    /* */
    uint32_t portnum =VOXOWL_DEFAULT_PORT;
    if( args.count( "port" ) > 0 ) {
        portnum =args.at( "port" ).toInt();
    }

    /* Setup the packet buffer and run it in another thread */
    parser.setScope( environment );
    pbuffer.setIncomingPacketHandler( &incoming_packet );
    pbuffer.setOutgoingPacketHandler( &outgoing_packet );
    std::thread pbuffer_thread( pbufferMain, &pbuffer );

    // dummy
    //std::thread dummy( &dummy_data_thread, &server );

    /* Setup the server and run the listener in this thread */
    server.setPort( portnum );
    server.setLogStream( std::cerr );
    server.setControlPBuffer( &pbuffer );
    server.mainloop( environment );

    /* Wait for any threads to join */
    pbuffer.stopThread();
    pbuffer_thread.join();
}

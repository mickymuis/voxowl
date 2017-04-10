#include <stdint.h>
#include <string.h>
#include <stdlib.h>
#include <thread>
#include <chrono>
#include <errno.h>
#include <iostream>
#include <list>

#include "platform.h"
#include "server.h"
#include "types.h"
#include "parser.h"
#include "packetbuffer.h"
#include "fbrendertarget.h"
#include "camera.h"
#include "raycast_cuda.h"
#include "mengersponge.h"
#include "volume_loader.h"
#include "performance_counter.h"
#include "testing.h"

Parser parser;

void 
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

void 
outgoing_packet( const Packet& p_send ) {
    if( !p_send.connection || !p_send.connection->pbuffer )
        return;

    /* Delegate this packet to the queue of the outgoing connection */
    p_send.connection->pbuffer->enqueue( p_send );
}

//#define TESTING

int
main( int argc, char** argv ) {

#ifdef TESTING
    return testingMain( argc, argv );
#endif

    /* Setup the environment */
    Object root("root");
    Server server( "server", &root );
    PacketBuffer pbuffer;

    FBRemoteTarget fb( "framebuffer", &root );
    fb.setSize( 1024, 768 );
    fb.setMode( CompressedFramebuffer::JPEG );
    fb.setPixelFormat( RenderTarget::RGB888 );
    fb.setAASamples( 1, 1 );
    fb.setClearColor( glm::vec4( 1.f, 1.f, 1.f, 1.f ) );
    //fb.setClearColor( glm::vec4( .85f, .84f, .75f, 1.f ) );
    fb.reinitialize();

    Camera camera( "camera", &root );
    
  /*  MengerSponge sponge( "mengersponge", &root );
    sponge.setDepth( 7 );*/

    VolumeLoader loader( "volumeloader", &root );
    if( argc > 1 ) {
        std::string path( argv[1] );
        if( !loader.open( path ) )
            std::cerr << loader.errorString() << std::endl;
    }

    RaycasterCUDA renderer( "renderer", &root );
    renderer.setTarget( &fb );
    renderer.setCamera( &camera );
    renderer.setVolume( &loader );

    /*renderer.beginRender();
    renderer.synchronize();

        VolumeVoxelmapStorageDetail* detail =dynamic_cast<VolumeVoxelmapStorageDetail*>(loader.storageDetail());
        if( !detail ) {
            fprintf( stderr, "File not a voxelmap\n" );
            return -1;
        }
        voxelmap_t* voxelmap =detail->getVoxelmap();
        VolumeSVMipmap volume( "volume", &root );
        
        svmipmap_t svmm;
        bzero( &svmm, sizeof( svmipmap_t ) );
        svmm_encode_opts_t opts;
        svmmSetOpts( &opts, voxelmap, 50 );

        svmmEncode( &svmm, voxelmap, opts );
        volume.setSVMipmap( &svmm );
        renderer.setVolume( &volume );*/

    /* */
    uint32_t portnum =6789;
    if( argc > 1 && atoi( argv[1] ) != 0 )
        portnum =atoi( argv[1] );

    /* Setup the packet buffer and run it in another thread */
    parser.setScope( &root );
    pbuffer.setIncomingPacketHandler( &incoming_packet );
    pbuffer.setOutgoingPacketHandler( &outgoing_packet );
    std::thread pbuffer_thread( pbufferMain, &pbuffer );

    // dummy
    //std::thread dummy( &dummy_data_thread, &server );

    /* Setup the server and run the listener in this thread */
    server.setPort( portnum );
    server.setLogStream( std::cerr );
    server.setControlPBuffer( &pbuffer );
    server.mainloop( &root );

    /* Wait for any threads to join */
    pbuffer.stopThread();
    pbuffer_thread.join();

    PerformanceCounter::printAll( std::cout );
    PerformanceCounter::cleanup();

    return 0;
}


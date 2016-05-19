#include <stdint.h>
#include <string.h>
#include <thread>
#include <chrono>
#include <errno.h>
#include <iostream>
#include <list>

#include "server.h"
#include "platform.h"
#include "types.h"
#include "parser.h"
#include "packetbuffer.h"
#include "main-gpu.h"

Parser parser;

void 
incoming_packet( const Packet& p_recv ) {

    if( p_recv.mode == Packet::CHAR ) {
        std::string str( (char*)p_recv.payload, p_recv.size );
        std::cerr << str.c_str() << std::endl;
            
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

/*void 
dummy_data_thread( Server *server ) {
    Packet packet;
    packet.direction =Packet::SEND;
    packet.mode =Packet::CHAR;
    packet.payload =(void*)"Molly!";
    packet.own_payload =false;
    packet.size =7;

    while( true ) {
        //std::this_thread::sleep_for( std::chrono::seconds( 1 ) );
        sleep( 1 );

        Connection *conn =server->getDataConnection();
        if( conn ) {
            packet.connection =conn;
            conn->pbuffer->enqueue( packet );
            std::cerr << "Enqueue yar!" << std::endl;
        }
    }
}*/

/* Test class
   class Gpu : public Object {
    public:
        Gpu( const char* name, Object* parent ) : Object( name, parent ) {
            addMethod( "exec" );
        }

        virtual Variant callMeta( const std::string& method, const Variant::list& args ) {
            if( method == "exec" ) {
                main_gpu(0,0);
                return Variant( "done" );
            }
            return Object::callMeta( method, args );
        }
};
*/

int
main( int argc, char** argv ) {

    /* Setup the environment */
    Object root("root");
    Server server( "server", &root );
    PacketBuffer pbuffer;
    //Gpu gpu( "gpu", &root );

    /* */
    uint32_t portnum =5678;
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

    return 0;
}


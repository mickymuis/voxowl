#include <stdint.h>
#include <string.h>
#include <thread>
#include <errno.h>
#include <iostream>
#include <list>

#include "network.h"
#include "platform.h"
#include "types.h"
#include "parser.h"

typedef struct {
    ClientSocket *connection;
    std::thread *thread;
} conthread_t;
typedef std::list<conthread_t> conthreadlist_t;

void
newConnection( ClientSocket* c ) {
    Parser h( c->getBuffer(), 0 );
    while( h.evaluateNext() ) {}

    c->close();
    std::cerr << "Connection closed" << std::endl;
}

int
main( int argc, char** argv ) {

    uint32_t portnum =5678;

    ListenSocket sock;
    if( !sock.bind( portnum ) || !sock.listen()  ) {
        std::cerr << "Could not bind to port " << portnum << std::endl;
        std::cerr << strerror( errno ) << std::endl;
        exit( 1 );
    }

    std::cerr << VERSION_FULL_NAME << " listening to port " << portnum << std::endl;

    conthreadlist_t connection_list;

    while( 1 ) {
        ClientSocket *c;
        if( ( c = sock.accept() ) ) {
            std::cerr << "Accepting new connection" << std::endl;
            conthread_t t;
            t.connection =c;
            t.thread =new std::thread( newConnection, c );
            connection_list.push_back( t );
        }
        else
            break;

        // Cleanup any closed connections
        conthreadlist_t::iterator it;
        for( it = connection_list.begin(); it != connection_list.end(); )
            if( !(*it).connection->isOpen() ) {
                delete (*it).connection;
                delete (*it).thread;
                it =connection_list.erase( it );
            }
            else
                it++;
    }

    conthreadlist_t::iterator it;
    for( it = connection_list.begin(); it != connection_list.end(); ) {
        delete (*it).connection;
        delete (*it).thread;
    }

    return 0;
}


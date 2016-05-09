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
newConnection( ClientSocket* c, Object* scope ) {
    Parser h( c->getBuffer(), scope );
    while( h.evaluateNext() ) {}

    c->close();
    std::cerr << "Connection closed" << std::endl;
}

class MyClass : public Object {
    public:
        MyClass( const std::string& name, Object* obj ) : Object( name, obj ) {
            method_list.push_back( "func" );
            method_list.push_back( "me" );
            method_list.push_back( "add2d" );
            addProperty( "name" );
            addProperty( "foo" );
            addProperty( "bar" );
        }

        virtual Variant callMeta( const std::string& method, const Variant::list& args ) {
            std::cerr << method << " called" << std::endl;
            if( method == "func" ) {
                if( args.size() )
                    return Variant( "You said: " + args[0].toString() );
                else
                    return Variant( "Hello world!" );
            } else if( method == "me" ) {
                return Variant( this );

            } else if( method == "add2d" ) {
                if( args.size() > 1 )
                    return Variant( 5. + args[0].toReal() + args[1].toReal() );
                else
                    return Variant();
            }
            else
                return Object::callMeta( method, args );
        }
};

int
main( int argc, char** argv ) {

    /* Test code */

    Object root("root");
    MyClass *myinstance = new MyClass( "myclass1", &root );
    ObjectFactory::newFactory( new ObjectFactoryT<MyClass>("MyClass") );

    /* */

    uint32_t portnum =5678;
    if( argc > 1 && atoi( argv[1] ) != 0 )
        portnum =atoi( argv[1] );

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
            t.thread =new std::thread( newConnection, c, &root );
            connection_list.push_back( t );
        }
        else
            break;

        // Cleanup any closed connections
        conthreadlist_t::iterator it;
        for( it = connection_list.begin(); it != connection_list.end(); )
            if( !(*it).connection->isOpen() ) {
                delete (*it).connection;
 //               delete (*it).thread;
                it =connection_list.erase( it );
            }
            else
                it++;
    }

    conthreadlist_t::iterator it;
    for( it = connection_list.begin(); it != connection_list.end(); ) {
        delete (*it).connection;
//        delete (*it).thread;
    }

    return 0;
}


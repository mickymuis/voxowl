#include "server.h"
#include "parser.h"
#include "platform.h"
#include "packetbuffer.h"

uint32_t
Connection::counter =0;

Connection::Connection( Object* parent )
    : Object( std::string( CONN_NAME_PREFIX ) + std::to_string( (long long unsigned int)counter++ ), parent ),
    pbuffer(0), socket(0), thread(0), should_close( false ) {
}

Connection::~Connection() 
{
    if( thread ) {
        try {
            thread->join();
        } catch( std::system_error& e ) {}
        delete thread;
    }
    if( pbuffer )
        delete pbuffer;     
    if( socket )
        delete socket;
}

//////

void 
connectionSendFunc( const Packet& packet ) {
    if( !packet.payload || !packet.connection || !packet.connection->socket->isOpen() ) return;

    if( packet.mode == Packet::CHAR ) {

        std::iostream stream(packet.connection->socket->getBuffer() );
        std::string str( (char*)packet.payload, packet.size );

        stream << str << std::endl;
    }
}

void
connectionMain( Connection* c, Server* server ) {

   std::iostream stream( c->socket->getBuffer() );

    do {

        if( c->socket->poll( 0, 1000 ) ) {
            if( stream.eof() || stream.bad() ) {
                c->socket->setState( Socket::CLOSED );
                break;
            }
            std::string buffer;
            std::getline( stream, buffer );
            if( buffer.size() <= 1 )
                continue;
            buffer.resize( buffer.size() -1);
            Packet packet;

            packet.connection = c;
            packet.direction =Packet::RECEIVE;
            packet.mode =Packet::CHAR;
            packet.size =buffer.size();
            packet.own_payload =true;
            packet.payload =malloc( buffer.size() * sizeof(char) );
            memcpy( packet.payload, buffer.c_str(), buffer.size() * sizeof(char) );

            server->pbuffer_control->enqueue( packet );
        }

        c->pbuffer->dispatchOutgoing();

        if( c->should_close )
            c->socket->close();
    } while( c->socket->isOpen() );

    std::cerr << "connection closing..." << std::endl;

//    c->socket->close();
    std::lock_guard<std::mutex> lock( server->write_lock );
    if( server->getDataConnection() == c )
        server->setDataConnection( NULL );

    server->log << "Connection closed" << std::endl;
}

//////

Server::Server( const std::string& name, Object* parent )
    : Object( name, parent ), _stop(false), log( std::cerr.rdbuf() ),
    connections( "connections", this ), data_connection( 0 ) {

    addMethod( "stop" );
    addMethod( "getDataConnection" );
    addMethod( "setDataConnection" );
    addProperty( "port" );
}

Server::~Server() {
    // Cleanup is either done at the end of mainloop() or by ~Object()
}

void
Server::setLogStream( const std::ostream& s ) {
    log.rdbuf( s.rdbuf() );
}

void 
Server::setPort( uint32_t p ) {
    _portnum =p;
    setMeta( "port", Variant( p ) );
}

bool 
Server::mainloop( Object* root ) {

    ListenSocket sock;
    sock.setBlocking( false );

    if( !sock.bind( _portnum ) || !sock.listen()  ) {
        log << "Could not bind to port " << _portnum << std::endl;
        log << strerror( errno ) << std::endl;
        exit( 1 );
    }

    log << VERSION_FULL_NAME << " listening to port " << _portnum << std::endl;

    connection_list_t::iterator it;
    while( !_stop ) {

        // Cleanup any closed connections
       /* We cannot do this because it's thread unsafe. We delete everything
        * on exit, which may also be a potential problem. FIXME
        for( it = connection_list.begin(); it != connection_list.end(); ) {
            if( !(*it)->socket->isOpen() ) {
                delete (*it);
                it =connection_list.erase( it );
            }
            else
                it++;
        }*/

        // Wait for one second at a time
        if( !sock.poll( 1 ) ) {
            continue;
        }

        // It seems a new connection attempt is being made
        ClientSocket *csock;
        if( ( csock = sock.accept() ) ) {
            log << "Accepting new connection" << std::endl;
            Connection *c = new Connection( &connections );
            c->socket =csock;
            c->pbuffer =new PacketBuffer();
            c->pbuffer->setOutgoingPacketHandler( &connectionSendFunc );
            c->thread =new std::thread( connectionMain, c, this );
            connection_list.push_back( c );
        }
        else
            break;

    }

    log << "Stopping server, waiting for connections to close." << std::endl;

    for( it = connection_list.begin(); it != connection_list.end(); it++ ) {
        if( (*it)->socket->isOpen() )
            (*it)->socket->close();
        delete (*it);
    }

    log << "All connections terminated." << std::endl;
    return true;
}

void 
Server::setDataConnection( Connection* conn ) {
    data_connection =conn;
}


Variant 
Server::callMeta( const std::string& method, const Variant::list& args ) {
    Variant v;
    if( method == "stop" ) {
        std::lock_guard<std::mutex> lock ( write_lock );
        _stop =true;
        return v;
    } else if( method == "getDataConnection" ) {
        return Variant( data_connection );
    } else if( method == "setDataConnection" ) {
        if( args.size() > 0 ) {
            Object *obj =args[0].toObject();
            //log << typeid(*obj).name() << " - " << typeid(Connection).name() << std::endl;
            
            if( typeid(*obj) == typeid(Connection) ) {
                std::lock_guard<std::mutex> lock ( write_lock );
                v.set( std::string( "Sending data to " ) +
                obj->getName() );
                Connection* conn= dynamic_cast<Connection*>(obj);
                setDataConnection( conn );
                return v;
            }
        }
    }

    return Object::callMeta( method, args );
}

#pragma once
#include "object.h"
#include "network.h"
#include "packetbuffer.h"

#include <stdint.h>
#include <string.h>
#include <thread>
#include <errno.h>
#include <iostream>
#include <ostream>
#include <list>
#include <string>

#define CONN_NAME_PREFIX "conn"

class Server;
class Connection;
class PacketBuffer;

typedef std::list<Connection*> connection_list_t;

void connectionMain( Connection* c, Server* server );
void connectionSendFunc( const Packet& packet );

class Server : public Object {
    public:
        Server( const std::string& name, Object* parent );
        ~Server();

        void setLogStream( const std::ostream& );
        void setPort( uint32_t );
        bool mainloop( Object *root );

        void setControlPBuffer( PacketBuffer* p ) { pbuffer_control =p; }

        void setDataConnection( Connection* );
        Connection *getDataConnection() const { return data_connection; }

        virtual Variant callMeta( const std::string& method, const Variant::list& args );

    private:
        friend void connectionMain( Connection*, Server* );
        // FIXME should be std::atomic_bool
        bool _stop;
        uint32_t _portnum;
        std::ostream log;

        Object connections;
        Connection * data_connection;
        connection_list_t connection_list;

        PacketBuffer *pbuffer_control;

        std::mutex write_lock;
};

class Connection : public Object {
    public:
        Connection( Object *parent );
        ~Connection();

        void closeDeferred() { should_close =true; }

        PacketBuffer *pbuffer;

    private:
        friend class Server;
        friend void connectionMain( Connection *, Server * );
        friend void connectionSendFunc( const Packet& );
        ClientSocket *socket;
        std::thread *thread;
        //std::mutex access_lock;
        bool should_close;

        //bool isDataConnection;

    private:
        static uint32_t counter;
};

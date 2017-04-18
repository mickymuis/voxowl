#pragma once
#include <sys/types.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <stdint.h>

#include "socketbuf.h"

class Socket {
    public:
        enum STATE {
            OPEN,
            CLOSED,
            ERROR
        };
        Socket( int fd, sockaddr_in addr );
        Socket( int fd );
        Socket ();
        ~Socket();

        void setSocket( int fd );
        void setRemoteAddr( sockaddr_in addr );

        int getSocket() const;
        sockaddr_in getRemoteAddr() const;

        bool poll( int timeout_sec =0, int timeout_usec =0 );

        bool writeBuffer( void* buffer, size_t size );

        bool setBlocking( bool );
        bool isOpen() const;
        STATE getState() const { return state; }
        void setState( STATE s ) { state =s; }
        void close();

    protected:
        int sockfd;
        sockaddr_in remote_addr;
        STATE state;

        size_t block_size;
};

class ClientSocket : public Socket {
    public:
        ClientSocket( int fd, sockaddr_in addr );
        ~ClientSocket();

        socketbuf* getBuffer() { return &sockbuf; }

    private:
        socketbuf sockbuf;

};

class ListenSocket : public Socket {
    public:
        ListenSocket();
        ~ListenSocket();

        bool bind( int portnum );
        bool listen();

        ClientSocket* accept();

        void close();

    private:
        sockaddr_in listen_addr;
};


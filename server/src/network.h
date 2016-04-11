#pragma once
#include <sys/types.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <stdint.h>

#include "socketbuf.h"

class ClientSocket {
    public:
        ClientSocket( int fd, sockaddr_in addr );
        ~ClientSocket();

        bool isOpen( ) const { return open; }
        void close( );

        socketbuf* getBuffer() { return &sockbuf; }

    private:
        int sockfd;
        sockaddr_in client_addr;
        bool open;
        socketbuf sockbuf;

};

class ListenSocket {
    public:
        ListenSocket();
        ~ListenSocket();

        bool bind( int portnum );
        bool listen();

        ClientSocket* accept();

        void close();

    private:
        int sockfd;
        sockaddr_in listen_addr;
};

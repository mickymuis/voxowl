#include "network.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/types.h> 
#include <sys/socket.h>
#include <netinet/in.h>

// Implementation of ClientSocket

ClientSocket::ClientSocket( int fd, sockaddr_in addr ) : sockfd( fd ), client_addr( addr ), open( true ) {
    sockbuf.set_socket( fd );
}

ClientSocket::~ClientSocket() {
    close();
}

void
ClientSocket::close() {
    open =false;
    ::close( sockfd );
}

// Implementation of ListenSocket

ListenSocket::ListenSocket() {
    bzero( (char*) &listen_addr, sizeof( listen_addr ) );
}

ListenSocket::~ListenSocket() {
    close( );
}

bool 
ListenSocket::bind( int portnum ) {
    sockfd = socket( AF_INET, SOCK_STREAM, 0 );
    if( sockfd < 0 )
        return false;
    listen_addr.sin_family =AF_INET;
    listen_addr.sin_addr.s_addr =INADDR_ANY;
    listen_addr.sin_port =htons( portnum );
    if( ::bind( sockfd, (struct sockaddr*)&listen_addr, sizeof( listen_addr ) ) < 0 )
        return false;
    return true;
}

bool 
ListenSocket::listen() {
    return( ::listen( sockfd, 5 ) == 0 );
}

ClientSocket* 
ListenSocket::accept() {
    sockaddr_in client_addr;
    socklen_t client_addr_len = sizeof( client_addr );
    int fd =::accept( sockfd, (struct sockaddr*)&client_addr, &client_addr_len );
    if( fd < 0 )
        return 0;
    
    return new ClientSocket( fd, client_addr );
}

void 
ListenSocket::close() {
    ::close( sockfd );
}

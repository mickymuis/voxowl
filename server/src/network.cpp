#include "network.h"
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/types.h> 
#include <sys/socket.h>
#include <netinet/in.h>
#include <fcntl.h>

#define LISTEN_SOCKET_BACKLOG 4

// Socket implementation

Socket::Socket( int fd, sockaddr_in addr ) 
    : sockfd( fd ), remote_addr( addr )
{}

Socket::Socket( int fd ) 
    : sockfd( fd ) 
{}

Socket::Socket () {}

Socket::~Socket() {
    close();
}

void 
Socket::setSocket( int fd ) {
    sockfd =fd;
}

void 
Socket::setRemoteAddr( sockaddr_in addr ) {
    remote_addr =addr;
}

int 
Socket::getSocket() const {
    return sockfd;
}

sockaddr_in 
Socket::getRemoteAddr() const {
    return remote_addr;
}

bool 
Socket::poll(int timeout_sec, int timeout_usec ) {
    int result;
    struct timeval t;
    t.tv_sec =(long)timeout_sec;
    t.tv_usec =(long)timeout_usec;

    fd_set rdfd;
    FD_ZERO( &rdfd );
    FD_SET( sockfd, &rdfd );

    result =select( sockfd+1, &rdfd, 0, 0, &t );

    if( result == -1 ) {
        switch( errno ) {
            case EBADF:
                state =CLOSED;
                break;
            case EAGAIN:
                break;
            default:
                state =ERROR;
                std::cerr << strerror( errno ) << std::endl;
        }
    } else
        state =OPEN;

    return result == 1;
}

bool
Socket::setBlocking( bool blocking ) {
    int flags; 

    if ((flags = fcntl(sockfd, F_GETFL, 0)) < 0) 
        return false;

    if( !blocking ) {
        if (fcntl(sockfd, F_SETFL, flags | O_NONBLOCK) < 0) 
            return false; 
    } else {
        if (fcntl(sockfd, F_SETFL, flags & (~O_NONBLOCK)) < 0) 
            return false; 
    }
    return true;
}

bool 
Socket::isOpen() const {
    return state == OPEN;
}

void 
Socket::close() {
    ::close( sockfd );
    state =CLOSED;
}


// Implementation of ClientSocket

ClientSocket::ClientSocket( int fd, sockaddr_in addr ) : Socket( fd, addr ) {
    sockbuf.set_socket( fd );
}

ClientSocket::~ClientSocket() {
    close();
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
    return( ::listen( sockfd, LISTEN_SOCKET_BACKLOG  ) == 0 );
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



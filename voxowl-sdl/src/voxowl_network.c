/*
 * 
 *
 */

#include "voxowl_network.h"
#include <stdio.h>
#include <string.h>
#include <netdb.h>

#define VOXOWL_READLINE_INCREMENT 64
#define VOXOWL_READLINE_MAX 8192

int 
voxowl_connect_host( struct voxowl_socket_t* sock, const char* host, const char* port ) {
    struct addrinfo hints;
    struct addrinfo *result, *rp;
    int sfd, s, j;
    size_t len;

    /* Obtain address(es) matching host/port */
    memset(&hints, 0, sizeof(struct addrinfo));
    hints.ai_family = AF_UNSPEC;    /* Allow IPv4 or IPv6 */
    hints.ai_socktype = SOCK_STREAM;
    hints.ai_flags = 0;
    hints.ai_protocol = 0;          /* Any protocol */

    s = getaddrinfo(host, port, &hints, &result);
    if (s != 0) {
        fprintf(stderr, "getaddrinfo: %s\n", gai_strerror(s));
        return -1;
    }

    for (rp = result; rp != NULL; rp = rp->ai_next) {
        sfd = socket(rp->ai_family, rp->ai_socktype,
        rp->ai_protocol);
        if (sfd == -1)
            continue;

        if (connect(sfd, rp->ai_addr, rp->ai_addrlen) != -1)
            break;                  /* Success */

        close(sfd);
    }

    if (rp == NULL) {               /* No address succeeded */
        fprintf(stderr, "Could not connect\n");
        return -1;
    }

    freeaddrinfo(result);           /* No longer needed */

    /* Next, set the file descriptor to asynchronous mode */
    int flags; 

    if ((flags = fcntl(sfd, F_GETFL, 0)) < 0) {
        close( sfd );
        fprintf( stderr, "%s\n", strerror( errno ) );
        return -1;
    }


    if (fcntl(sfd, F_SETFL, flags | O_NONBLOCK) < 0) {
        close( sfd );
        fprintf( stderr, "%s\n", strerror( errno ) );
        return -1;
    }

    /* Lastly, obtain the socket's internal buffer size */

    unsigned int m = sizeof( sock->block_size );
    getsockopt( sfd, SOL_SOCKET, SO_SNDBUF, (void *)&sock->block_size, &m );

    sock->sockfd =sfd;
    sock->is_open =true;

    return 0;
}


int
voxowl_disconnect( struct voxowl_socket_t* sock ) {
    if( sock->is_open ) {
        voxowl_sendline( sock, "quit", 4 );
        close( sock->sockfd );
    }

    sock->is_open =false;

    return 0;
}

int
voxowl_poll_incoming( struct voxowl_socket_t* sock ) {
    int result;
    struct timeval t;
    t.tv_sec =0L;
    t.tv_usec =0L;

    fd_set rdfd;
    FD_ZERO( &rdfd );
    FD_SET( sock->sockfd, &rdfd );

    result =select( sock->sockfd+1, &rdfd, 0, 0, &t );

    if( result == -1 ) {
        switch( errno ) {
            case EBADF:
                sock->is_open =false;
                break;
            default:
                fprintf( stderr, "voxowl_poll_incoming(): %s\n", strerror( errno ) );
        }
        return -1;
    }

    return result == 1;
}

int 
voxowl_peek_pktmode( struct voxowl_socket_t* sock, int* mesg_type ) {
    if( !sock->is_open )
        return -1;

    uint32_t peek;
    if( recv( sock->sockfd, (void*)&peek, 4, MSG_PEEK ) != 4 ) {
        return 0;
    }
    
    fprintf( stderr, "Magic number: %x match %d\n", peek, peek == VOXOWL_PACKET_MAGIC );

    if( peek == VOXOWL_PACKET_MAGIC )
        *mesg_type = VOXOWL_MODE_DATA;
    else
        *mesg_type = VOXOWL_MODE_CHAR;
    return 1;
}

int 
voxowl_readline( struct voxowl_socket_t* sock, char **str ) {
    size_t str_len =VOXOWL_READLINE_INCREMENT;
    *str =malloc( str_len );
    size_t offs =0;

    while( true ) {

        if( str_len <= offs + 1 ) {
            str_len +=VOXOWL_READLINE_INCREMENT;
            if( str_len > VOXOWL_READLINE_MAX ) {
                free( *str );
                fprintf( stderr, "voxowl_readline(): max buffer size reached\n" );
                return -1;
            }
            *str =realloc( *str, str_len );
        }

        if( voxowl_read( sock, (void*)(*str + offs), 1 ) != 1 ) {
            free( str );
            return -1;
        }
        
        if( (*str)[offs] == '\n' ) {
            (*str)[offs+1] = '\0';
            break;
        }

        offs++;

    }
    return str_len;
}

int 
voxowl_read_frame_header( struct voxowl_socket_t* sock, struct voxowl_frame_header_t* buffer ) {
    return voxowl_read( sock, (void*)buffer, sizeof( struct voxowl_frame_header_t ) );
}

int 
voxowl_read( struct voxowl_socket_t* sock, void* buffer, size_t size ) {

    if( !sock->is_open )
        return -1;
    
    ssize_t result;
    struct timeval t;
    t.tv_sec =(long)VOXOWL_TIMEOUT_SEC;
    t.tv_usec =0L;
    fd_set rdfd;
    FD_ZERO( &rdfd );
    FD_SET( sock->sockfd, &rdfd );

    size_t offs =0;
    size_t len =0;
    while( offs < size ) {

        if( size - offs < sock->block_size )
            len = size - offs;
        else
            len = sock->block_size;

        // Wait until timeout or until data is ready to be read
        result =select( sock->sockfd+1, &rdfd, 0, 0, &t );
        if( result == 0 ) {
            // Timeout
            return -1;
        }

        if( result > 0 )
            result =recv( sock->sockfd, (char*)buffer + offs, len, MSG_DONTWAIT );
        
        if( result < 0 ) {
            fprintf( stderr, "voxowl_read(): %s\n", strerror( errno ) );
            return -1;
        }
        offs += (size_t)result;
    }
    return offs;
}

int 
voxowl_sendline( struct voxowl_socket_t* sock, const char *line, size_t len ) {
    if( voxowl_send( sock, (void*)line, len ) != len )
        return -1;

    char *eot ="\n";
    if( voxowl_send( sock, (void*)eot, 1 ) != 1 )
        return -1;
    return 0;
}

int 
voxowl_send( struct voxowl_socket_t* sock, void *buffer, size_t size ) {
    if( !sock->is_open )
        return -1;
    
    ssize_t result;
    fd_set sdfd;
    FD_ZERO( &sdfd );
    FD_SET( sock->sockfd, &sdfd );

    size_t offs =0;
    size_t len =0;
    while( offs < size ) {

        if( size - offs < sock->block_size )
            len = size - offs;
        else
            len = sock->block_size;

        // Block until the buffer is empty
        result =select( sock->sockfd+1, 0, &sdfd, 0, 0 );
        if( result > 0 )
            result =send( sock->sockfd, (char*)buffer + offs, len, MSG_DONTWAIT );
        
        if( result < 0 ) {
            fprintf( stderr, "voxowl_send(): %s\n", strerror( errno ) );
            return -1;
        }
        offs += (size_t)result;
    }
    return offs;
}

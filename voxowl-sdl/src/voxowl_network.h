/*
 * 
 *
 */

#ifndef VOXOWL_NETWORK_H
#define VOXOWL_NETWORK_H

#include "voxowl.h"

#include <sys/socket.h>
#include <sys/types.h>
#include <sys/time.h>
#include <sys/select.h>
#include <unistd.h>
#include <fcntl.h>
#include <errno.h>
#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <stdbool.h>

#define VOXOWL_TIMEOUT_SEC 15

enum voxowl_packet_mode_t {
  VOXOWL_MODE_CHAR,
  VOXOWL_MODE_DATA
};

struct voxowl_socket_t {
  int sockfd;
  bool is_open;
  size_t block_size;
};

int voxowl_connect_host( struct voxowl_socket_t* sock, const char* host, const char* port );
int voxowl_disconnect( struct voxowl_socket_t* sock );

int voxowl_poll_incoming( struct voxowl_socket_t* sock );

int voxowl_peek_pktmode( struct voxowl_socket_t* sock, int* mesg_type );
int voxowl_readline( struct voxowl_socket_t* sock, char **str );
int voxowl_read_frame_header( struct voxowl_socket_t* sock, struct voxowl_frame_header_t* buffer );
int voxowl_read( struct voxowl_socket_t* sock, void* buffer, size_t size );

int voxowl_send( struct voxowl_socket_t* sock, void *buffer, size_t size );
int voxowl_sendline( struct voxowl_socket_t* sock, const char *line, size_t len );

#endif

#include "packetbuffer.h"
#include "server.h"
#include "network.h"

bool pbufferMain( PacketBuffer* buffer ) {

    while( !buffer->stop ) {
        buffer->dispatchIncoming();
        buffer->dispatchOutgoing();
    }
    return true;
}

PacketBuffer::PacketBuffer( ) 
    : recv_func(0), stop (false) {

}

PacketBuffer::~PacketBuffer() {

}

void PacketBuffer::enqueue( const Packet& packet ) {
    if( packet.direction == Packet::RECEIVE ) {  
            std::lock_guard<std::mutex> recv_lock( recv_queue_mutex );
            recv_queue.push( packet );
    }
    else if( packet.direction == Packet::SEND ) {
            std::lock_guard<std::mutex> send_lock( send_queue_mutex );
            send_queue.push( packet );
    }
}
Packet 
PacketBuffer::popRecvQueue() {
    Packet p_recv;
    memset( &p_recv, 0, sizeof( Packet ) );

    std::lock_guard<std::mutex> recv_lock( recv_queue_mutex );
    
    if( !recv_queue.empty() ) {
        p_recv =recv_queue.front();
        recv_queue.pop();
    }
    return p_recv;
}

Packet 
PacketBuffer::popSendQueue() {

    Packet p_send;
    memset( &p_send, 0, sizeof( Packet ) );

    std::lock_guard<std::mutex> send_lock( send_queue_mutex );
    
    if( !send_queue.empty() ) {
        p_send =send_queue.front();
        send_queue.pop();
    }
    return p_send;
}

bool 
PacketBuffer::hasIncoming() const {
    return !recv_queue.empty();
}

bool 
PacketBuffer::hasOutgoing() const {
    return !send_queue.empty();
}

bool 
PacketBuffer::dispatchIncoming() {

    Packet p_recv =popRecvQueue();

    /* We have an incoming packet, process it */
    if( p_recv.direction != Packet::NONE ) {
        if( recv_func )
            (*recv_func)(p_recv);
        p_recv.cleanup();

        return true;
    }
    return false;
}

bool 
PacketBuffer::dispatchOutgoing() {

    Packet p_send =popSendQueue();

    /* We have an outgoing packet, send it */
    if( p_send.direction != Packet::NONE ) {
        if( send_func )
            (*send_func)(p_send);
        p_send.cleanup();
    }
    return false;
}


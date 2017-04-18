#pragma once

#include "../core/object.h"
#include "../core/voxowl.h"

#include <string>
#include <queue>
#include <mutex>

class Parser;
class Connection;
class PacketBuffer;

class Packet {
    public:
        typedef std::queue<Packet> Queue;
        enum DIRECTION {
            NONE =0,
            RECEIVE,
            SEND
        } direction;
        
        enum MODE {
            CHAR,
            DATA
        } mode;

        size_t size;
        Connection* connection;
        void *payload;
        bool own_payload;
//        struct voxowl_data_header_t data_header;

        void cleanup() { if( own_payload ) free(payload); }
};

bool pbufferMain( PacketBuffer* );

typedef void(*recv_func_t)( const Packet& );
typedef void(*send_func_t)( const Packet& );

class PacketBuffer {
    public:
        PacketBuffer();
        ~PacketBuffer();

        void setIncomingPacketHandler( recv_func_t f ) { recv_func =f; }
        void setOutgoingPacketHandler( send_func_t f ) { send_func =f; }
        
        void enqueue( const Packet& );
        Packet popRecvQueue();
        Packet popSendQueue();

        bool hasIncoming() const;
        bool hasOutgoing() const;

        bool dispatchIncoming();
        bool dispatchOutgoing();

        void stopThread() { stop =true; }

    private:
        friend bool pbufferMain( PacketBuffer* );
        recv_func_t recv_func;
        send_func_t send_func;

        // FIXME should be std::atomic_bool
        bool stop;
        Parser* parser;

        Packet::Queue send_queue;
        Packet::Queue recv_queue;

        std::mutex send_queue_mutex;
        std::mutex recv_queue_mutex;
};

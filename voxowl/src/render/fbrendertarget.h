#pragma once

#include "framebuffer.h"

/**
 * Implements a render target (framebuffer) as an image on the local filesystem 
 */
class FBImageTarget : public CompressedFramebuffer {
    public:
        FBImageTarget( const char* name, Object* parent =0 );
        ~FBImageTarget();

        void setPath( const std::string& p ) { path =p; }
        std::string getPath() const { return path; }

        virtual bool synchronize();
            
    private:
        std::string path;

};

/**
 * Implements a render target (framebuffer) on a remote machine 
 */
class FBRemoteTarget : public CompressedFramebuffer {
    public:
        FBRemoteTarget( const char* name, Object* parent =0 );
        ~FBRemoteTarget();

        virtual void reinitialize();
        virtual bool synchronize();

    private:
        void* header_begin;

};


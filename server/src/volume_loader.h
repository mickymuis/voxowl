#pragma once

#include "volume.h"
#include <string>

/*! Provides a high-level means of loading supported data types from disk */
class VolumeLoader : public Volume { 
    public:
        VolumeLoader( const char* name, Object* parent =0 );
        ~VolumeLoader();

        VolumeStorageDetail* storageDetail() { return m_detail; }

        bool open( std::string& path );
        void close();

        bool isOpen() const { return m_isOpen; }
        std::string errorString() const { return err_str; }
        ivec3_32_t size() const;
        
        virtual Variant callMeta( const std::string& method, const Variant::list& args );

    private:
        bool setError( bool, const std::string& );
        
        VolumeStorageDetail *m_detail;
        std::string err_str;
        bool m_isOpen;

};




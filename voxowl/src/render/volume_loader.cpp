#include "volume_loader.h"
#include "volume_detail.h"

#include <stdio.h>
#include <string.h>

#include <voxel.h>
#include <svmipmap.h>
#include <voxelmap.h>

class VoxelmapLoaderStorageDetail : public VolumeVoxelmapStorageDetail {
    public:
        VoxelmapLoaderStorageDetail();
        virtual ~VoxelmapLoaderStorageDetail() { close(); }

        bool open( std::string &path );
        void close();

    protected:
        voxelmap_t m_voxelmap;
        voxelmap_mapped_t voxelmap_mapped;
};

VoxelmapLoaderStorageDetail::VoxelmapLoaderStorageDetail() {
    bzero( &voxelmap_mapped, sizeof( voxelmap_mapped_t ) );
    bzero( &m_voxelmap, sizeof( voxelmap_t ) );
    voxelmap = &m_voxelmap;
}

bool 
VoxelmapLoaderStorageDetail::open( std::string &path ) {
    if( voxelmapOpenMapped( &voxelmap_mapped, path.c_str() ) == -1 ) {
        return false;
    }
    voxelmapFromMapped( &m_voxelmap, &voxelmap_mapped );
    return true;
}

void
VoxelmapLoaderStorageDetail::close() {
    voxelmapUnmap( &voxelmap_mapped );
}

//

class SVMipmapLoaderStorageDetail : public VolumeSVMipmapStorageDetail {
    public:
        SVMipmapLoaderStorageDetail();
        virtual ~SVMipmapLoaderStorageDetail() { close(); }

        bool open( std::string &path );
        void close();

    protected:
        svmipmap_t svmipmap_mapped;
};

SVMipmapLoaderStorageDetail::SVMipmapLoaderStorageDetail() {
    bzero( &svmipmap_mapped, sizeof( svmipmap_t ) );
    svmm =&svmipmap_mapped;
}

bool 
SVMipmapLoaderStorageDetail::open( std::string &path ) {
    if( svmmOpenMapped( &svmipmap_mapped, path.c_str() ) == -1 ) {
        return false;
    }
    return true; 
}

void
SVMipmapLoaderStorageDetail::close() {
    svmmFree( &svmipmap_mapped );
}


//

VolumeLoader::VolumeLoader( const char* name, Object* parent )
    : Volume( name, parent ), m_detail( 0 ), m_isOpen( false ) {
    addMethod( "open" );
    addMethod( "close" );
}

VolumeLoader::~VolumeLoader() {
    if( m_detail ) delete m_detail;
}

bool 
VolumeLoader::open( std::string& path ) {
    // We need to peek into the file to determine its precise format
   
    newConfiguration();

    if( isOpen() )
        close();
    
    enum {
        VOXELMAP,
        SVMM
    } filetype;

    FILE* f =fopen( path.c_str(), "r" );
    if( !f ) {
        return setError( true, "Cannot open '" + path + "' for reading" );
    }
    char buf[2];
    fread( buf, 1, 2, f );
    if( buf[0] == VOXELMAP_MAGIC1 && buf[1] == VOXELMAP_MAGIC2 )
        filetype =VOXELMAP;
    else if( buf[0] == SVMM_MAGIC1 && buf[1] == SVMM_MAGIC2 )
        filetype =SVMM;
    else {
        return setError( true, path + " is not a Voxowl file format." );
    }
    fclose( f );

    switch( filetype ) {
        case VOXELMAP: {
            VoxelmapLoaderStorageDetail *d =new VoxelmapLoaderStorageDetail();
            if( !d->open( path ) ) {
                delete d;
                return setError( true, "Error opening voxelmap " + path );
            }

            m_detail =d;
            break;               
        }
        case SVMM: {
            SVMipmapLoaderStorageDetail *d =new SVMipmapLoaderStorageDetail();
            if( !d->open( path ) ) {
                delete d;
                return setError( true, "Error opening svmm " + path );
            }

            m_detail =d;
            break;
        }
    }

    m_isOpen =true;
    return setError( false, "" );
}

void 
VolumeLoader::close() {
    newConfiguration();

    if( m_detail ) {
        delete m_detail;
        m_detail =0;
        m_isOpen =false;
    }
}

ivec3_32_t 
VolumeLoader::size() const {
    if( m_detail )
        return m_detail->size();
    return ivec3_32(0);
}

Variant 
VolumeLoader::callMeta( const std::string& method, const Variant::list& args ) {
    if( method == "close" ) {
        close();
        return Variant();
    }
    else if( method == "open" ) {
        if( args.size() != 1 )
            return Variant( "Invalid arguments" );
        std::string path =args[0].toString();
        open( path );
        return errorString();
    }
    return Object::callMeta( method, args );
}

bool 
VolumeLoader::setError( bool err, const std::string& str ) {
    if( err )
        err_str =str;
    else
        err_str ="";
    return !err;
}

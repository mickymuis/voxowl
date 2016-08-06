#include "mengersponge.h"

#include <math.h>
#include <string.h>
#include "glm/vec3.hpp"
#include <svmipmap.h>
#include <voxelmap.h>

void 
menger( voxelmap_t* V, glm::ivec3 size, glm::ivec3 cube, glm::ivec3 offset ) {
    uint32_t step = cube.x / 3;
    if( step < 1 )
        return;
    for( int x =0; x < 3; x++ )
        for( int y =0; y < 3; y++ )
            for( int z =0; z < 3; z++ ) {
                glm::ivec3 offs = offset + glm::ivec3( x * step, y * step, z * step );
                glm::ivec3 new_cube( step, step, step );
                // middle element
                if( ( z == 1 && ( y == 1 || x == 1 ) )
                    || ( x == 1 && y == 1 ) ) {
                    for( uint16_t i = offs.x; i < offs.x + step; i++ )
                        for( uint16_t j = offs.y; j < offs.y + step; j++ )
                            for( uint16_t k = offs.z; k < offs.z + step; k++ ) {
                                glm::vec4 color =voxelmapUnpack( V, ivec3_32(i,j,k ) );
                                color.a =0.f;
                                voxelmapPack(  V, ivec3_32( i, j, k ), color );
                            }
                }
                // corner element, expand recursively
                else
                    menger( V, size, new_cube, offs );
            }
}

MengerSponge::MengerSponge( const char *name, Object *parent ) 
    //: VolumeSVMipmap( name, parent ) {
    : VolumeVoxelmap( name, parent ) {
    depth =1;
    mode =COLORS_RGBA;
    //memset( &volume, 0, sizeof( voxelmap_t ) );
    volume.data =NULL;
    //makeSponge();
    //setSVMipmap( &volume_svmm );
    setVoxelmap( &volume );
}

MengerSponge::~MengerSponge() {
    if( volume.data ) voxelmapFree( &volume );
}

void 
MengerSponge::setDepth( int d ) {
    if( d < 1 )
        return;
    depth =d;
    makeSponge();
}

void
MengerSponge::setMode( int m ) {
    mode =m;
    makeSponge();
}

static float
randomf( float min, float max ) {
    return min + static_cast <float> (rand()) /( static_cast <float> (RAND_MAX/(max-min)));
}
void 
MengerSponge::makeSponge() {
    if( volume.data ) 
        voxelmapFree( &volume );

/*    voxelmap_mapped_t vm;
    if( voxelmapOpenMapped( &vm, "/home/s1407937/data/test.vxwl" ) == -1 ) {
        perror( "" );
        fprintf( stderr, "Could not open voxelmap\n" );
    }

    voxelmapFromMapped( &volume, &vm );

    return;*/

    size_t s = (int)pow(3, depth );

    ivec3_32_t size =ivec3_32( s );
    
    voxelmapCreate( &volume, VOXEL_RGBA_UINT32, s, s, s );
//    voxelmapCreate( &volume, VOXEL_INTENSITY_UINT8, s, s, s );

    for( int x=0; x < size.x; x++ )
        for( int y=0; y < size.y; y++ )
            for( int z=0; z < size.z; z++ ) {
                   float p =(float)(( size.x/2 - abs(x - size.x / 2.f) ) / (size.x / 2.f))
                    * (float)(( size.y/2 - abs(y - size.y / 2.f) ) / (size.y / 2.f))
                    * (float)(( size.z/2 - abs(z - size.z / 2.f) ) / (size.z / 2.f));
                    p *= randomf( .5f, 1.f );
                    float c =(p > 0.05f);
//                    voxelmapPack( &volume, ivec3_32(x,y,z), glm::vec4(1,1,1,c) );

                voxelmapPack( &volume, ivec3_32(x,y,z), glm::vec4(
                    (float)x / (float)(size.x-1),
                    (float)y / (float)(size.y-1),
                    (float)z / (float)(size.z-1),
                    1.f ));
            }

//    uint32_t white =0xffffffff;
//    uint8_t white =0xff;
//    voxelmapFill( &volume, &white );
    menger( &volume, glm_ivec3_32(size), glm_ivec3_32(size), glm::ivec3(0) );
    
  /*  svmm_encode_opts_t opts;
    svmmSetOpts( &opts, &volume, 75 );
//    opts.bitmapBaselevel =false;
//    opts.rootwidth =64;
//    opts.blockwidth =2;
    bzero( &volume_svmm, sizeof( svmipmap_t ) );
    svmmEncode( &volume_svmm, &volume, opts );*/

//    svmmDecode( &volume, &volume_svmm );
    
//    svmmTest( &volume, 90 );
}

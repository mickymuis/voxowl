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
    for( size_t x =0; x < 3; x++ )
        for( size_t y =0; y < 3; y++ )
            for( size_t z =0; z < 3; z++ ) {
                if( size.x == cube.x )
                    printf( "[%zu]\n", x*y*z );
                glm::ivec3 offs = offset + glm::ivec3( x * step, y * step, z * step );
                glm::ivec3 new_cube( step, step, step );
                // middle element
                if( ( z == 1 && ( y == 1 || x == 1 ) )
                    || ( x == 1 && y == 1 ) ) {
                   // #pragma omp parallel for
                    for( size_t i = offs.x; i < offs.x + step; i++ )
                        for( size_t j = offs.y; j < offs.y + step; j++ )
                            for( size_t k = offs.z; k < offs.z + step; k++ ) {
                                //glm::vec4 color =voxelmapUnpack( V, ivec3_32(i,j,k ) );
                                //color.a =0.f;
                                voxelmapPack(  V, ivec3_32( i, j, k ), glm::vec4(0) );
                                //((uint32_t*)V->data)[size.x * size.y * k + size.x * j + i] &= ~(uint32_t)0xff;
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
//    if( volume.data ) voxelmapFree( &volume );
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

    size_t s = pow(3, depth );

    ivec3_32_t size =ivec3_32( s );

/*    printf( "Voxelmap %d size %zu bytes\n", size.x, voxelmapSize( VOXEL_INTENSITY_UINT8, size ) );

    voxelmap_mapped_t vol_disk;
    if( voxelmapCreateMapped( &vol_disk, "/local/s1407937/data/sphere128.vxwl", VOXEL_INTENSITY_UINT8, size ) == -1 ) {
        printf( "Error creating mmapped voxelmap\n" );
        return;
    }
    voxelmapFromMapped( &volume, &vol_disk );*/
    voxelmapCreate( &volume, VOXEL_DENSITY_UINT8, s, s, s );
//
    uint32_t black =0x0;
//    uint8_t white =0xff;
    voxelmapFill( &volume, &black );

    // FOR SPHERE
    float maxDist =glm::min(size.x, glm::min( size.y, size.z ) ) / 2;
    float minDist =(float) maxDist * 0.8;
    glm::vec3 center( size.x / 2, size.y / 2, size.z / 2 );

    const float step = 1.f / size.x;

    for( int z=0; z < size.z; z++ ) {
        printf( "(%d) \n", z );
        for( int y=0; y < size.y; y++ )
//            #pragma omp parallel for
            for( int x=0; x < size.x; x++ ) {
/*                   float p =(float)(( size.x/2 - abs(x - size.x / 2.f) ) / (size.x / 2.f))
                    * (float)(( size.y/2 - abs(y - size.y / 2.f) ) / (size.y / 2.f))
                    * (float)(( size.z/2 - abs(z - size.z / 2.f) ) / (size.z / 2.f));
                    p *= randomf( .5f, 1.f );
                    float c =(p > 0.05f);
                    voxelmapPack( &volume, ivec3_32(x,y,z), glm::vec4(1,1,1,c) );*/
/*
                float dist =glm::distance( center, glm::vec3( x, y, z ) );
                if( dist <= maxDist  )
                //if( dist <= maxDist && dist >= minDist )
                voxelmapPack( &volume, ivec3_32(x,y,z), glm::vec4(
                    (float)x / (float)(size.x-1),
                    (float)y / (float)(size.y-1),
                    (float)z / (float)(size.z-1),
                    1.f ));
//                    voxelmapPack( &volume, ivec3_32(x,y,z), glm::vec4(1,1,1,1) );
                else
                    voxelmapPack( &volume, ivec3_32(x,y,z), glm::vec4(0) );*/

               /* int r,g,b;
                r = (x <= 2040 ? (x >> 3) : 0xff) << 24;
                g = (y <= 2040 ? (y >> 3) : 0xff) << 16;
                b = (z <= 2040 ? (z >> 3) : 0xff) << 8;

                ((uint32_t*)volume.data)[(size_t)size.x * (size_t)size.y * (size_t)z + (size_t)size.x * (size_t)y + (size_t)x] 
                    = r | g | b | 0xff;*/


                voxelmapPack( &volume, ivec3_32(x,y,z), glm::vec4(
                    (float)x * step,
                    (float)y * step,
                    (float)z * step,
                    1.f ));


            }
    }
    menger( &volume, glm_ivec3_32(size), glm_ivec3_32(size), glm::ivec3(0) );

    
/*    svmm_encode_opts_t opts;
    svmmSetOpts( &opts, &volume, 75 );
    opts.bitmapBaselevel =false;
//    opts.rootwidth =8;
//    opts.blockwidth =2;
    bzero( &volume_svmm, sizeof( svmipmap_t ) );
    svmmEncode( &volume_svmm, &volume, opts );*/

//    svmmDecode( &volume, &volume_svmm );
    
    svmmTest( &volume, 90 );
//    voxelmapUnmap( &vol_disk );
}

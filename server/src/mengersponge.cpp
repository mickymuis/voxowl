#include "mengersponge.h"

#include <math.h>
#include <string.h>
#include "glm/vec3.hpp"

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
                    for( uint32_t i = offs.x; i < offs.x + step; i++ )
                        for( uint32_t j = offs.y; j < offs.y + step; j++ )
                            for( uint32_t k = offs.z; k < offs.z + step; k++ ) {
                                glm::vec4 color =voxelmapUnpack( V, glm::ivec3(i,j,k ) );
                                color.a =0.f;
                                voxelmapPack(  V, glm::ivec3( i, j, k ), color );
                            }
                }
                // corner element, expand recursively
                else
                    menger( V, size, new_cube, offs );
            }
}

MengerSponge::MengerSponge( const char *name, Object *parent ) 
    : Volume( name, parent ) {
    depth =1;
    mode =COLORS_RGBA;
    //memset( &volume, 0, sizeof( voxelmap_t ) );
    volume.data =NULL;
    makeSponge();
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

void 
MengerSponge::makeSponge() {
    if( volume.data ) 
        voxelmapFree( &volume );

    size_t s = (int)pow(3, depth );

    glm::ivec3 size( s );
    
    voxelmapCreate( &volume, VOXEL_RGBA_UINT32, s, s, s );

    for( int x=0; x < size.x; x++ )
        for( int y=0; y < size.y; y++ )
            for( int z=0; z < size.z; z++ ) {
                voxelmapPack( &volume, glm::ivec3(x,y,z), glm::vec4(
                    (float)x / (float)(size.x-1),
                    (float)y / (float)(size.y-1),
                    (float)z / (float)(size.z-1),
                    1.f ));
            }
    menger( &volume, size, size, glm::ivec3(0) );
}

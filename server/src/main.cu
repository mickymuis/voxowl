#include <stdio.h>
#include "platform.h"
#include "volumetric.h"

#define float4 glm::vec4

typedef Volumetric<float> VolFloat;

__global__ void
simpleAdd( VolFloat *a, VolFloat *b, VolFloat *c, int size ) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if( idx < size)
       c->at(0,0,idx) = a->at(0,0,idx) + b->at(0,0,idx);
}

int 
main( int argc, char **argv ) {

    VolFloat a(16,16,16,2.0f);
    VolFloat b(16,16,16,1.0f);
    VolFloat c(16,16,16);

    VolFloat *dev_a, *dev_b, *dev_c;

    dev_a =VolFloat::createOnDevice( 16, 16, 16 );
    dev_b =VolFloat::createOnDevice( 16, 16, 16 );
    dev_c =VolFloat::createOnDevice( 16, 16, 16 );
    
    VolFloat::copyToDevice( dev_a, &a );
    VolFloat::copyToDevice( dev_b, &b );

    simpleAdd<<< 0, a.size() >>>( dev_a, dev_b, dev_c, a.size() );

    VolFloat::copyFromDevice( &c, dev_c );

    for( int x =0; x < c.xSize(); x++ )
    {
        for( int y =0; y < c.ySize(); y++ ) {
            for( int z =0; z < c.zSize(); z++ ) {
                printf( "%f ", c.at(x,y,z) );
            }
            printf( "\n" );
        }
        printf( "\n" );
    }

   VolFloat::deleteFromDevice( dev_a );
   VolFloat::deleteFromDevice( dev_b );
   VolFloat::deleteFromDevice( dev_c );

   return 0;

}

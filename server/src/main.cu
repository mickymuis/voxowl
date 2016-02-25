#include <cuda_runtime.h>
#include <stdio.h>
#include <helper_math.h>

__global__ void
simpleAdd( float4 *a, float4 *b, float4 *c, int size ) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if( idx < size)
        c[idx] = a[idx] + b[idx];
}

int 
main( int argc, char **argv ) {

    float4 a[1024];
    float4 b[1024];
    float4 c[1024];

    for( int i =0; i < 1024; i++ ) { 
        a[i].x = b[i].x = i / 1024;
        a[i].y = b[i].y = i / 1024;
        a[i].z = b[i].z = i / 1024;
        a[i].w = b[i].w = i / 1024;
    }

    float4 *dev_a, *dev_b, *dev_c;
    cudaMalloc( &dev_a, 1024 * sizeof( float4 ) );
    cudaMalloc( &dev_b, 1024 * sizeof( float4 ) );
    cudaMalloc( &dev_c, 1024 * sizeof( float4 ) );
    cudaMemcpy( dev_a, a, 1024 * sizeof( float4 ), cudaMemcpyHostToDevice );
    cudaMemcpy( dev_b, b, 1024 * sizeof( float4 ), cudaMemcpyHostToDevice );

    simpleAdd<<< 0, 1024 >>>( dev_a, dev_b, dev_c, 1024 );

    cudaMemcpy( c, dev_c, 1024 * sizeof( float4 ), cudaMemcpyDeviceToHost );

    for( int i =0; i < 1024; i++ )
        printf( "%f, %f, %f, %f\n", c[i].x, c[i].y, c[i].z, c[i].w );

   return 0;

}

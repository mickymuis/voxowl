#include "testing.h"

#include <stdint.h>
#include <string.h>
#include <stdlib.h>
#include <thread>
#include <chrono>
#include <errno.h>
#include <iostream>
#include <list>
#include <cuda_runtime.h>

#include "platform.h"
#include "server.h"
#include "types.h"
#include "parser.h"
#include "packetbuffer.h"
#include "fbrendertarget.h"
#include "camera.h"
#include "raycast_cuda.h"
#include "mengersponge.h"
#include "volume_loader.h"
#include "performance_counter.h"

unsigned int
getCudaMemUsed() {
    size_t avail;
    size_t total;
    cudaMemGetInfo( &avail, &total );
    size_t used = total - avail;
    return used / 1048576UL;
}

class Test {
    public:
    Test() {
        frame_perf = PerformanceCounter::create( "frame", "Total render time", "fps" );
        raycast_perf = PerformanceCounter::create( "raycast", "Raycast kernel", "ms" );
        PerformanceCounter::create( "ssao", "SSAO kernel", "ms" );
        PerformanceCounter::create( "ssna", "SSNA kernel", "ms" );
        PerformanceCounter::create( "aa", "AA kernel", "ms" );
        PerformanceCounter::create( "lighting", "Lighting kernel", "ms" );
    }

    void  doSimulation( ) {
        for( int r =0; r < 64; r++ ) {
            camera->rotateAround( glm::vec3( 0.f, -1.f, 0.f ) );
            renderer->beginRender();
            renderer->synchronize();
        }
        for( int r =0; r < 10; r++ ) {
            camera->translate( glm::vec3( 0.f, 0.f, 1.f ) );
            renderer->beginRender();
            renderer->synchronize();
        }
        for( int r =0; r < 20; r++ ) {
            camera->translate( glm::vec3( 0.f, 0.f, -1.f ) );
            renderer->beginRender();
            renderer->synchronize();
        }
    }

    void doSVMMTest( int w, int r, int q, bool bc ) {
        printf( "\n*** w: %d, r: %d, q: %d, bc: %d ***\n\n", w, r, q, bc );

        svmipmap_t svmm;
        svmm_encode_opts_t opts;
        svmmSetOpts( &opts, voxelmap, q );

        opts.blockwidth =w;
        opts.rootwidth =r;
        opts.bitmapBaselevel =bc;
        if( w == 0 ) {
            opts.blockwidth =2;
            opts.shiftBlockwidth =true;
        }

        svmmEncode( &svmm, voxelmap, opts );
        volume->setSVMipmap( &svmm );
        renderer->setVolume( volume );

        doSimulation( );
        
        fprintf( log, "%d,\t%d,\t%d,\t%d,\t%f,\t%f,\t%f,\t%d,\t%d%,\t%d,\t%d\n",
                w, r, q, bc,
                raycast_perf->mean(), raycast_perf->min(), raycast_perf->max(), (int)frame_perf->mean(), 
                (int)(((float)svmm.data_size / (float)voxelmapSize( voxelmap )) * 100.f),
                getCudaMemUsed(), svmm.header.levels );
        PerformanceCounter::resetAll();

        svmmFree( &svmm );
        fflush( log );
    }

    void doBaselineTest( Volume* v ) {
        renderer->setVolume( v );
        doSimulation( );

        fprintf( log, "0,\t0,\t0,\t0,\t%f,\t%f,\t%f,\t%d,\t100%,\t%d,\t1\n",
                raycast_perf->mean(), raycast_perf->min(), raycast_perf->max(), 
                (int)frame_perf->mean(), getCudaMemUsed() );
        PerformanceCounter::resetAll();
    }

    RaycasterCUDA* renderer;
    Camera* camera;
    VolumeSVMipmap* volume;
    FILE* log;
    voxelmap_t* voxelmap;
    PerformanceCounter *frame_perf;
    PerformanceCounter *raycast_perf;
    int gpu_bare_usage;
};


int
testingMain( int argc, char** argv ) {

    Test test;

    /* Setup the environment */

    
    Object root("root");

    FBRemoteTarget fb( "framebuffer", &root );
    fb.setSize( 1920, 1080 );
    fb.setMode( CompressedFramebuffer::JPEG );
    fb.setPixelFormat( Framebuffer::RGB888 );
    fb.setAASamples( 1, 1 );
    fb.setClearColor( glm::vec4( .85f, .84f, .75f, 1.f ) );
    fb.reinitialize();

    Camera camera( "camera", &root );
    
    VolumeLoader loader( "volumeloader", &root );
    if( argc > 1 ) {
        std::string path( argv[1] );
        if( !loader.open( path ) )
            std::cerr << loader.errorString() << std::endl;
    } else
        return 0;

    VolumeVoxelmapStorageDetail* detail =dynamic_cast<VolumeVoxelmapStorageDetail*>(loader.storageDetail());
    if( !detail ) {
        fprintf( stderr, "File not a voxelmap\n" );
        return -1;
    }
    test.voxelmap =detail->getVoxelmap();
    VolumeSVMipmap volume( "volume", &root );
    test.volume =&volume;

    std::string log_path =std::string( argv[1] ) + ".log";
    test.log =fopen( log_path.c_str(), "w" );
    fprintf( test.log, "w \tr \tq \tbc \ttime \t\tmin \t\tmax \t\tfps \tratio \tmem \tdepth\n" );

    RaycasterCUDA renderer( "renderer", &root );
    renderer.setTarget( &fb );
    renderer.setCamera( &camera );
    renderer.enable( Renderer::FEATURE_SSAO );
    test.renderer =&renderer;
    test.camera =&camera;

    // Perform the baseline test
    //test.doBaselineTest( &loader );
    //test.doBaselineTest( &loader );


    // Perform SVMM test
    int n_r_values =3, n_w_values = 3, n_q_values = 2, n_bc_values = 2;
    int r_values[] = { 8, 32, 128 };
    int w_values[] = { 4, 8, 0 };
    int q_values[] = { 45, 80 };
    bool bc_values[] = { true, false };
    
    for( int r =0; r < n_r_values; r++ )
        for( int w =0; w < n_w_values; w++ )
            for( int q =0; q < n_q_values; q++ )
                for( int bc =0; bc < n_bc_values; bc++ ) {
                   test.doSVMMTest( 
                           w_values[w], 
                           r_values[r], 
                           q_values[q] , 
                           bc_values[bc] );
                }

    // Perform the 'SVO' test
    test.doSVMMTest( 2, 2, 99, false );

    PerformanceCounter::cleanup();

    fclose( test.log );
    return 0;
}


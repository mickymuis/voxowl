/*
 * 
 *
 */
 
#include <SDL2/SDL.h>
#include "voxowl_network.h"
#include <stdbool.h>

#define DEFAULT_WIDTH 800
#define DEFAULT_HEIGHT 600
#define USE_TURBOJPEG

#ifdef USE_TURBOJPEG
#include <turbojpeg.h>
#endif

SDL_Texture* _texture;

typedef struct frame_info_ {
    struct voxowl_frame_header_t header;
    uint32_t size;
    unsigned char *data;
    unsigned char *image;
    bool new;

} frame_info_t;

void
frame_init( frame_info_t* frame ) {
    frame->size =0;;
    frame->new =false;
    frame->data =0;
    frame->image =0;
}

void 
updateTexture( SDL_Renderer* render, frame_info_t *frame ) {
    int width =0, height =0;
    int color_components;
    if( frame->header.pixel_format == VOXOWL_PF_RGB888 )
        color_components =3;
    else
        return;

    if( _texture ) 
        SDL_QueryTexture( _texture, NULL, NULL, &width, &height);

    if( width != frame->header.width || height != frame->header.height ) {
        if( _texture )
            SDL_DestroyTexture( _texture );
        _texture = SDL_CreateTexture(render, SDL_PIXELFORMAT_RGB24, SDL_TEXTUREACCESS_STREAMING, frame->header.width, frame->header.height);
    }

#ifdef USE_TURBOJPEG
    static unsigned char* image =0;
    if( frame->header.fb_mode == VOXOWL_FBMODE_JPEG ) {

        int jpegSubsamp, width, height;

        tjhandle jpegDecompressor = tjInitDecompress();

        tjDecompressHeader2(jpegDecompressor, frame->data, frame->size, &width, &height, &jpegSubsamp);
        
        //printf( "decoding  width = %d, height = %d\n", width, height );
        unsigned char image[width*height*color_components];
        
        tjDecompress2(jpegDecompressor, frame->data, frame->size, image, width, 0/*pitch*/, height, TJPF_RGB, TJFLAG_FASTDCT);
        tjDestroy(jpegDecompressor);


        SDL_UpdateTexture( _texture, NULL, image, width*color_components );        
    } else
#endif
    {
        frame->image =frame->data;
        SDL_UpdateTexture( _texture, NULL, frame->image, frame->header.width*color_components );        
    }
    frame->new =false;    
}

void 
renderTexture( SDL_Renderer* render, SDL_Window* window ) {
    int width =0, height =0;
    if( _texture ) 
        SDL_QueryTexture( _texture, NULL, NULL, &width, &height);
    else
        return;

    SDL_RenderClear( render );
    SDL_SetWindowSize( window, width, height );
        
    SDL_RenderCopy( render, _texture, 0, 0 );
    SDL_RenderPresent( render );
}

int 
fetchIncoming( struct voxowl_socket_t *sock, frame_info_t *frame ) {
    int msg_type;
    if( voxowl_peek_pktmode( sock, &msg_type ) == -1 )
        return -1;
    
    if( msg_type == VOXOWL_MODE_CHAR ) {
        /* Text coded message from the server, just print it in the terminal */
        char *msg;
        if( voxowl_readline( sock, &msg ) == -1 )
            return -1;
        if( strlen( msg ) != 0 )
            printf( "server: %s\n", msg );
        free( msg );
    } 
    else if( msg_type == VOXOWL_MODE_DATA ) {
        /* There should be a frame buffered */
        struct voxowl_frame_header_t header;
        if( voxowl_read_frame_header( sock, &header ) == -1 )
            return -1;

        //fprintf( stderr, "Receiving frame (%u bytes) %dx%d\n", header.frame_size, header.width, header.height );
        
        frame->header =header;
        
        if( frame->size != header.frame_size ) {
            if( frame->data )
                free( frame->data );
            frame->data =malloc( header.frame_size );
            frame->size =header.frame_size;
        }

        frame->new =true;
        
        if( voxowl_read( sock, frame->data, frame->size ) == -1 )
            return -1;
    }
    return 0;
}
 
int 
main ( int argc, char **argv ) {
    if( argc != 3 ) {
            printf( "Usage: %s [server] [port]\n", *argv );
            exit( -1 );
    }
   
    const char* refresh_cmd ="renderer.render()\nframebuffer.synchronize()";
    const char* rotate_left_cmd ="camera.rotateAround(0.0, -0.1, 0.0)";
    const char* rotate_right_cmd ="camera.rotateAround(0.0, 0.1, 0.0)";
    const char* rotate_up_cmd ="camera.rotateAround(0.0, 0.0, -0.1)";
    const char* rotate_down_cmd ="camera.rotateAround(0.0, 0.0, 0.1)";
    const char* zoomin_cmd ="camera.translate(0.0, 0.0, 0.1)";
    const char* zoomout_cmd ="camera.translate(0.0, 0.0, -0.1)";
    const char* server_stop_cmd ="server.stop()";
    const char* enable_aa_cmd ="set renderer.featureAA 1";
    const char* disable_aa_cmd ="set renderer.featureAA 0";
    const char* enable_ssao_cmd ="set renderer.featureSSAO 1";
    const char* disable_ssao_cmd ="set renderer.featureSSAO 0";
    const char* enable_ssna_cmd ="set renderer.featureSSNA 1";
    const char* disable_ssna_cmd ="set renderer.featureSSNA 0";
    const char* enable_lighting_cmd ="set renderer.featureLighting 1";
    const char* disable_lighting_cmd ="set renderer.featureLighting 0";

    const char* load_1_cmd ="volumeloader.open( \"/local/s1407937/data/visoog-density.vxwl\" )";
    const char* load_2_cmd ="volumeloader.open( \"/local/s1407937/data/visoog-density-svmm90.vxwl\")";
    const char* load_3_cmd ="volumeloader.open( \"/local/s1407937/data/visoog-density-svmm65.vxwl\" )";
    const char* load_4_cmd ="volumeloader.open( \"/local/s1407937/data/visoog-density-svmm-bc.vxwl\")";
    const char* load_5_cmd ="volumeloader.open( \"/local/s1407937/data/menger-5-svmm.vxwl\")";

    bool aa_enabled =false, ssao_enabled =false, ssna_enabled =false, lighting_enabled =false;

    frame_info_t frame;
    frame_init( &frame );
    _texture =0;

    /* We'll setup SDL video first */
    
    if( SDL_Init( SDL_INIT_VIDEO ) ) {
            printf( "SDL_Init() error: %s\n", SDL_GetError() );
            return -1;
    }
    
    SDL_Window *window = SDL_CreateWindow( 
            "voxowl-sdl", 
            SDL_WINDOWPOS_UNDEFINED, SDL_WINDOWPOS_UNDEFINED,
            DEFAULT_WIDTH, DEFAULT_HEIGHT, SDL_WINDOW_SHOWN );
            
    SDL_Renderer *render = SDL_CreateRenderer(
            window, -1, SDL_RENDERER_ACCELERATED | SDL_RENDERER_PRESENTVSYNC );
    if( !render ) {
            SDL_DestroyWindow( window );
            SDL_Quit();
            return -1;
    }

    /* Next, we setup the network connection */

    struct voxowl_socket_t sock;
    if( voxowl_connect_host( &sock, argv[1], argv[2] ) != 0 ) {
      exit( -1 );
    }

    fprintf( stderr, "%s: connected to %s:%s\n", argv[0], argv[1], argv[2] );


    SDL_Event e;
    bool quit =false;
    bool ready_frame =true;
    
    while(!quit) {
            if( voxowl_poll_incoming( &sock ) != 0 ) {
                if( fetchIncoming( &sock, &frame ) == -1 ) {
                    fprintf( stderr, "Incomming package returned error.\n" );
                    frame.new =false;
                }
            }
            if( frame.new ) {
                updateTexture( render, &frame );
                renderTexture( render, window );
                ready_frame =true;
            }
            while (SDL_PollEvent(&e)){
                    if (e.type == SDL_QUIT){
                            quit = true;
                    }
                    if (e.type == SDL_KEYDOWN){
                            switch (e.key.keysym.sym)	{
                                    case SDLK_ESCAPE:
                                    case SDLK_q:
                                            quit = true;
                                            break;
                                    case SDLK_r: {
                                            if( ready_frame ) {
                                                ready_frame =false;
                                                voxowl_sendline( &sock, refresh_cmd, strlen( refresh_cmd ) );
                                            }
                                            break;
                                    }
                                    case SDLK_s: {
                                            voxowl_sendline( &sock, server_stop_cmd, strlen( server_stop_cmd ) );
                                            break;
                                    }
                                    case SDLK_LEFT: {
                                            if( ready_frame ) {
                                                ready_frame =false;
                                                voxowl_sendline( &sock, rotate_left_cmd, strlen( rotate_left_cmd ) );
                                                voxowl_sendline( &sock, refresh_cmd, strlen( refresh_cmd ) );
                                            }
                                            break;
                                    }
                                    case SDLK_RIGHT: {
                                            if( ready_frame ) {
                                                ready_frame =false;
                                                voxowl_sendline( &sock, rotate_right_cmd, strlen( rotate_right_cmd ) );
                                                voxowl_sendline( &sock, refresh_cmd, strlen( refresh_cmd ) );
                                            }
                                            break;
                                    }
                                    case SDLK_UP: {
                                            if( ready_frame ) {
                                                ready_frame =false;
                                                voxowl_sendline( &sock, rotate_up_cmd, strlen( rotate_up_cmd ) );
                                                voxowl_sendline( &sock, refresh_cmd, strlen( refresh_cmd ) );
                                            }
                                            break;
                                    }
                                    case SDLK_DOWN: {
                                            if( ready_frame ) {
                                                ready_frame =false;
                                                voxowl_sendline( &sock, rotate_down_cmd, strlen( rotate_down_cmd ) );
                                                voxowl_sendline( &sock, refresh_cmd, strlen( refresh_cmd ) );
                                            }
                                            break;
                                    }
                                    case SDLK_a: {
                                            if( ready_frame ) {
                                                ready_frame =false;
                                                voxowl_sendline( &sock, zoomin_cmd, strlen( zoomin_cmd ) );
                                                voxowl_sendline( &sock, refresh_cmd, strlen( refresh_cmd ) );
                                            }
                                            break;
                                    }
                                    case SDLK_z: {
                                            if( ready_frame ) {
                                                ready_frame =false;
                                                voxowl_sendline( &sock, zoomout_cmd, strlen( zoomout_cmd ) );
                                                voxowl_sendline( &sock, refresh_cmd, strlen( refresh_cmd ) );
                                            }
                                            break;
                                    }
                                    case SDLK_1: {
                                            if( aa_enabled )
                                                voxowl_sendline( &sock, disable_aa_cmd, strlen( disable_aa_cmd ) );
                                            else
                                                voxowl_sendline( &sock, enable_aa_cmd, strlen( enable_aa_cmd ) );
                                            aa_enabled =! aa_enabled;
                                            if( ready_frame ) {
                                                ready_frame =false;
                                                voxowl_sendline( &sock, refresh_cmd, strlen( refresh_cmd ) );
                                            }
                                            break;
                                    }
                                    case SDLK_2: {
                                            if( ssao_enabled )
                                                voxowl_sendline( &sock, disable_ssao_cmd, strlen( disable_ssao_cmd ) );
                                            else
                                                voxowl_sendline( &sock, enable_ssao_cmd, strlen( enable_ssao_cmd ) );
                                            ssao_enabled =! ssao_enabled;
                                            if( ready_frame ) {
                                                ready_frame =false;
                                                voxowl_sendline( &sock, refresh_cmd, strlen( refresh_cmd ) );
                                            }
                                            break;
                                    }
                                    case SDLK_3: {
                                            if( ssna_enabled )
                                                voxowl_sendline( &sock, disable_ssna_cmd, strlen( disable_ssna_cmd ) );
                                            else
                                                voxowl_sendline( &sock, enable_ssna_cmd, strlen( enable_ssna_cmd ) );
                                            ssna_enabled =! ssna_enabled;
                                            if( ready_frame ) {
                                                ready_frame =false;
                                                voxowl_sendline( &sock, refresh_cmd, strlen( refresh_cmd ) );
                                            }
                                            break;
                                    }
                                    case SDLK_4: {
                                            if( lighting_enabled )
                                                voxowl_sendline( &sock, disable_lighting_cmd, strlen( disable_lighting_cmd ) );
                                            else
                                                voxowl_sendline( &sock, enable_lighting_cmd, strlen( enable_lighting_cmd ) );
                                            lighting_enabled =! lighting_enabled;
                                            if( ready_frame ) {
                                                ready_frame =false;
                                                voxowl_sendline( &sock, refresh_cmd, strlen( refresh_cmd ) );
                                            }
                                            break;
                                    }
                                    case SDLK_6: {
                                            voxowl_sendline( &sock, load_1_cmd, strlen( load_1_cmd ) );
                                            if( ready_frame ) {
                                                ready_frame =false;
                                                voxowl_sendline( &sock, refresh_cmd, strlen( refresh_cmd ) );
                                            }
                                            break;
                                    }
                                    case SDLK_7: {
                                            voxowl_sendline( &sock, load_2_cmd, strlen( load_2_cmd ) );
                                            if( ready_frame ) {
                                                ready_frame =false;
                                                voxowl_sendline( &sock, refresh_cmd, strlen( refresh_cmd ) );
                                            }
                                            break;
                                    }
                                    case SDLK_8: {
                                            voxowl_sendline( &sock, load_3_cmd, strlen( load_3_cmd ) );
                                            if( ready_frame ) {
                                                ready_frame =false;
                                                voxowl_sendline( &sock, refresh_cmd, strlen( refresh_cmd ) );
                                            }
                                            break;
                                    }
                                    case SDLK_9: {
                                            voxowl_sendline( &sock, load_4_cmd, strlen( load_4_cmd ) );
                                            if( ready_frame ) {
                                                ready_frame =false;
                                                voxowl_sendline( &sock, refresh_cmd, strlen( refresh_cmd ) );
                                            }
                                            break;
                                    }
                                    case SDLK_0: {
                                            voxowl_sendline( &sock, load_5_cmd, strlen( load_5_cmd ) );
                                            if( ready_frame ) {
                                                ready_frame =false;
                                                voxowl_sendline( &sock, refresh_cmd, strlen( refresh_cmd ) );
                                            }
                                            break;
                                    }
                                    default:
                                            break;
                            }
                            
                    }
/*			if (e.type == SDL_MOUSEBUTTONDOWN){
                            reloadImage( render );
                    }*/
            }
            
    }

    voxowl_disconnect( &sock );
    
    if( _texture )
            SDL_DestroyTexture( _texture );
    SDL_DestroyRenderer( render );
    SDL_DestroyWindow( window );	
    SDL_Quit();
    
    return 0; 
}

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

        printf ( "decoding jpeg size = %d bytes\n", frame->size );

        tjhandle jpegDecompressor = tjInitDecompress();

        tjDecompressHeader2(jpegDecompressor, frame->data, frame->size, &width, &height, &jpegSubsamp);
        
        printf( "decoding  width = %d, height = %d\n", width, height );
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
        printf( "server: %s\n", msg );
        free( msg );
    } 
    else if( msg_type == VOXOWL_MODE_DATA ) {
        /* There should be a frame buffered */
        struct voxowl_frame_header_t header;
        if( voxowl_read_frame_header( sock, &header ) == -1 )
            return -1;

        fprintf( stderr, "Receiving frame (%u bytes) %dx%d\n", header.frame_size, header.width, header.height );
        
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
   
    const char* refresh_cmd ="renderer.render()\nframebuffer.write()";

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
                                            voxowl_sendline( &sock, refresh_cmd, strlen( refresh_cmd ) );
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

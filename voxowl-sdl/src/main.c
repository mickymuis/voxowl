/*
 * 
 *
 */
 
#include <SDL2/SDL.h>
#include "voxowl_network.h"

SDL_Texture* _texture;

void 
reloadImage( SDL_Renderer* render, const char* path, const char* command ) {
	if( _texture )
		SDL_DestroyTexture( _texture );
		
	if( system( command ) )
		return;
		
	SDL_Surface *surf =SDL_LoadBMP( path );
	if( surf ) {
		_texture =SDL_CreateTextureFromSurface( render, surf );
		SDL_FreeSurface( surf );
	}
}
 
int 
main ( int argc, char **argv ) {

	if( argc != 3 ) {
		printf( "Usage: %s [server] [port]\n", *argv );
                exit( -1 );
	}
	
	int width =800, height =600;
	_texture =0;

        /* We'll setup SDL video first */
	
	if( SDL_Init( SDL_INIT_VIDEO ) ) {
		printf( "SDL_Init() error: %s\n", SDL_GetError() );
		return -1;
	}
	
	SDL_Window *window = SDL_CreateWindow( 
		"voxowl-sdl", 
		SDL_WINDOWPOS_UNDEFINED, SDL_WINDOWPOS_UNDEFINED,
		width, height, SDL_WINDOW_SHOWN );
		
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
						reloadImage( render, argv[1], argv[2] );
						SDL_RenderClear( render );
						if( _texture ) {
							SDL_QueryTexture( _texture, NULL, NULL, &width, &height);
							SDL_SetWindowSize( window, width, height );
							
							SDL_RenderCopy( render, _texture, 0, 0 );
							SDL_RenderPresent( render );
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
	
	if( _texture )
		SDL_DestroyTexture( _texture );
	SDL_DestroyRenderer( render );
	SDL_DestroyWindow( window );	
	SDL_Quit();
	
	return 0; 
}

/*
 * 
 *
 */
 
#include <SDL2/SDL.h>

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
		printf( "Usage: %s [local image path] [refresh command]\n", *argv );
	}
	
	int width =800, height =600;
	_texture =0;
	
	if( SDL_Init( SDL_INIT_VIDEO ) ) {
		printf( "SDL_Init() error: %s\n", SDL_GetError() );
		return -1;
	}
	
	SDL_Window *window = SDL_CreateWindow( 
		"scpviewer", 
		SDL_WINDOWPOS_UNDEFINED, SDL_WINDOWPOS_UNDEFINED,
		width, height, SDL_WINDOW_SHOWN );
		
	SDL_Renderer *render = SDL_CreateRenderer(
		window, -1, SDL_RENDERER_ACCELERATED | SDL_RENDERER_PRESENTVSYNC );
	if( !render ) {
		SDL_DestroyWindow( window );
		SDL_Quit();
		return -1;
	}
	
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
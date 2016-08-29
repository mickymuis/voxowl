/*
 *  Z a n d b a k 
 *  met vormpjes!
 */

#define GLM_SWIZZLE

#include <SDL2/SDL.h>
#include "glm/glm.hpp"
#include "glm/vec2.hpp"
#include "glm/vec3.hpp"
#include "glm/vec4.hpp"
#include "main.h"
#include "utils.h"

// Window properties
#define WIDTH 700 // pixels
#define HEIGHT 700 // pixels
#define BORDER 30 // pixels
#define VOXEL_WIDTH 10 //pixels

//#define USE_TURBOJPEG

#ifdef USE_TURBOJPEG
#include <turbojpeg.h>
#endif

static SDL_Texture* _tex_grid =0;
static SDL_Texture* _tex_plot =0;
static SDL_Renderer* _renderer =0;

static int _steps =0;
static glm::ivec4 _ray_coords_px;
static plot_t _plot_ray;


plot_t
newPlot( glm::ivec2 begin_px, glm::vec4 color ) {
    plot_t p;
    p.color =color;
    p.last_px =begin_px;
    return p;
}

/*! voxel space to screenspace (pixels) */
glm::ivec2
voxelPx( glm::ivec3 v ) {
    return glm::ivec2( BORDER, HEIGHT-BORDER) + glm::ivec2( v.x*VOXEL_WIDTH, -v.y*VOXEL_WIDTH );
}

/*! world space to screenspace */
glm::ivec2
worldspacePx( glm::vec3 v ) {
    return glm::ivec2( BORDER, HEIGHT-BORDER) + glm::ivec2( v.x*VOXEL_WIDTH, -v.y*VOXEL_WIDTH );
}

/*! world space from screenspace */
glm::vec3
worldspaceFromPx( glm::ivec2 px ) {
    return glm::vec3(
            ((float)px.x - BORDER) / VOXEL_WIDTH,
            (- ((float)px.y - (HEIGHT-BORDER))) / VOXEL_WIDTH, 0 );
}

/*! unit cube space from world space */
glm::vec3
unitcubeFromWorld( glm::vec3 pos, glm::ivec3 size ) {
    return glm::vec3 (
            pos.x / size.x - .5f,
            pos.y / size.y - .5f,
            pos.z / size.z - .5f );
}

void
updateScreen() {
    SDL_SetRenderTarget( _renderer, NULL );
    SDL_RenderClear( _renderer );
    SDL_RenderCopy( _renderer, _tex_grid, NULL, NULL );
    SDL_RenderCopy( _renderer, _tex_plot, NULL, NULL );

    SDL_RenderPresent( _renderer );
}

void 
plot( plot_t* p, glm::ivec2 line_to_px ) {
    SDL_SetRenderTarget( _renderer, _tex_plot );
    SDL_SetRenderDrawColor( _renderer, 
            (int)p->color.r*255.f,
            (int)p->color.g*255.f,
            (int)p->color.b*255.f,
            (int)p->color.a*255.f );
    SDL_RenderDrawLine( _renderer, 
            p->last_px.x,
            p->last_px.y,
            line_to_px.x,
            line_to_px.y );
//    SDL_SetRenderTarget( _renderer, NULL );

    p->last_px =line_to_px;
}

void
plotCell( glm::ivec3 v, cell_mode_t mode, int scale ) {
    SDL_Rect r;
    glm::ivec2 px =voxelPx( v );
    r.x =px.x+(mode == OUTLINE ? 0 : 0 );
    r.y =px.y-scale*VOXEL_WIDTH+(mode == OUTLINE ? 0 : 0 );
    r.w =scale*VOXEL_WIDTH-(mode == OUTLINE ? 0 : 0 );
    r.h =scale*VOXEL_WIDTH-(mode == OUTLINE ? 0 : 0 );

//    printf( "draw rect: %d %d %d %d \n", r.x, r.y, r.w, r.h );

    switch( mode ) {
        default:
        case OUTLINE:
            SDL_SetRenderDrawColor( _renderer, 200, 200, 200, 255 );
            SDL_RenderDrawRect( _renderer, &r );
            break;
        case HIGHLIGHT:
            SDL_SetRenderDrawColor( _renderer, 225, 20, 30, 100 );
            SDL_RenderFillRect( _renderer, &r );
            break;
        case HIGHLIGHT2:
            SDL_SetRenderDrawColor( _renderer, 50, 20, 255, 100 );
            SDL_RenderFillRect( _renderer, &r );
            break;
        case COLORED:
            SDL_SetRenderDrawColor( _renderer, 25, 200, 150, 255 );
            SDL_RenderFillRect( _renderer, &r );
            break;
    }

}

void 
populateDummy( volume_t* v ) {
    level_t *level =v->levels;
    glm::ivec3 size;
    for( int i =0; i < v->n_levels; i++ ) {
        int f =level->mipmap_factor;
        size =v->size / f;
        for( int x =0; x < size.x; x++ )
            for( int y=0; y < size.y; y++ )
                for( int z=0; z < size.z; z++ ) {
                    char c =0;
                    int center = (v->size.x >> 1) + (v->size.y >> 1);
                    float pos = abs((x*f) - (v->size.x >> 1)) + abs((y*f)- (v->size.y >> 1));
                    pos = pos / center;
                    c |= TERMINAL * ( (float)(rand() % ((i+1)*300)) / 100.f <= pos );
                    c |= FILLED * ( (float)(rand() % 100) / 300.f > pos );
                    level->data_ptr[z+y*size.z+x*size.y*size.z] = c;
                }
        level++;
    }
}

void
drawGridRecursive( level_t *level, glm::ivec3 offset, glm::ivec3 size, int scale, int blockwidth ) {
    if( level->data_ptr == NULL )
        return;
    glm::ivec3 level_size =size / scale;
    for( int x =offset.x; x < offset.x+blockwidth; x++ )
        for( int y =offset.y; y < offset.y+blockwidth; y++ ) {
            char voxel =level->data_ptr[0+y*level_size.z+x*level_size.y*level_size.z];
            glm::ivec3 position =glm::ivec3( x,y,0 )*scale;
            level_t* next =level+1;
            if( next->data_ptr && !(voxel & TERMINAL) ) {
                plotCell( position, OUTLINE , scale );
                drawGridRecursive( next, position/next->mipmap_factor, size, next->mipmap_factor, blockwidth ); 
            } else {
                plotCell( position, (voxel&FILLED) ? COLORED : OUTLINE , scale );
                
            }
        }
}

void drawGrid( volume_t* v ) {
    drawGridRecursive( v->levels, glm::ivec3(0), v->size, v->levels->mipmap_factor, v->blockwidth );
}

void rebuildGrid( volume_t* volume ) {
    SDL_SetRenderTarget( _renderer, _tex_grid );
    SDL_SetRenderDrawColor( _renderer, 255, 255, 255, 255 );
    SDL_RenderClear( _renderer );
    
    populateDummy( volume );
    drawGrid( volume );
    SDL_SetRenderTarget( _renderer, NULL );
}

void raycast( volume_t* v, bool precise ) {
    glm::vec3 from, to;
    from =worldspaceFromPx( glm::ivec2( _ray_coords_px.x, _ray_coords_px.y ) );
    to =worldspaceFromPx( glm::ivec2( _ray_coords_px.z, _ray_coords_px.w ) );
    /*printf( "New ray: %d %d to %d %d (screen) %f %f to %f %f (world)\n", 
        _ray_coords_px.x, _ray_coords_px.y,_ray_coords_px.z,_ray_coords_px.w,
        from.x, from.y, to.x, to.y );*/
    
    // Completely clear the texture
    
    SDL_SetRenderTarget( _renderer, _tex_plot );
    SDL_SetRenderDrawColor( _renderer, 0,0,0,0 );
    SDL_RenderClear( _renderer );

    // Plot the ray

    _plot_ray =newPlot( _ray_coords_px.xy(), glm::vec4( 1,0,0,1 ) );
    plot( &_plot_ray, _ray_coords_px.zw() );

    // Convert the ray to 'unit cube space' (hackish)
    
    ray_t r;
    r.origin =unitcubeFromWorld( from, v->size );
    r.direction =unitcubeFromWorld( to, v->size ) - r.origin;

    int s =svmmRaycast( v, r, _steps, precise );
   // printf( "\r\tRaycast complete: %d steps taken, max. steps %d", s, _steps );
    printf( "\n" );
    fflush( stdout );

    updateScreen();
}

int 
main ( int argc, char **argv ) {

    /* We'll setup SDL video first */
    
    if( SDL_Init( SDL_INIT_VIDEO ) ) {
            printf( "SDL_Init() error: %s\n", SDL_GetError() );
            return -1;
    }
    
    SDL_Window *window = SDL_CreateWindow( 
            "vormpjes in de zandbak", 
            SDL_WINDOWPOS_UNDEFINED, SDL_WINDOWPOS_UNDEFINED,
            WIDTH, HEIGHT, SDL_WINDOW_SHOWN );
            
    _renderer = SDL_CreateRenderer(
            window, -1, SDL_RENDERER_ACCELERATED | SDL_RENDERER_PRESENTVSYNC );
    if( !_renderer ) {
            SDL_DestroyWindow( window );
            SDL_Quit();
            return -1;
    }
    
    SDL_Event e;
    bool quit =false;
    bool precise =true;

    /* Setup textures */

    _tex_grid = SDL_CreateTexture( _renderer, SDL_PIXELFORMAT_ARGB8888, SDL_TEXTUREACCESS_TARGET, WIDTH, HEIGHT ); 
    _tex_plot = SDL_CreateTexture( _renderer, SDL_PIXELFORMAT_ARGB8888, SDL_TEXTUREACCESS_TARGET, WIDTH, HEIGHT ); 
    SDL_SetTextureBlendMode( _tex_grid, SDL_BLENDMODE_BLEND );
    SDL_SetTextureBlendMode( _tex_plot, SDL_BLENDMODE_BLEND );
    SDL_SetRenderTarget( _renderer, _tex_plot );
    SDL_SetRenderDrawColor( _renderer, 0,0,0,0 );
    SDL_RenderClear( _renderer );

    /* Setup a dummy volume */

    char root[4][4][4];
    char level1[16][16][16];
    char level2[64][64][64];

    level_t levels[4];
    levels[0].data_ptr =(char*)root;
    levels[0].mipmap_factor =16;
    levels[1].data_ptr =(char*)level1;
    levels[1].mipmap_factor =4;
    levels[2].data_ptr =(char*)level2;
    levels[2].mipmap_factor =1;
    levels[3].data_ptr =NULL; // list end
    levels[3].mipmap_factor =1;
    volume_t volume;
    volume.size =glm::ivec3( 64 );
    volume.levels =levels;
    volume.n_levels =3;
    volume.blockwidth =4;

    /* Zandtaartjes bakken */

    rebuildGrid( &volume );

    _ray_coords_px =glm::ivec4( 10, 10, 660, 400 );
    raycast( &volume, precise );

    
    while(!quit) {
    //        SDL_WaitEvent( &e );
            while (SDL_PollEvent(&e)){
                    if (e.type == SDL_QUIT){
                            quit = true;
                    }
                    if (e.type == SDL_KEYDOWN){
                            switch (e.key.keysym.sym)	{
                                    case SDLK_ESCAPE:
                                    case SDLK_q:
                                        quit =true;
                                        break;
                                    case SDLK_r: 
                                        rebuildGrid( &volume );
                                        raycast( &volume, precise );
                                        break;
                                    case SDLK_LEFT:
                                        if( _steps > 0 )
                                            _steps--;
                                        raycast( &volume, precise );
                                        break;
                                    case SDLK_RIGHT:
                                        _steps++;
                                        raycast( &volume, precise );
                                        break;
                                    case SDLK_SPACE:
                                        precise = !precise;
                                        raycast( &volume, precise );
                                    default:
                                            break;
                            }
                            
                    }
		    if (e.type == SDL_MOUSEMOTION){
                        int x, y;
                        uint32_t state =SDL_GetMouseState( &x, &y );
                        if( state & SDL_BUTTON( SDL_BUTTON_LEFT ) ) {
                            _ray_coords_px.x =x;
                            _ray_coords_px.y =y;
                            raycast( &volume, precise );
                        }
                        else if( state & SDL_BUTTON( SDL_BUTTON_RIGHT ) ) {
                            _ray_coords_px.z =x;
                            _ray_coords_px.w =y;
                            raycast( &volume, precise );
                        }
                    }
            }
            
    }

    
    if( _tex_grid )
            SDL_DestroyTexture( _tex_grid );
    if( _tex_plot )
            SDL_DestroyTexture( _tex_plot );
    SDL_DestroyRenderer( _renderer );
    SDL_DestroyWindow( window );	
    SDL_Quit();
    printf ( "\nDone.\n" );
    
    return 0; 
}

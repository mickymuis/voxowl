
#include <stdio.h>
#include <string.h>
#include <stdbool.h>

#include "voxowl_convert.h"

typedef enum {
    IN_UNKNOWN,
    IN_TIFF,
    IN_VOXOWL
} in_filetype_t;

void 
printHelp( char *exec ) {
    printf( "Convert/compress volumetric data to Voxowl\n \
usage: %s input output [-iofqwrcsv]\n \
\n \
\t -i \t Set optional input type, either 'tiff' or 'voxowl'\n \
\t -o \t Set optional output type, either 'voxelmap' (default) or 'svmm'\n \
\t -f \t Force pixel format to be used, either one of:\n \
\t\t\t rgba32\n \
\t\t\t intensity8\n \
\t\t\t bitmap8\n \
\t\t\t rgb24_8alpha1\n \
\t    \t When svmm is used, only 'rgba32' and 'intensity8' can be used\n \
\t -q \t Quality [0-100] when compressing (svmm)\n \
\t -w \t Set blockwidth (svmm)\n \
\t -r \t Set rootwidth (svmm)\n \
\t -c \t Enable chroma compression of the base-level (svmm)\n \
\t -s \t Silent, no output\n \
\t -v \t Verify output if svmm is used\n \
\t -h \t This information\n", exec );
}

void
parseInFiletype( in_filetype_t *type, char *str ) {
    if( strncmp( str, "tiff", 4 ) == 0 )
        *type = IN_TIFF;
    if( strncmp( str, "voxowl", 6 ) == 0 )
        *type = IN_VOXOWL;
}

void
parseOutFiletype( filetype_t *type, char *str ) {
    if( strncmp( str, "svmm", 4 ) == 0 )
        *type = FILETYPE_SVMM;
    if( strncmp( str, "voxelmap", 9 ) == 0 )
        *type = FILETYPE_VOXELMAP;
}

void
parseInFilename( in_filetype_t *type, char *name ) {
    char *ext =0;
    ext =strrchr( name, '.' );
    if( ext && strncasecmp( ext, ".tif", 4 ) == 0 )
        *type = IN_TIFF;
    else if( ext && strncasecmp( ext, ".vxwl", 5 ) == 0 )
        *type = IN_VOXOWL;
}

bool
parseVoxelFormat( voxel_format_t* format, char* str ) {
    if( strncasecmp( str, "rgba32", 6 ) == 0 )
        *format =VOXEL_RGBA_UINT32;
    else if( strncasecmp( str, "intensity8", 10 ) == 0 )
        *format =VOXEL_INTENSITY_UINT8;
    else if( strncasecmp( str, "bitmap8", 7 ) == 0 )
        *format =VOXEL_BITMAP_UINT8;
    else if( strncasecmp( str, "rgb24_8alpha1", 13 ) == 0 )
        *format =VOXEL_RGB24_8ALPHA1_UINT32;
    else
        return false;
    return true;
}

bool
parseNumber( int* n, char* str, int max ) {
    int i =atoi( str );
    if( !i || i > max ) {
        fprintf( stderr, "Invalid numerical argument '%s'\n\n", str );
        return false;
    }
    *n =i;
    return true;
}

int 
main( int argc, char **argv ) {
    convert_settings_t s;
    s.in_filename =0;
    s.out_filename =0;
    s.silent =false;
    s.verify =false;
    s.set_baselevel_bitmap =false;
    s.set_blockwidth =0;
    s.set_rootwidth =0;
    s.set_quality =75; // default quality?
    s.set_voxel_format =false;

    in_filetype_t in_filetype =IN_UNKNOWN;
    s.out_filetype =FILETYPE_VOXELMAP;
    
    for( int i =1; i < argc; i++ ) {
        if( strncmp( argv[i], "-i", 2 ) == 0 ) {
            parseInFiletype( &in_filetype, argv[++i] );    
        } else if( strncmp( argv[i], "-o", 2 ) == 0 ) {
            parseOutFiletype( &s.out_filetype, argv[++i] );
        } else if( strncmp( argv[i], "-f", 2 ) == 0 ) {
            if( !parseVoxelFormat( &s.voxel_format, argv[++i] ) ) {
                fprintf( stderr, "Unknown format '%s'\n\n", argv[i] );
                return -1;
            } else
                s.set_voxel_format =true;
        } else if( strncmp( argv[i], "-q", 2 ) == 0 ) {
            if( !parseNumber( &s.set_quality, argv[++i], 100 ) ) {
                return -1;
            }
        } else if( strncmp( argv[i], "-w", 2 ) == 0 ) {
            if( !parseNumber( &s.set_blockwidth, argv[++i], 128 ) ) {
                return -1;
            }
            if( s.set_blockwidth % 2 ) {
                fprintf( stderr, "Blockwidth must be multiple of two.\n\n" );
                return -1;
            }
        } else if( strncmp( argv[i], "-r", 2 ) == 0 ) {
            if( !parseNumber( &s.set_rootwidth, argv[++i], 1024 ) ) {
                return -1;
            }
            if( s.set_rootwidth % 2 ) {
                fprintf( stderr, "Rootwidth must be multiple of two.\n\n" );
                return -1;
            }
        } else if( strncmp( argv[i], "-c", 2 ) == 0 ) {
            s.set_baselevel_bitmap =true;
        } else if( strncmp( argv[i], "-s", 2 ) == 0 ) {
            s.silent =true;
        } else if( strncmp( argv[i], "-v", 2 ) == 0 ) {
            s.verify =true;
        } else if( strncmp( argv[i], "-h", 2 ) == 0 ) {
            printHelp( argv[0] );
            return 0;
        } else {
            if( !s.in_filename ) {
                parseInFilename( &in_filetype, argv[i] );
                s.in_filename =argv[i];
            } else if( !s.out_filename ) {
                s.out_filename =argv[i];
            } else {
                fprintf( stderr, "Stray argument '%s'\n\n", argv[i] );
                printHelp( argv[0] );
                return -1;
            }
        }
    }
    if( !s.in_filename || !s.out_filename ) {
        fprintf( stderr, "To few files specified\n\n" );
        printHelp( argv[0] );
        return -1;
    }
    if( !in_filetype || !s.out_filetype ) {
        fprintf( stderr, "Filetype(s) not recognized. Try -i\n\n" );
        printHelp( argv[0] );
        return -1;
    }
    if( in_filetype == IN_VOXOWL ) {
        // We need to peek into the file to determine its precise format
        FILE* f =fopen( s.in_filename, "r" );
        if( !f ) {
            fprintf( stderr, "Cannot open '%s' for reading\n", s.in_filename );
            return -1;
        }
        char buf[2];
        fread( buf, 1, 2, f );
        if( buf[0] == VOXELMAP_MAGIC1 && buf[1] == VOXELMAP_MAGIC2 )
            s.in_filetype =FILETYPE_VOXELMAP;
        else if( buf[0] == SVMM_MAGIC1 && buf[1] == SVMM_MAGIC2 )
            s.in_filetype =FILETYPE_SVMM;
        else {
            fprintf( stderr, "'%s' is not a Voxowl file format.\n", s.in_filename );
            return -1;
        }
        fclose( f );
    } else
        s.in_filetype =FILETYPE_TIFF;

    return convert( s );
}

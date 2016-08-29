#include "voxowl_convert.h"

#include <string.h>
#include <tiffio.h>

int
tiffConvert( voxelmap_t *v, 
             TIFF* tif,
             convert_settings_t s ) {

    // We're now ready to write the TIFF image to the voxelmap
    // Unfortunately, libtiff 'put methods' are not clearly documented, so we take the long route.
    
    int x, y, z =0;
    TIFFSetDirectory( tif, 0 );

    // Iterate over each image plane (directory in TIFF jargon)
    do {
        TIFFRGBAImage img;
        char emsg[1024];
        if( !s.silent )
            printf( "Writing plane %d of %d\n", z+1, v->size.z );

        if( TIFFRGBAImageBegin( &img, tif, 0, emsg ) ) {
            size_t npixels;
            uint32* raster;

            npixels = img.width * img.height;
            raster = (uint32*) _TIFFmalloc( npixels * sizeof (uint32) );
            if (raster != NULL) {
                if( TIFFRGBAImageGet( &img, raster, img.width, img.height ) ) {
                    // Now store this single plane in the voxelmap
                    for( x =0; x < img.width; x++ )
                        for( y =0; y < img.height; y++ ) {
                            glm::vec4 rgba =unpackABGR_UINT32( *(raster+(img.height-y)*img.width+x) );
                            if( rgba.r < 0.1 && rgba.g < 0.1 && rgba.b < 0.1 )
                                voxelmapPack( v, ivec3_32( x, y, z ), glm::vec4(0) );
                            else
                                voxelmapPack( v, ivec3_32( x, y, z ), rgba );
                        }
                }
                _TIFFfree( raster );
            }
                TIFFRGBAImageEnd( &img );
        } else {
            TIFFError( "", emsg );
            return -1;
        }
        z++;
        
    } while( TIFFReadDirectory( tif ) );
    
    TIFFClose( tif );
    return 0;
}

 
bool 
svmmVerify( voxelmap_t* uncompressed, svmipmap_t* mipmap, svmm_encode_opts_t opts ) {
    int errors =0;

    for( int x=0; x < uncompressed->size.x; x++ )
        for( int y=0; y < uncompressed->size.y; y++ )
            for( int z=0; z < uncompressed->size.z; z++ ) {
                //fprintf( stderr, "@ %d, %d, %d: ", x, y, z );
                glm::vec4 orig =voxelmapUnpack( uncompressed, ivec3_32( x, y, z ) );
                glm::vec4 enc =svmmDecodeVoxel( mipmap, ivec3_32( x, y, z ) );
                bool test =glm::all( glm::lessThanEqual ( glm::abs( orig - enc ), glm::vec4( opts.delta ) ) );
                if( !test ) {
                    if( !errors++ )
                        fprintf( stderr, "Unexpected value. original: (%f %f %f %f), encoded: (%f %f %f, %f), delta: %f, at (%d %d %d)\n",
                        orig.r, orig.g, orig.b, orig.a, enc.r, enc.g, enc.b, enc.a, opts.delta, x, y, z );
                }
                //DEBUG
                voxelmapPack( uncompressed, ivec3_32( x, y, z ), enc );
            }
    fprintf( stdout, "Test completed. errors: %d\n", errors );
    return errors == 0;
}

int 
convert( convert_settings_t s ) {
    voxelmap_t v_out, v_in;
    voxelmap_mapped_t vm_out, vm_in;
    voxelmap_t* v =&v_out;
    svmipmap_t svmm;
    
    memset( &v_out, 0, sizeof( voxelmap_t ) );
    memset( &vm_out, 0, sizeof( voxelmap_mapped_t ) );
    memset( &v_in, 0, sizeof( voxelmap_t ) );
    memset( &vm_in, 0, sizeof( voxelmap_mapped_t ) );
    memset( &svmm, 0, sizeof( svmipmap_t ) );

    TIFF *tif;
    // In certain conditions, we need to convert to an in-memory intermediate voxelmap
    bool intermediate =s.out_filetype == FILETYPE_SVMM && s.in_filetype != FILETYPE_VOXELMAP; 

    ivec3_32_t volume_size;

    /* Open and inspect the input file */
    if( s.in_filetype == FILETYPE_TIFF ) {
        // Let's open the TIFF file and read its properties
        //
        int width, height, dircount =0, samples_per_pixel;

        tif = TIFFOpen( s.in_filename, "r" );
        if (tif) {
            TIFFGetField( tif, TIFFTAG_IMAGELENGTH, &height );
            TIFFGetField( tif, TIFFTAG_IMAGEWIDTH, &width );
            TIFFGetField( tif, TIFFTAG_SAMPLESPERPIXEL, &samples_per_pixel );

            do {
                dircount++;
            } while ( TIFFReadDirectory( tif ) );
        } else {
            fprintf( stderr, "Cannot open TIFF file in '%s'\n", s.in_filename );
            return -1;
        }
        if( !s.silent )
            printf( "Reading TIFF file: %d x %d (%d image planes) at %d channels/pixel\n",
                    width, height, dircount, samples_per_pixel );
        
        if( !s.set_voxel_format ) {
            switch( samples_per_pixel ) {
                case 1:
                    s.voxel_format =VOXEL_INTENSITY_UINT8;
                    break;
                case 2:
                case 3:
                case 4:
                    s.voxel_format =VOXEL_RGBA_UINT32;
                    break;
                default:
                    fprintf( stderr, "Unsupported TIFF image-type\n" );
                    return -1;
            }
        }
        volume_size =ivec3_32( width, height, dircount );
    }
    else if( s.in_filetype == FILETYPE_VOXELMAP ) {
        if( voxelmapOpenMapped( &vm_in, s.in_filename ) == -1 ) {
            printf( "Could not open voxelmap '%s'\n", s.in_filename );
            perror( "" );
            return -1;
        }
        voxelmapFromMapped( &v_in, &vm_in );
        if( !s.set_voxel_format )
            s.voxel_format =v_in.format;
        volume_size =v_in.size;
        if( !s.silent )
            printf( "Opened voxelmap (%d x %d x %d) with format %s\n", 
                  volume_size.x, volume_size.y, volume_size.z, strVoxelFormat( v_in.format ) );
        v =&v_in; // Read from this voxelmap
    }
    else if( s.in_filetype == FILETYPE_SVMM ) {
        if( svmmOpenMapped( &svmm, s.in_filename ) == -1 ) {
            printf( "Could not open sparse voxel mipmap '%s'\n", s.in_filename );
            perror( "" );
            return -1;
        }
        if( !s.set_voxel_format )
            s.voxel_format =svmm.header.format;
        volume_size =svmm.header.volume_size;
        if( !s.silent )
            printf( "Opened sparse voxel mipmap (%d x %d x %d) with format %s, %d mipmap levels\n", 
                  volume_size.x, volume_size.y, volume_size.z, strVoxelFormat( svmm.header.format ), svmm.header.levels );
    }
    
    /* Create a same-sized voxelmap, either as intermediate or as final output */
    if( s.in_filetype != FILETYPE_VOXELMAP || s.out_filetype == FILETYPE_VOXELMAP ) {
        
        if( !s.silent )
            printf( "Writing to voxelmap (%d,%d,%d) using voxel format %s\n",
                    volume_size.x, volume_size.y, volume_size.z, strVoxelFormat( s.voxel_format ) );

        if( !intermediate ) {
            if( voxelmapCreateMapped( &vm_out, s.out_filename, s.voxel_format, volume_size ) == - 1 ) {
                fprintf( stderr, "Could not allocate voxelmap\n" );
                perror( "" );
                TIFFClose( tif );
                return -1;
            }
            voxelmapFromMapped( &v_out, &vm_out );
        } else
            voxelmapCreate( &v_out, s.voxel_format, volume_size );
    }

    /* If the input is not already a voxelmap, convert it */
    if( s.in_filetype == FILETYPE_TIFF ) {
        if( tiffConvert( &v_out, tif, s ) == -1 )
            return -1;
    } else if( s.in_filetype == FILETYPE_SVMM ) {
        if( !svmmDecode( &v_out, &svmm ) == -1 )
            return -1;
    }
    /* Otherwise, just copy */
    else if( s.in_filetype == FILETYPE_VOXELMAP && s.out_filetype == FILETYPE_VOXELMAP ) {
        voxelmapSafeCopy( &v_out, &v_in );
    }

    /* If the output type is SVMM, we still need to compress the data */

    if( s.out_filetype == FILETYPE_SVMM ) {
        svmm_encode_opts_t opts;
        svmmSetOpts( &opts, v, s.set_quality );

        if( s.set_blockwidth )
            opts.blockwidth = s.set_blockwidth;
        if( s.set_rootwidth )
            opts.rootwidth = s.set_rootwidth;
        opts.shiftBlockwidth =s.set_step_blockwidth;
        opts.bitmapBaselevel = s.set_baselevel_bitmap;
        opts.format =svmmFormat( s.voxel_format );

        if( !s.silent )
            printf( "Encoding svmm: blockwidth=%d, rootwidth=%d, delta=%f, quality=%d, bitmap_baselevel=%d, format=%s\n",
                    opts.blockwidth, opts.rootwidth, opts.delta, s.set_quality, opts.bitmapBaselevel, strVoxelFormat( opts.format ) );

        ssize_t svmm_size;
//        svmipmap_t svmm;
//        svmm.is_mmapped =false;
        if( (svmm_size = svmmEncodeFile( s.out_filename, v, opts )) == -1 ) {
            fprintf( stderr, "Error occured in svmm encoding\n" );
            perror( "" );
            return -1;
        }
        if( !s.silent )
            fprintf( stdout, "Compressed size: %dK, uncompressed size: %dK, ratio: %d%\n",
                svmm_size/1024, voxelmapSize( v )/1024, 
                (int)(((float)svmm_size / (float)voxelmapSize( v ))*100.f) );
        if( s.verify ) {

        }
    }

    /* Close files and be done */

    if( v_out.data ) {
        if( !intermediate )
            voxelmapUnmap( &vm_out );
        else
            voxelmapFree( &v_out );
    }
    if( vm_in.mapped_addr )
        voxelmapUnmap( &vm_in );

    return 0;
}

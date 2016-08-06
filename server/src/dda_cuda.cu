#include "dda_cuda.h"

VOXOWL_DEVICE
glm::vec4
voxelTex3D( cudaTextureObject_t &texture, voxel_format_t format, glm::ivec3 index ) {
    glm::ivec3 block =glm_ivec3_32( blockPosition( format, ivec3_32( index ) ) );
    glm::vec4 vox;

    switch( format ) {
        case VOXEL_RGBA_UINT32: {
            vox =unpackRGBA_UINT32( tex3D<uint32_t>( texture, block.x, block.y, block.z ) );
            break;
        }
        case VOXEL_INTENSITY_UINT8: {
            uint8_t gray =tex3D<uint8_t>( texture, block.x, block.y, block.z );
            vox =glm::vec4( unpackINTENSITY_UINT8( gray ) );
            vox.a =(float)(vox.r != 0.f);
            break;
        }
        case VOXEL_BITMAP_UINT8: {
            uint8_t bitmap =tex3D<uint8_t>( texture, block.x, block.y, block.z );
            unsigned bit_offs =blockOffset( format, ivec3_32(index) );
            vox =glm::vec4( (int)unpackBIT_UINT8( bitmap, bit_offs ) );
            break;
        }
        case VOXEL_RGB24_8ALPHA1_UINT32: {
            uint32_t rgb24_8alpha1 =tex3D<uint32_t>( texture, block.x, block.y, block.z );
            unsigned bit_offs =blockOffset( format, ivec3_32(index) );
            vox =unpackRGBA_RGB24_8ALPHA1_UINT32( rgb24_8alpha1, bit_offs );

            break;
        }
        case VOXEL_RGB24A1_UINT32: {
            vox =unpackRGB24A1_UINT32( tex3D<uint32_t>( texture, block.x, block.y, block.z ) );
            break;
        }
    }

    return vox;
}

VOXOWL_DEVICE 
glm::vec4 
voxelTex3D_clamp( 
        cudaTextureObject_t &texture, 
        voxel_format_t format, 
        glm::ivec3 index, 
        glm::ivec3 clamp_size ) {
    glm::vec4 vox =voxelTex3D( texture, format, index );
    vox.a *= (float)( glm::all( glm::greaterThanEqual( index, glm::ivec3(0) ) ) && glm::all( glm::lessThan( index, clamp_size ) ) );
/*    if(! (float)( glm::all( glm::greaterThanEqual( index, glm::ivec3(0) ) ) && glm::all( glm::lessThan( index, clamp_size ) ) ) )
        vox =glm::vec4(1,0,0,1);*/

    return vox;
}

/* Cast one ray r into the volume bounded by v. The actual volume data is obtained from the global volume texture */
VOXOWL_DEVICE
fragment_t
voxelmapRaycast( voxelmapDevice_t *v, const ray_t& r ) {
    double tmin, tmax;
    glm::ivec3 size =v->size;
    
    fragment_t frag;
    frag.color =glm::vec4( 0,0,0,1 );
    frag.normal =glm::vec3(0);
    frag.position =glm::vec3(0,0,-1000.f);
    frag.position_vs =glm::vec3(0);

    box_t b = volumeSizeToAABB( size );
    if( !rayAABBIntersect( r, b, tmin, tmax ) )
        return frag;

    glm::vec3 rayEntry = r.origin + r.direction * (float)max( 0.0, tmin );
    glm::vec3 rayExit = r.origin + r.direction * (float)tmax;

    // Determine the side of the unit cube the ray enters
    // In order to do this, we need the component with the largest absolute number
    // These lines are optimized to do so without branching
    const glm::vec3 box_plane( 0, 1, 2 ); // X, Y and Z dividing planes
    glm::vec3 r0 = glm::abs( rayEntry / b.max );
    float largest =max( r0.x, max( r0.y, r0.z ) ); // Largest relative component
    glm::vec3 r1 = glm::floor( r0 / largest ); // Vector with a '1' at the largest component
    int side = glm::clamp( glm::dot( box_plane, r1 ), 0.f, 2.f );
   
    // Map the ray entry from unit-cube space to voxel space
    largest =max( size.x, max( size.y, size.z ) );
    glm::vec3 rayEntry_vs =(rayEntry + b.max) * largest;

    // Calculate the index in the volume by chopping off the decimal part
    glm::ivec3 index = glm::clamp( 
        glm::ivec3( glm::floor( rayEntry_vs ) ) ,
        glm::ivec3( 0 ),
        glm::ivec3( size.x-1, size.y-1, size.z-1 ) );

    frag.position_vs = glm::clamp( 
        glm::vec3( rayEntry_vs ) ,
        glm::vec3( 0 ),
        glm::vec3( size.x-1, size.y-1, size.z-1 ) );

    // Determine the sign of the stepping through the volume
    glm::ivec3 step = glm::sign( r.direction );

    // EDIT: mitigate the division-by-zero problem
    glm::bvec3 zeros =glm::equal( r.direction, glm::vec3(0) );
    glm::vec3 direction =glm::vec3(glm::not_(zeros)) * r.direction + glm::vec3(zeros) * glm::vec3(.00000001f); 
    // deltaDist gives the distance on the ray path for each following dividing plane
    glm::vec3 deltaDist =glm::abs( glm::vec3( glm::length( r.direction ) ) / r.direction );

    // Computes the distances to the next voxel for each component
    glm::vec3 boxDist = ( sign( r.direction ) * (glm::vec3(index) - rayEntry_vs)
                        + (sign( r.direction ) * 0.5f ) + 0.5f ) * deltaDist;

    
    while(1) {

  //      if( index[side] < 0 || index[side] >= size[side] )
  //          break;
        if( glm::any( glm::lessThan( index, glm::ivec3(0) ) ) || glm::any( glm::greaterThanEqual( index, size ) ) )
            break;
        
        glm::vec4 vox = voxelTex3D( v->texture, v->format, glm::ivec3( glm::floor( frag.position_vs ) ) );
/*        vox.r *= (3-side)/3.f;
        vox.g *= (3-side)/3.f;
        vox.b *= (3-side)/3.f;*/

        frag.color =blendF2B( vox, frag.color );

        if( frag.color.a < 0.1f ) {
            // We calculate the position in unit-cube space..
            frag.position =frag.position_vs / (float)largest - b.max;
            // ..and the normal of the current 'face' of the voxel
            frag.normal[side] = -step[side];
            break;
        }


        // Branchless equivalent for
        //for( int i =0; i < 3; i++ ) 
        //    if( boxDist[side] > boxDist[i] )
        //        side =i;*/
        glm::bvec3 b0= glm::lessThan( boxDist, glm::vec3( boxDist.y, boxDist.z, boxDist.x ) /*boxDist.yzx()*/ );
        glm::bvec3 b1= glm::lessThanEqual( boxDist, glm::vec3( boxDist.z, boxDist.x, boxDist.y ) /*boxDist.zxy()*/ );
        glm::vec3 mask =glm::vec3( b0.x && b1.x, b0.y && b1.y, b0.z && b1.z );
/*              glm::vec3 boxDist2 =glm::floor( boxDist );
                glm::bvec3 b0= glm::lessThanEqual( boxDist2, glm::vec3( boxDist2.y, boxDist2.z, boxDist2.x ) );
                glm::bvec3 b1= glm::lessThanEqual( boxDist2, glm::vec3( boxDist2.z, boxDist2.x, boxDist2.y ) );
                glm::vec3 mask =glm::vec3( b0.x && b1.x, b0.y && b1.y, b0.z && b1.z );*/
//        side = glm::dot( box_plane, mask );

        boxDist += deltaDist * mask;
        index += step * glm::ivec3(mask);
        frag.position_vs += step * glm::ivec3(mask);
    }

    frag.color.a = 1.f - frag.color.a;

    return frag;
}

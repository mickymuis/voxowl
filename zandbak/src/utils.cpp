/*
 *  Z a n d b a k 
 *  met vormpjes!
 */
#include "utils.h"
#include "main.h"
#include <algorithm>

// Plots!

#define PLOT_INDEX_COLOR glm::vec4( .1f, .2f, .8f, 1.f )
static plot_t _plot_index;

/* Produces an Axis Aligned Bounding Box (AABB) from a given size vector
 *    - The resulting box is axis aligned, unit sized and centered around (0,0,0)
 *       - The ratios width/height/depth match the given size vector 
 *          - The largest direction in the size vector will measure 1.0 in the box */
box_t
volumeSizeToAABB( glm::ivec3 size ) {
    int largest = std::max( size.x, std::max( size.y, size.z ) );
    largest *=2;
    box_t b;
    b.min = -glm::vec3 (
        (float)size.x / (float)largest,
        (float)size.y / (float)largest,
        (float)size.z / (float)largest );
    b.max =-b.min - glm::vec3(0.0) ;
    return b;
}

/* Given a ray r and an AABB b, return true if r intersects b in any point
 *    Additionally, the entry and exit constants tmin and tmax are set */
bool
rayAABBIntersect( const ray_t &r, const box_t& b, double& tmin, double& tmax ) {
    glm::vec3 n_inv = glm::vec3( 
        1.f / r.direction.x,
        1.f / r.direction.y,
        1.f / r.direction.z );
    double tx1 = (b.min.x - r.origin.x)*n_inv.x;
    double tx2 = (b.max.x - r.origin.x)*n_inv.x;
 
    tmin = std::min(tx1, tx2);
    tmax = std::max(tx1, tx2);
 
    double ty1 = (b.min.y - r.origin.y)*n_inv.y;
    double ty2 = (b.max.y - r.origin.y)*n_inv.y;
 
    tmin = std::max(tmin, std::min(ty1, ty2));
    tmax = std::min(tmax, std::max(ty1, ty2));
    
    double tz1 = (b.min.z - r.origin.z)*n_inv.z;
    double tz2 = (b.max.z - r.origin.z)*n_inv.z;
 
    tmin = std::max(tmin, std::min(tz1, tz2));
    tmax = std::min(tmax, std::max(tz1, tz2));
 
    return tmax >= tmin;
}

int
svmmRaycast( volume_t* v, const ray_t& r, int steps, bool precise ) {
    double tmin, tmax;
    glm::ivec3 size =v->size;
    
    fragment_t frag;
    frag.color =glm::vec4( 0,0,0,1 );
    frag.normal =glm::vec3(0);
    frag.position =glm::vec3(0);
    frag.position_vs =glm::vec3(0);

    box_t b = volumeSizeToAABB( size );
    if( !rayAABBIntersect( r, b, tmin, tmax ) )
        return 0;

    glm::vec3 rayEntry = r.origin + r.direction * (float)std::max( 0.0, tmin );
    glm::vec3 rayExit = r.origin + r.direction * (float)tmax;

    // Determine the side of the unit cube the ray enters
    // In order to do this, we need the component with the largest absolute number
    // These lines are optimized to do so without branching
    const glm::vec3 box_plane( 0, 1, 2 ); // X, Y and Z dividing planes
    glm::vec3 r0 = glm::abs( rayEntry / b.max );
    float largest =std::max( r0.x, std::max( r0.y, r0.z ) ); // Largest relative component
    glm::vec3 r1 = glm::floor( r0 / largest ); // Vector with a '1' at the largest component
    int side = glm::clamp( glm::dot( box_plane, r1 ), 0.f, 2.f );
   
    // Map the ray entry from unit-cube space to voxel space
    largest =std::max( size.x, std::max( size.y, size.z ) );
    glm::vec3 rayEntry_vs =(rayEntry + b.max) * largest;

    // Calculate the index in the volume by chopping off the decimal part
/*    glm::ivec3 index = glm::clamp( 
        glm::ivec3( glm::floor( rayEntry_vs ) ) ,
        glm::ivec3( 0 ),
        glm::ivec3( size.x-1, size.y-1, size.z-1 ) );*/

    /*frag.position_vs = glm::clamp( 
        glm::vec3( rayEntry_vs ) ,
        glm::vec3( 0 ),
        glm::vec3( size.x-1, size.y-1, size.z-1 ) );*/

    frag.position_vs =rayEntry_vs;

    // Determine the sign of the stepping through the volume
    glm::vec3 step = glm::sign( r.direction );

    // EDIT: mitigate the division-by-zero problem
    glm::bvec3 zeros =glm::equal( r.direction, glm::vec3(0) );
    glm::vec3 direction =glm::vec3(glm::not_(zeros)) * r.direction + glm::vec3(zeros) * glm::vec3(.00000001f); 
    // deltaDist gives the distance on the ray path for each following dividing plane
    glm::vec3 deltaDist =glm::abs( glm::vec3( glm::length( r.direction ) ) / direction );
    printf( "deltaDist: (%f,%f,%f)\n", 
            deltaDist.x, deltaDist.y, deltaDist.z );
    deltaDist *= ((b.max * 2.f) / glm::vec3(size));
        //glm::vec3 deltaDist =((b.max * 2.f) / glm::vec3(size)) / glm::normalize(direction);

    // Computes the distances to the next voxel for each component
    glm::vec3 boxDist = ( sign( r.direction ) * (glm::floor(rayEntry_vs) - rayEntry_vs)
                        + (sign( r.direction ) * 0.5f ) + 0.5f ) * deltaDist;

/*    glm::vec3 ddist;
    ddist.x =( step.x > 0 ? (glm::floor( rayEntry_vs.x ) + 1.f - rayEntry_vs.x) : (rayEntry_vs.x - glm::floor( rayEntry_vs.x ) ) );
    ddist.y =( step.y > 0 ? (glm::floor( rayEntry_vs.y ) + 1.f - rayEntry_vs.y) : (rayEntry_vs.y - glm::floor( rayEntry_vs.y ) ) );
    ddist.z =( step.z > 0 ? (glm::floor( rayEntry_vs.z ) + 1.f - rayEntry_vs.z) : (rayEntry_vs.z - glm::floor( rayEntry_vs.z ) ) );

    glm::vec3 boxDist( deltaDist / ddist );*/

    printf( "deltaDist: (%f,%f,%f)  boxDist: (%f,%f,%f)\n", 
            deltaDist.x, deltaDist.y, deltaDist.z,
            boxDist.x, boxDist.y, boxDist.z );

    // PLOT
    _plot_index =newPlot( voxelPx( glm::floor( frag.position_vs ) ), PLOT_INDEX_COLOR );

    // ABOVE IS THE SAME AS voxelmapRaycast()
    //
    level_t *cur_level =v->levels;
    int level =0;

    char vox;
    glm::ivec3 abs_index, block_index, block_offset, subblock_index, grid_offset, voxel_index, mipmap_size;
//    uint64_t block_num =0;
//    uint32_t vox_raw;
    glm::ivec3 superblock_bounds;
    
    bool first =true;
    int s;
    for( s =0; s < steps; s++ ) {

        glm::ivec3 index =glm::floor( frag.position_vs );
        
       // if( glm::any( glm::lessThan( index, glm::ivec3(0) ) ) || glm::any( glm::greaterThanEqual( index, size ) ) )
       //     break;
        
        cur_level =v->levels;
        level =0;
        //block_num =0;
        //

        while(1) {
            mipmap_size =v->size / cur_level->mipmap_factor; // REDUNDANT
            // Absolute position in the mipmap level
            abs_index =index / cur_level->mipmap_factor;
            // Relative position in the block
            block_index =abs_index;
            // The block is devided in subblocks: calculate the offset within the subblock
            block_offset = glm::ivec3( 0 );
            // Calculate the index of the first block of the subblock
            subblock_index =block_index - block_offset;
            // Blocks are ordered linear on disk but as 3D texture on the GPU,
            // this is the 'block grid'. We need to calculate the block's
            // position in the grid.
            grid_offset =glm::ivec3(0);

            voxel_index =grid_offset * v->blockwidth + block_index;
            //printf( "(%d,%d,%d) ", voxel_index.x, voxel_index.y, voxel_index.z );
            // Determine the level's format
            vox =cur_level->data_ptr[voxel_index.z+voxel_index.y*mipmap_size.z+voxel_index.x*mipmap_size.z*mipmap_size.y];

            if( level == v->n_levels-1 || vox & TERMINAL ) {
                break;
            }

            level++;
            cur_level++;
        }

//        frag.color =blendF2B( vox, frag.color );

        if( vox & FILLED ) {
            // We calculate the position in unit-cube space..
            frag.position =frag.position_vs / (float)largest - b.max;
            // ..and the normal of the current 'face' of the voxel
            frag.normal[side] = -step[side];
            first =false;
            break;
        }

/*        if( frag.color.a < 0.1f ) {
            break;
        }*/
        
//        plot( &_plot_index, voxelPx( index ) ); // PLOT
        //plotCell( index, HIGHLIGHT, 1 );
        plotCell( abs_index*cur_level->mipmap_factor, HIGHLIGHT, cur_level->mipmap_factor );
//        plotCell( superblock_bounds, HIGHLIGHT2, /*cur_level->mipmap_factor*/ 1 );


        glm::vec3 stepToParentBoundary( 
                .5f * glm::vec3( cur_level->mipmap_factor-1 ) + .5f * step * glm::vec3( cur_level->mipmap_factor-1 )
                - step * glm::vec3( index & (cur_level->mipmap_factor-1 ) ) );

/*        glm::vec3 minStep( index & (cur_level->mipmap_factor-1) );
        glm::vec3 maxStep( glm::vec3( cur_level->mipmap_factor-1) - minStep );
        glm::vec3 stepToParentBoundary(
                step.x == 1 ? maxStep.x : minStep.x,
                step.y == 1 ? maxStep.y : minStep.y,
                step.z == 1 ? maxStep.z : minStep.z );*/

        glm::vec3 distToParentBoundary( stepToParentBoundary * deltaDist );

        glm::vec3 dist( boxDist + distToParentBoundary );

        glm::bvec3 b0= glm::lessThan( dist, dist.yzx() );
        glm::bvec3 b1= glm::lessThanEqual( dist, dist.zxy() );
        glm::vec3 mask =glm::ivec3( b0.x && b1.x, b0.y && b1.y, b0.z && b1.z );
        glm::vec3 mask2 =glm::abs( glm::vec3(1) - mask );

        glm::vec3 sec_dist =glm::vec3( glm::dot( mask, distToParentBoundary ) ) / deltaDist;
        glm::vec3 multi_step =(sec_dist * mask2) + stepToParentBoundary * mask;

        printf( "step %d: index: (%d,%d,%d) stepToBoundary: (%f, %f, %f) distToBoundary: (%f, %f, %f) multi_step: (%f %f %f)\n",
                s, 
                index.x, index.y, index.z, 
                stepToParentBoundary.x, stepToParentBoundary.y, stepToParentBoundary.z,
                distToParentBoundary.x,distToParentBoundary.y,distToParentBoundary.z,
                multi_step.x, multi_step.y, multi_step.z );

//        if( cur_level->mipmap_factor != 1 ) {
            boxDist += deltaDist * multi_step;
//        index += step * glm::ivec3(mask);
            frag.position_vs += step * glm::floor( multi_step ); 
//            plotCell( glm::floor( frag.position_vs ) , HIGHLIGHT2, 1 );
//        }


        // Branchless equivalent for
/*        for( int i =0; i < 3; i++ ) 
            if( boxDist[side] > boxDist[i] )
                side =i;*/
        
        //while( glm::all( glm::equal( abs_index, index / cur_level->mipmap_factor ) ) ) {
        if( 1 ) {
            glm::vec3 mask;
            if( precise ) {
                glm::bvec3 b0= glm::lessThan( boxDist, boxDist.yzx() );
                glm::bvec3 b1= glm::lessThanEqual( boxDist, boxDist.zxy() );
                mask =glm::ivec3( b0.x && b1.x, b0.y && b1.y, b0.z && b1.z );

            } else {
                glm::vec3 boxDist2 =glm::floor( boxDist );
                glm::bvec3 b0= glm::lessThanEqual( boxDist2, boxDist2.yzx() );
                glm::bvec3 b1= glm::lessThanEqual( boxDist2, boxDist2.zxy() );
                mask =glm::ivec3( b0.x && b1.x, b0.y && b1.y, b0.z && b1.z );
            }
            //side = glm::dot( box_plane, mask );
        
//        mask[side]=1;

        //printf ( "boxdist: %f %f %f mask %f %f %f side %d\n", 
        //    boxDist.x, boxDist.y, boxDist.z,  mask.x, mask.y, mask.z, side );
            boxDist += deltaDist * mask;
            frag.position_vs += step * mask;
            //plotCell( index, HIGHLIGHT2, 1 );
        }
  //      plot( &_plot_index, voxelPx( index ) ); // PLOT
        plotCell( glm::floor( frag.position_vs ) , HIGHLIGHT2, 1 );

    }
    return s;
}

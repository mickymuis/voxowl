#include "voxel.h"

/* Initialize volume and allocate its buffer */
VOXOWL_HOST
void 
voxelmapCreate( voxelmap_t *v, voxel_format_t f, glm::ivec3 size ) {
    v->format =f;
    v->size =size;
    v->data =malloc( voxelmapSize( v ) );
}

VOXOWL_HOST
void 
voxelmapCreate( voxelmap_t *v, voxel_format_t f, uint32_t size_x, uint32_t size_y, uint32_t size_z ) {
    voxelmapCreate( v, f, glm::ivec3( size_x, size_y, size_z ) );
}

/* Free a volume's buffer */
void 
voxelmapFree( voxelmap_t *v ) {
    if( v->data ) free( v->data );
    v->data =NULL;
}

/* Return the size of one voxel in bytes */
VOXOWL_HOST_AND_DEVICE
size_t 
voxelSize( voxel_format_t f ) {
    size_t s;
    switch( f ) {
        case VOXEL_RGBA_UINT32:
            s =4;
            break;
        default:
            s =0;
            break;
    }
    return s;
}

/* Return the size of a volume's data in bytes */
VOXOWL_HOST_AND_DEVICE
size_t 
voxelmapSize( voxelmap_t *v ) {
    return voxelSize( v->format ) * v->size.x * v->size.y * v->size.z;
}

/* Access an array based volume by coordinates. Returns a pointer to the first element */
VOXOWL_HOST_AND_DEVICE
void* 
voxel( voxelmap_t* v, uint32_t x, uint32_t y, uint32_t z ) {
    return (void*)((char*)v->data + v->size.z * v->size.y * x + v->size.z * y + z );
}

VOXOWL_HOST_AND_DEVICE
void* 
voxel( voxelmap_t* v, glm::ivec3 position ) {
    return voxel( v, position.x, position.y, position.z );
}


/* Fills a given volume by copying a value to every position.  */
VOXOWL_HOST_AND_DEVICE
void 
voxelmapFill( voxelmap_t* v, void *value ) {

    uint32_t voxel_size =voxelSize( v->format );
    size_t byte_count =voxelmapSize( v );
    for( size_t i =0; i < byte_count; i+= voxel_size ) {
        memcpy( (char*)v->data + i, value, voxel_size );
    }
    
}

/* Pack an rgba-4float value into an arbitrarily formatted  voxelmap */
VOXOWL_HOST_AND_DEVICE 
void 
voxelmapPack( voxelmap_t* v, glm::ivec3 position, glm::vec4 rgba ) {
   void *elem_ptr =voxel( v, position );
   switch( v->format ) {
        case VOXEL_RGBA_UINT32:
            packRGBA_UINT32( (uint32_t*)elem_ptr, rgba );
   }
}

/* Unpack an rgba-4float value from an arbitrarily formatted  voxelmap */
VOXOWL_HOST_AND_DEVICE 
glm::vec4 
voxelmapUnpack( voxelmap_t* v, glm::ivec3 position ) {
   void *elem_ptr =voxel( v, position );
   glm::vec4 rgba;
   switch( v->format ) {
        case VOXEL_RGBA_UINT32:
            rgba =unpackRGBA_UINT32( *((uint32_t*)elem_ptr) );
   }
   return rgba;
}


VOXOWL_HOST_AND_DEVICE
void
packRGBA_UINT32( uint32_t* rgba, const glm::vec4 &v ) {
    *rgba =
        ((uint32_t)(v.r * 255) << 24) |
        ((uint32_t)(v.g * 255) << 16) |
        ((uint32_t)(v.b * 255) << 8) |
        (uint32_t)(v.a * 255);
}

VOXOWL_HOST_AND_DEVICE
glm::vec4
unpackRGBA_UINT32( uint32_t rgba ) {
    return glm::vec4(
        (float)((rgba >> 24) & 0xff) / 255.f,
        (float)((rgba >> 16) & 0xff) / 255.f,
        (float)((rgba >> 8) & 0xff) / 255.f,
        (float)(rgba & 0xff) / 255.f );
}

VOXOWL_HOST_AND_DEVICE
glm::vec4
blendF2B( glm::vec4 src, glm::vec4 dst ) {
    glm::vec4 c;
    c.r = dst.a*(src.r * src.a) + dst.r;
    c.g = dst.a*(src.g * src.a) + dst.g;
    c.b = dst.a*(src.b * src.a) + dst.b;
    c.a = (1.f - src.a) * dst.a;
    return c;
}

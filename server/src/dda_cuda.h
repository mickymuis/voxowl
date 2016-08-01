#pragma once
#include "raycast.h"
#include "voxel.h"

struct voxelmapDevice_t {
    glm::ivec3 size;
    glm::ivec3 blocks;
    cudaArray *data;
    cudaTextureObject_t texture;
    voxel_format_t format;
};

VOXOWL_DEVICE glm::vec4 voxelTex3D( cudaTextureObject_t &texture, voxel_format_t format, glm::ivec3 index );

VOXOWL_DEVICE glm::vec4 voxelTex3D_clamp( 
        cudaTextureObject_t &texture, 
        voxel_format_t format, 
        glm::ivec3 index, 
        glm::ivec3 clamp_size );

/* Cast one ray r into the volume bounded by v. The actual volume data is obtained from the volume texture */
VOXOWL_DEVICE fragment_t voxelmapRaycast( voxelmapDevice_t *v, const ray_t& r ); 


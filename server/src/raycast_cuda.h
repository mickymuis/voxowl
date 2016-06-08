#pragma once

#include "raycast.h"
#include "renderer.h"
#include "voxel.h"

typedef struct {
    glm::ivec3 size;
    glm::mat4 matModel;
    cudaArray *data;
    voxel_format_t format;
} volumeDevice_t;

typedef struct {
    int width, height;
    cudaArray *data;
    int aaXSamples, aaYSamples;
    voxowl_pixel_format_t format;
} framebufferDevice_t;

class RaycasterCUDA : public Renderer {
    public:
        RaycasterCUDA( const char* name, Object* parent );
        ~RaycasterCUDA();

        bool beginRender();
        bool synchronize();
        
    private:
        bool setCudaErrorStr( cudaError_t code, char *file, int line );

        volumeDevice_t d_volume;
        framebufferDevice_t d_framebuffer;

};

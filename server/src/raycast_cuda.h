#pragma once

#include "raycast.h"
#include "renderer.h"
#include "voxel.h"
#include "svmipmap.h"
#include "dda_cuda.h"
#include "svmm_cuda.h"

typedef enum {
    NOT_INITIALIZED =0,
    VOXELMAP,
    SVMM
} storage_mode_t;

struct volumeDevice_t {
    box_t bounding_box;
    float voxel_width;
    union __volume {
        __volume(){}
        voxelmapDevice_t voxelmap;
        svmipmapDevice_t svmm;
    } volume;
    storage_mode_t mode;
};

typedef struct {
    int width, height;
    short int aaXSamples, aaYSamples;
    voxowl_pixel_format_t format;
    cudaTextureObject_t normal_depth_sampler;
    glm::vec4 clear_color;
} framebufferDevice_t;

typedef struct {
    cudaArray *color_data;
    cudaArray *normal_depth_data;
    framebufferDevice_t fb_d;
} framebuffer_t;

typedef struct {
    int kernelSize;
    int noiseSize;
    cudaArray *sampleKernel;
    cudaArray *noise;
    float radius;
} ssaoInfo_t;

class RaycasterCUDA : public Renderer {
    public:
        RaycasterCUDA( const char* name, Object* parent );
        ~RaycasterCUDA();

        bool beginRender();
        bool synchronize();
        
    private:
        bool initVoxelmap( voxelmap_t *v );
        bool initSVMM( svmipmap_t *svmm );
        bool initFramebuffer();
        
        bool freeVolumeMem();
        bool freeFramebufferMem();
        
        bool setCudaErrorStr( cudaError_t code, const char *file, int line );

        volumeDevice_t d_volume;
        config_t last_config_volume;

        framebuffer_t framebuffer;

        VOXOWL_HOST bool initSSAO();
        ssaoInfo_t ssao_info;

        // Events for performance counting
        cudaEvent_t render_begin;
        cudaEvent_t render_finish;
        cudaEvent_t ssao_step;
        cudaEvent_t ssna_step;
        cudaEvent_t aa_step;
        cudaEvent_t lighting_step;

};

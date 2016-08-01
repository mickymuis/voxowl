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
    glm::mat4 matModel;
    union __volume {
        __volume(){}
        voxelmapDevice_t voxelmap;
        svmipmapDevice_t svmm;
    } volume;
    storage_mode_t mode;
};

typedef struct {
    int width, height;
    cudaArray *color_data;
    cudaArray *normal_depth_data;
    int aaXSamples, aaYSamples;
    voxowl_pixel_format_t format;
} framebufferDevice_t;


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

        framebufferDevice_t d_framebuffer;

        VOXOWL_HOST bool initSSAO();
        ssaoInfo_t ssao_info;

};

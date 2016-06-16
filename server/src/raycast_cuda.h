#pragma once

#include "raycast.h"
#include "renderer.h"
#include "voxel.h"

typedef struct {
    glm::ivec3 size;
    glm::ivec3 blocks;
    glm::mat4 matModel;
    cudaArray *data;
    voxel_format_t format;
} volumeDevice_t;

typedef struct {
    int width, height;
    cudaArray *color_data;
    cudaArray *normal_depth_data;
    int aaXSamples, aaYSamples;
    voxowl_pixel_format_t format;
} framebufferDevice_t;

typedef struct {
    glm::vec4 color; //rgba
    glm::vec3 position;
    glm::vec3 position_vs; // position in continuous voxel space
    glm::vec3 normal;
} fragment_t;

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
        bool setCudaErrorStr( cudaError_t code, char *file, int line );

        volumeDevice_t d_volume;
        framebufferDevice_t d_framebuffer;

        VOXOWL_HOST bool initSSAO();
        ssaoInfo_t ssao_info;

};

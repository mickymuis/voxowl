#ifndef VOXOWL_H
#define VOXOWL_H

#include <inttypes.h>

#define VOXOWL_VERSION_MAJOR 0
#define VOXOWL_VERSION_MINOR 2
#define VOXOWL_VERSION_REVISION 0
#define VOXOWL_VERSION_NAME "Voxowl"
#define VOXOWL_VERSION_FULL_NAME "Voxowl 0.2.0"

#define VOXOWL_DEFAULT_PORT 4007

/* 32 bits magic number that precedes the frame header */
#define VOXOWL_FRAME_MAGIC 0x746f6f68 /* 'hoot' */
/* 32 bits magic number for the fragment header */
#define VOXOWL_FRAGMENT_MAGIC 0x12345678

/* Framebuffer pixel formats */
typedef enum {
    VOXOWL_PF_NONE =0x0,
    VOXOWL_PF_RGB888 =0x1
} voxowl_pixel_format_t;

/* Framebuffer modi */
typedef enum {
    VOXOWL_FBMODE_NONE =0x0,
    VOXOWL_FBMODE_PIXMAP =0x1, // One huge pixmap, not compressed or fragmented
    VOXOWL_FBMODE_JPEG =0x2, // One huge jpeg-compressed image, not fragmented
} voxowl_fbmode_t;

/* Render targets */
typedef enum {
    VOXOWL_TARGET_NONE =0x0,
    VOXOWL_TARGET_FILE =0x1,
    VOXOWL_TARGET_REMOTE =0x2
} voxowl_target_t;

/* Renderers */
#define VOXOWL_RENDERER_NONE 0x0
#define VOXOWL_RENDERER_RAYCAST_CPU 0x1
#define VOXOWL_RENDERER_RAYCAST_CUDA 0x2

struct voxowl_frame_header_t {
    uint32_t magic;             // Magic number, should be VOXOWL_FRAME_MAGIC

    uint8_t seq_num;            // Frame sequence number
    uint8_t pixel_format;       // Pixel format/size, any of VOXOWL_PF_
    uint8_t fb_mode;            // Layout of the framebuffer, any of VOXOWL_FBMODE_
    uint8_t reserved;

    uint16_t width;             // Width in pixels of the entire buffer
    uint16_t height;            // Height in pixels of the entire buffer
    
    uint32_t frame_size;        // Size in bytes of the entire (uncompressed) buffer
    uint16_t n_fragments;        // If the fb_mode is a fragmented type, the total number of fragments
};

struct voxowl_fragment_header_t {
    uint32_t magic;             // Magic number, should be VOXOWL_FRAGMENT_MAGIC
    uint16_t fragment_num;      // Fragment's position in the buffer [0,n_fragments-1] 
    uint16_t fragment_size;     // Data size in bytes that follows after this header
    uint8_t seg_num;            // Sequence number of the entire frame
};

#define VOXOWL_DEFAULT_WIDTH 800
#define VOXOWL_DEFAULT_HEIGHT 600

#endif

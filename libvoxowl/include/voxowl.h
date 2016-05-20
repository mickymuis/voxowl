#ifndef VOXOWL_H
#define VOXOWL_H

#include "inttypes.h"

#define VOXOWL_VERSION_MAJOR 0
#define VOXOWL_VERSION_MINOR 1
#define VOXOWL_VERSION_REVISION 0
#define VOXOWL_VERSION_NAME "Voxowl"
#define VOXOWL_VERSION_DESCRIPTION "Render Server"
#define VOXOWL_VERSION_FULL_NAME "Voxowl Render Server 0.1.0"

/* 32 bits magic number that precedes the data packet */
#define VOXOWL_PACKET_MAGIC 0x746f6f68 /* 'hoot' */

/* Framebuffer pixel formats */
#define VOXOWL_PF_NONE 0x0
#define VOXOWL_PF_RGBA 0x1

/* Framebuffer modi */
#define VOXOWL_FBMODE_NONE 0x0
#define VOXOWL_FBMODE_PIXMAP 0x0

/* Render targets */
#define VOXOWL_TARGET_NONE 0x0
#define VOXOWL_TARGET_FILE 0x1
#define VOXOWL_TARGET_REMOTE 0x2

struct voxowl_frame_header_t {
    uint32_t magic;
    char pixel_format;
    char fb_mode;
    uint16_t width;
    uint16_t height;
    uint32_t frame_size;
};


#endif

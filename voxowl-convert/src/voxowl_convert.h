#pragma once

#include <voxowl_platform.h>
#include <voxowl.h>
#include <voxelmap.h>
#include <voxel.h>
#include <svmipmap.h>

typedef enum {
    FILETYPE_NONE,
    FILETYPE_TIFF,
    FILETYPE_VOXELMAP,
    FILETYPE_SVMM
} filetype_t;

typedef struct {
    char*               in_filename;
    char*               out_filename;
    filetype_t          in_filetype;
    filetype_t          out_filetype;
    bool                silent;
    bool                verify;
    bool                set_baselevel_bitmap;
    int                 set_blockwidth;
    int                 set_rootwidth;
    int                 set_quality;
    bool                set_voxel_format;
    voxel_format_t      voxel_format;
} convert_settings_t;

int convert( convert_settings_t s );

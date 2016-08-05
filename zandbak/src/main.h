#pragma once

enum voxel_value_t {
    EMPTY =0,
    FILLED =1,
    TERMINAL =2
};

typedef struct {
    glm::ivec2 last_px;
    glm::vec4 color;
} plot_t;

typedef struct {
    int mipmap_factor;
    char *data_ptr;
} level_t;

typedef struct {
    glm::ivec3 size;
    level_t* levels;
    int n_levels;
    int blockwidth;
} volume_t;

plot_t newPlot( glm::ivec2 begin_px, glm::vec4 color );

glm::ivec2 voxelPx( glm::ivec3 v );
glm::ivec2 worldspacePx( glm::vec3 v );
glm::vec3 worldspaceFromPx( glm::ivec2 px );

void updateScreen();

void plot( plot_t* p, glm::ivec2 line_to_px );

enum cell_mode_t {
    OUTLINE,
    COLORED,
    HIGHLIGHT,
    HIGHLIGHT2
};

void plotCell( glm::ivec3 v, cell_mode_t mode, int scale =1 );

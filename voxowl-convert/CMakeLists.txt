cmake_minimum_required(VERSION 2.8)

set( LOCAL_BUILD "~/local" )

find_path(TIFF_INCLUDE_DIR tiff.h HINTS ${LOCAL_BUILD}/include)
find_library(TIFF_LIBRARY NAMES tiff HINTS ${LOCAL_BUILD}/lib)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(TIFF DEFAULT_MSG TIFF_LIBRARY TIFF_INCLUDE_DIR)

if(TIFF_FOUND)
        set(TIFF_LIBRARIES ${TIFF_LIBRARY})
        add_definitions(-DHAS_TIFF)
endif()


# Specify include directories
include_directories(
    src
    ../libvoxowl/include
    ${TIFF_INCLUDE_DIR}
    )
# Specify library paths
link_directories( )

#find_package(TIFF REQUIRED HINTS ${LOCAL_BUILD}/include)

# For compilation ...
# Specify target & source files to compile it from
add_executable(
    voxowl-convert
    src/main.cpp
    ../libvoxowl/src/voxel.cu
    ../libvoxowl/src/svmipmap.cu
    ../libvoxowl/src/voxelmap.cpp
    )

# For linking ...
# Specify target & libraries to link it with
target_link_libraries(
    voxowl-convert
    ${TIFF_LIBRARIES}
    )

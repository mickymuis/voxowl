cmake_minimum_required(VERSION 2.8)

# Treat CUDA files a regular C++ sources in this case
#list(APPEND CMAKE_CXX_SOURCE_FILE_EXTENSIONS cu)
# .. but because of a bug in CMAKE this doesn't work. hack:
#set(CMAKE_CXX_SYSROOT_FLAG_CODE "list(APPEND CMAKE_CXX_SOURCE_FILE_EXTENSIONS cu)")
# .. gcc troubles us as well. Force it to compile .cu files
#if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
# add_definitions("-g -x c++")
#endif()

project( voxowl-convert CXX )

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
    ../ext/glm
    ${TIFF_INCLUDE_DIR}
    )
# Specify library paths
link_directories( )

#find_package(TIFF REQUIRED HINTS ${LOCAL_BUILD}/include)

message("CMAKE_CXX_SOURCE_FILE_EXTENSIONS ${CMAKE_CXX_SOURCE_FILE_EXTENSIONS}")

# For compilation ...
# Specify target & source files to compile it from
add_executable(
    voxowl-convert
    src/main.cpp
    src/voxowl_convert.cpp
    ../libsvmm/src/voxel.cpp
    ../libsvmm/src/svmipmap.cpp
    ../libsvmm/src/voxelmap.cpp
    ../libsvmm/src/ivec3_32.cpp
    )

# For linking ...
# Specify target & libraries to link it with
target_link_libraries(
    voxowl-convert
    ${TIFF_LIBRARIES}
    )

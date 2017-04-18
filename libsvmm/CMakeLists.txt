# 
# This file is part of the Voxowl volumetric rendering toolkit
#
# See also: https://github.com/mickymuis/voxowl
#

cmake_minimum_required(VERSION 2.8)

add_library (svmm 
             src/voxel.cpp
             src/voxelmap.cpp
             src/svmipmap.cpp
             src/ivec3_32.cpp)

target_include_directories(svmm PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)

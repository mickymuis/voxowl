
/*
 * Modified version of the BMP writer by Tom Alexander (2012)
 */

#ifndef BMP_H
 
#include <vector>
#include <cstring>
#include <stdint.h>
#include <iostream>

size_t
bitmap_encode_rgba(const uint32_t* rgba, int width, int height, std::vector<uint8_t>& data);

size_t
bitmap_encode_multichannel_8bit(const uint8_t* rgba, int width, int height, int num_channels, std::vector<uint8_t>& data);

size_t
bitmap_make_header( int width, int height, int bpp, std::vector<uint8_t>& data );

#endif

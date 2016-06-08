
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

#endif

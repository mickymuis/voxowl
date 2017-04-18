/*
 * Modified version of the BMP writer by Tom Alexander (2012)
 */
 
#include "bmp.h"

namespace
{
    std::vector<uint8_t> get_4_bytes(const uint32_t & data)
    {
        std::vector<uint8_t> ret;
        uint8_t* tmp = (uint8_t*)&data;
        ret.push_back(tmp[0]);
        ret.push_back(tmp[1]);
        ret.push_back(tmp[2]);
        ret.push_back(tmp[3]);
        return ret;
    }
    void insert_4_bytes(std::vector<uint8_t> & dest, const uint32_t & data)
    {
        std::vector<uint8_t> separated_data = get_4_bytes(data);
        std::copy(separated_data.begin(), separated_data.end(), back_inserter(dest));
    }
    std::vector<uint8_t> get_2_bytes(const uint16_t & data)
    {
        std::vector<uint8_t> ret;
        uint8_t* tmp = (uint8_t*)&data;
        ret.push_back(tmp[0]);
        ret.push_back(tmp[1]);
        return ret;
    }
    void insert_2_bytes(std::vector<uint8_t> & dest, const uint16_t & data)
    {
        std::vector<uint8_t> separated_data = get_2_bytes(data);
        copy(separated_data.begin(), separated_data.end(), back_inserter(dest));
    }
}

/** 
 * Encode an array of RGBA values into an array of bytes that can be written as a bitmap. The input array of RGB values starts at the top left corner. There can be no additional padding in the byte array
 * 
 * @param rgba The array of RGBA values, alpha channel is discarded
 * @param width The width of the image in pixels
 * @param height The height of the image in pixels
 * @param data Reference to an std::vector<> where the output will be stored
 * 
 * @return The number of bytes written to output
 */
size_t 
bitmap_encode_rgba(const uint32_t* rgba, int width, int height, std::vector<uint8_t>& data)
{
    size_t raw_pixel_array_size_offset =bitmap_make_header( width, height, 24, data );
    size_t file_size_offset =2;

    uint32_t size_of_header = data.size();
    for (uint_fast32_t y = 0; y < height; ++y)
    {
        for (uint_fast32_t x = 0; x < width; ++x)
        {
            //Write bottom pixels first since image is flipped
            //Also write pixels in BGR
            uint32_t pixel = rgba[(height-1-y)*(width) + x];
            uint8_t alpha =pixel & 0xFF;
            
            data.push_back( (uint8_t)(((( pixel >> 8 ) & 0xFF) * alpha) / 255 ) );
            data.push_back( (uint8_t)(((( pixel >> 16 ) & 0xFF) * alpha) / 255 ) );
            data.push_back( (uint8_t)(((( pixel >> 24) & 0xFF) * alpha) / 255 ) );
        }
        while ((data.size() - size_of_header)%4)
        {
            data.push_back(0);
        }
    }
    {
        uint32_t file_size = data.size();
        memcpy(&data[file_size_offset], &file_size, 4);
    }
    {
        uint32_t pixel_data_size = data.size() - size_of_header;
        memcpy(&data[raw_pixel_array_size_offset], &pixel_data_size, 4);
    }
    return data.size();
}


size_t
bitmap_encode_multichannel_8bit(const uint8_t* elems, int width, int height, int num_channels, std::vector<uint8_t>& data) {

    size_t raw_pixel_array_size_offset =bitmap_make_header( width, height, num_channels*8, data );
    size_t file_size_offset =2;

    uint32_t size_of_header = data.size();
    for (uint_fast32_t y = 0; y < height; ++y)
    {
        for (uint_fast32_t x = 0; x < width; ++x)
        {
            //Write bottom pixels first since image is flipped
            //Also write pixels in BGR
            // Cycles through the channels adding them in reverse order
            for( int_fast32_t c =num_channels-1; c >= 0; c-- ) {

                uint8_t elem = elems[(height-1-y)*(width)*num_channels + x*num_channels + c];
                data.push_back( elem );
            }
        }
        while ((data.size() - size_of_header)%4)
        {
            data.push_back(0);
        }
    }
    {
        uint32_t file_size = data.size();
        memcpy(&data[file_size_offset], &file_size, 4);
    }
    {
        uint32_t pixel_data_size = data.size() - size_of_header;
        memcpy(&data[raw_pixel_array_size_offset], &pixel_data_size, 4);
    }
    return data.size();
}

size_t
bitmap_make_header( int width, int height, int bpp, std::vector<uint8_t>& data ) {

    data.clear();
    data.push_back(0x42); //B
    data.push_back(0x4D); //M
    size_t file_size_offset = data.size();
    insert_4_bytes(data, 0xFFFFFFFF); //File Size, fill later
    data.push_back(0x00);
    data.push_back(0x00);
    data.push_back(0x00);
    data.push_back(0x00);
    size_t pixel_info_offset_offset = data.size();
    insert_4_bytes(data, 0); //pixel info offset, fill later
    insert_4_bytes(data, 40); //Size of BITMAPINFOHEADER
    insert_4_bytes(data, width);
    insert_4_bytes(data, height);
    insert_2_bytes(data, 1); //Number of color planes
    insert_2_bytes(data, bpp); //Bits per pixel
    insert_4_bytes(data, 0); //No compression
    size_t raw_pixel_array_size_offset = data.size();
    insert_4_bytes(data, 0); //size of raw data in pixel array, fill later
    insert_4_bytes(data, 2835); //Horizontal Resolution
    insert_4_bytes(data, 2835); //Vertical Resolution
    insert_4_bytes(data, 0); //Number of colors
    insert_4_bytes(data, 0); //Important colors
    {
        uint32_t data_size = data.size();
        memcpy(&data[pixel_info_offset_offset], &data_size, 4);
    }
    return raw_pixel_array_size_offset;
}

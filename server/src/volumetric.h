#ifndef VOLUMETRIC_H
#define VOLUMETRIC_H

#include "platform.h"
#include "shadow_macro.h"

template<typename T>
class Volumetric {
    public:
        __host__ __device__
        Volumetric( 
                unsigned int x, 
                unsigned int y, 
                unsigned int z ) : x_size(x), y_size(y), z_size(z) {
            d = new T[size()]();
        }
        __host__ __device__
        Volumetric( 
                unsigned int x, 
                unsigned int y, 
                unsigned int z,
                const T& init ) :x_size(x), y_size(y), z_size(z) {
            d = new T[size()]();
            for( int i =0; i < size(); i++ ) d[i] =init;
        }

        __host__ __device__
        ~Volumetric();

        __host__ __device__
        T &at( 
                unsigned int x, 
                unsigned int y, 
                unsigned int z );

        __host__ __device__
        const T &at( 
                unsigned int x, 
                unsigned int y, 
                unsigned int z ) const;

        __host__ __device__
        unsigned long int size() const { return x_size * y_size * z_size; }

        __host__ __device__
        unsigned int xSize() const { return x_size; }

        __host__ __device__
        unsigned int ySize() const { return y_size; }

        __host__ __device__
        unsigned int zSize() const { return z_size; }

        __host__ __device__
        T* data() { return d; }

        static Volumetric<T>* createOnDevice(
                unsigned int x,
                unsigned int y,
                unsigned int z );

    private:
        unsigned int x_size, y_size, z_size;
        T* d;
};

template<typename T>
__host__ __device__
Volumetric<T>::~Volumetric() {
    delete d;
}

template<typename T>
__host__ __device__
T &
Volumetric<T>::at( 
        unsigned int x, 
        unsigned int y, 
        unsigned int z ) {
    return *(d + x * ySize() * zSize() + y * zSize() + z);
}

template<typename T>
__host__ __device__
const T &
Volumetric<T>::at( 
        unsigned int x, 
        unsigned int y, 
        unsigned int z ) const {
    return *(d + x * ySize() * zSize() + y * zSize() + z);
}

template<typename T> void
__global__ 
__volumetric_device_alloc_helper( 
        Volumetric<T>** v, 
        unsigned int x,
        unsigned int y,
        unsigned int z) {
    *v = new Volumetric<T>( x, y, z );
}

template<typename T> Volumetric<T>** 
Volumetric<T>::createOnDevice(
        unsigned int x,
        unsigned int y,
        unsigned int z ) {
    Volumetric<T>** v;
    cudaMalloc( &v, sizeof( Volumetric<T>** ) );
    __volumetric_device_alloc_helper<<<1,1>>>( v, x, y, z );
    return v;
}

template<typename T> void
__global__ 
__volumetric_device_dealloc_helper( 
        Volumetric<T>** v ) {
    delete *v;
}

template<typename T> void 
Volumetric<T>::deleteFromDevice(
        Volumetric<T>** v ) {
    __volumetric_device_dealloc_helper<<<1,1>>>( v );
    cudaFree( &v );
}

template<typename T> void 
Volumetric<T>::copyFromDevice(
        Volumetric<T>* host_v
        Volumetric<T>** device_v ) {
    Volumetric<T>* device_ptr;
    cudaMemcpy( &device_ptr, device_v, sizeof( Volumetric<T>* ), cudaMemcpyDeviceToHost );
    cudaMemcpy( host_v->data(), device_ptr, cudaMemcpyDeviceToHost );
}

template<typename T> void 
Volumetric<T>::copyToDevice(
        Volumetric<T>** device_v,
        Volumetric<T>* host_v ) {
    Volumetric<T>* device_ptr;
    cudaMemcpy( &device_ptr, device_v, cudaMemcpyDeviceToHost );
    cudaMemcpy( device_ptr ), host_v->data(), cudaMemcpyHostToDevice );
}
#endif

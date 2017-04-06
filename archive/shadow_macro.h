/*
   The SHADOW_MACRO provides a simple means of accessing host and device
   allocated instances in a transparent manner.
   Kindly borrowed from Rob Farber, see
http://www.drdobbs.com/parallel/cuda-unifying-hostdevice-interactions-wi/240161436
*/

#ifndef SHADOW_MACRO_H
#define SHADOW_MACRO_H

#define SHADOW_MACRO(TYPE)                              \
      private:                                      \
  bool usrManaged;                                  \
  TYPE *my_d_ptr;                                   \
public:                                     \
  inline __host__ void set_d_ptr(TYPE* pt) {                    \
           if(!usrManaged && my_d_ptr) cudaFree(my_d_ptr);                \
           my_d_ptr = pt; usrManaged=true;                        \
        }                                         \
  inline __host__ TYPE* d_ptr() {                           \
          if(!my_d_ptr) {cudaMalloc(&my_d_ptr,sizeof(TYPE));usrManaged=false;}    \
          cpyHtoD();                                  \
          return my_d_ptr;                                \
          }                                           \
  inline __host__ void free_d_ptr() {                       \
          if(!usrManaged && my_d_ptr) cudaFree(my_d_ptr);             \
          my_d_ptr = NULL; usrManaged=false;                      \
          }                                           \
  inline __host__ void cpyHtoD() {                      \
          if(!my_d_ptr) my_d_ptr = d_ptr();                       \
          cudaMemcpy(my_d_ptr, this, sizeof(TYPE), cudaMemcpyDefault);        \
        }                                         \
  inline __host__ void cpyDtoH() {                      \
          if(!my_d_ptr) my_d_ptr = d_ptr();                       \
          cudaMemcpy(this, my_d_ptr, sizeof(TYPE), cudaMemcpyDefault);        \
        }
 
#define SHADOW_MACRO_INIT() my_d_ptr=NULL; usrManaged=false;
#define SHADOW_MACRO_CLEAN() free_d_ptr();

#endif

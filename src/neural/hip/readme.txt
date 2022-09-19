hip/hipdnn backend for lc0. Here is a brief description of various files:

1. network_hipdnn.cc -> cpp file containing network, computation, etc stuff related to lc0
2. layers.cc -> cpp files containing layer classes
3. layers.h -> header file for layer classes.
4. kernels.h -> header file for hip kernels
5. common_kernels.cu -> common kernels (fp32, and fp16 that can work with old GPUs)
6. fp16_kernels.cu -> fp16 specific kernels (not used on other GPUs)
7. hip_common.h -> header for common hip stuff like ReportHIPErrors, etc.
8. readme.txt -> this file

High level overview: network is built of layer objects, layers are either implemented using hipdnn/cublas libraries, or custom hip kernels.

lc0 search -> network_hipdnn -> layers -> kernels
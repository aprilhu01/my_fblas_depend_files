//Automatically generated file
//Automatically generated file


#pragma OPENCL EXTENSION cl_intel_channels : enable

#define W 16
#define INCX 1
#define INCY 1
#define INCW 1
#define KERNEL_NAME saxpy
#define CHANNEL_VECTOR_X channel_in_vector_x_0
#define CHANNEL_VECTOR_Y channel_in_vector_y_0
#define CHANNEL_VECTOR_OUT channel_out_vector_0
#define READ_VECTOR_X kernel_read_vector_x_0
#define READ_VECTOR_Y kernel_read_vector_y_0
#define WRITE_VECTOR kernel_write_vector_0
#define __STRATIX_10__

#include <commons.h>
channel TYPE_T CHANNEL_VECTOR_X __attribute__((depth(W)));
channel TYPE_T CHANNEL_VECTOR_Y __attribute__((depth(W)));
channel TYPE_T CHANNEL_VECTOR_OUT __attribute__((depth(W)));

__kernel void KERNEL_NAME(const TYPE_T alpha, int N)
{

    if(N==0) return;

    const int outer_loop_limit=1+(int)((N-1)/W); //ceiling
    TYPE_T res[W];

    for(int i=0; i<outer_loop_limit; i++)
    {
        //receive W elements from the input channels
        #pragma unroll
        for(int j=0;j<W;j++)
            res[j]=alpha*read_channel_intel(CHANNEL_VECTOR_X)+read_channel_intel(CHANNEL_VECTOR_Y);

        //sends the data to a writer
        #pragma unroll
        for(int j=0; j<W; j++)
            write_channel_intel(CHANNEL_VECTOR_OUT,res[j]);
    }

}
/**
    FBLAS: BLAS implementation for Intel FPGA
    Copyright (c) 2019 ETH-Zurich. All rights reserved.
    See LICENSE for license information.

    Reads a vector of type TYPE_T from memory and  push it
    into CHANNEL_VECTOR_X. The vector is accessed with stride INCX.
    The name of the kernel can be redefined by means of preprocessor MACROS.

    W memory reads are performed simultaneously. In the same way W channel push are performed.
    Data is padded to pad_size. Pad_size must be a multiple of W.
    So, for level 1 routines pad_size will be probably equal to W.
    For level 2, in which you have to respect some tiling, it will be equal to a tile size.

    Data is padded using zero elements.

    The vector is sent 'repetitions' times
*/

__kernel void READ_VECTOR_X(__global volatile TYPE_T *restrict data, unsigned int N, unsigned int pad_size, unsigned int repetitions)
{
    unsigned int ratio=pad_size/W;

    unsigned int padding_loop_limit=ceil(((float)N)/pad_size);
    unsigned int outer_loop_limit=padding_loop_limit*ratio;
    TYPE_T x[W];
    for(int t=0; t< repetitions;t++)
    {
        //compute the starting index
        int offset=((INCX) > 0 ?  0 : ((N) - 1) * (-(INCX)));

        for(int i=0;i<outer_loop_limit;i++)
        {
            //prepare data
            #pragma unroll
            for(int k=0;k<W;k++)
            {
                if(i*W+k<N)
                    x[k]=data[offset+(k*INCX)];
                else
                    x[k]=0;
            }
            offset+=W*INCX;

            //send data
            #pragma unroll
            for(int k=0;k<W;k++)
                write_channel_intel(CHANNEL_VECTOR_X,x[k]);
        }
    }
}
/**
    FBLAS: BLAS implementation for Intel FPGA
    Copyright (c) 2019 ETH-Zurich. All rights reserved.
    See LICENSE for license information.

    Reads a vector of type TYPE_T from memory and  push it
    into CHANNEL_VECTOR_Y. The vector is accessed with stride INCY.
    The name of the kernel can be redefined by means of preprocessor MACROS.

    W memory reads are performed simultaneously. In the same way W channel push are performed.
    Data is padded to pad_size. Pad_size must be a multiple of W.
    So, for level 1 routines pad_size will be probably equal to W.
    For level 2, in which you have to respect some tiling, it will be equal to a tile size.

    Data is padded to W using zero elements.

    The vector is sent 'repetitions' times.
*/


__kernel void READ_VECTOR_Y(__global volatile TYPE_T *restrict data, unsigned int N, unsigned int pad_size, unsigned int repetitions)
{

    unsigned int ratio=pad_size/W;
    unsigned int padding_loop_limit=ceil(((float)N)/pad_size);
    unsigned int outer_loop_limit=padding_loop_limit*ratio;
    TYPE_T y[W];
    for(int t=0; t< repetitions;t++)
    {
        //compute the starting index
        int offset=((INCY) > 0 ?  0 : ((N) - 1) * (-(INCY)));
        for(int i=0;i<outer_loop_limit;i++)
        {
            //prepare data
            #pragma unroll
            for(int k=0;k<W;k++)
            {
                if(i*W+k<N)
                    y[k]=data[offset+(k*INCY)];
                else
                    y[k]=0;
            }
            offset+=W*INCY;

            //send data
            #pragma unroll
            for(int k=0;k<W;k++)
                write_channel_intel(CHANNEL_VECTOR_Y,y[k]);
        }
    }
}
/**
    FBLAS: BLAS implementation for Intel FPGA
    Copyright (c) 2019 ETH-Zurich. All rights reserved.
    See LICENSE for license information.


    Write a vector of type TYPE_T into  memory.
    The vector elements are read from channel CHANNEL_VECTOR_OUT.
    The name of the kernel can be redefined by means of preprocessor MACROS.
    INCW represent the access stride.

    W reads are performed simultaneously.
    Data arrives padded at pad_size.
    Padding data (if present) is discarded.
*/

__kernel void WRITE_VECTOR(__global volatile TYPE_T *restrict out, unsigned int N,unsigned int pad_size)
{
    const unsigned int ratio=pad_size/W;
    const unsigned int padding_loop_limit=ceil(((float)N)/pad_size);
    const unsigned int outer_loop_limit=padding_loop_limit*ratio;
    TYPE_T recv[W];
    //compute the starting index
    int offset=((INCW) > 0 ?  0 : ((N) - 1) * (-(INCW)));
    //receive and store data into memory
    for(int i=0;i<outer_loop_limit;i++)
    {
        #pragma unroll
        for(int j=0;j<W;j++)
        {
            recv[j]=read_channel_intel(CHANNEL_VECTOR_OUT);

            if(i*W+j<N)
                out[offset+(j*INCW)]=recv[j];
        }
        offset+=W*INCW;
    }
}

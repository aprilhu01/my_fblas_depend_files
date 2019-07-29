//Automatically generated file
//Automatically generated file

#pragma OPENCL EXTENSION cl_intel_channels : enable

#define W 16
#define INCX 1
#define INCY 1
#define KERNEL_NAME sdot
#define CHANNEL_VECTOR_X channel_in_vector_x_0
#define CHANNEL_VECTOR_Y channel_in_vector_y_0
#define CHANNEL_OUT channel_out_scalar_0
#define READ_VECTOR_X kernel_read_vector_x_0
#define READ_VECTOR_Y kernel_read_vector_y_0
#define WRITE_SCALAR kernel_write_scalar_0
#define __STRATIX_10__

#include <commons.h>

channel TYPE_T CHANNEL_VECTOR_X __attribute__((depth(W)));
channel TYPE_T CHANNEL_VECTOR_Y __attribute__((depth(W)));
channel TYPE_T CHANNEL_OUT __attribute__((depth(1)));


/**
    Performs streaming dot product: data is received through
    CHANNEL_VECTOR_X and CHANNEL_VECTOR_Y. Result is sent
    to CHANNEL_OUT.
*/
__kernel void KERNEL_NAME(int N)
{
    TYPE_T acc_o=0;
    if(N>0)
    {

        const int outer_loop_limit=1+(int)((N-1)/W); //ceiling
        TYPE_T x[W],y[W];

        #ifdef DOUBLE_PRECISION
        TYPE_T shift_reg[SHIFT_REG+1]; //shift register

        for(int i=0;i<SHIFT_REG+1;i++)
           shift_reg[i]=0;
        #endif

        //Strip mine the computation loop to exploit unrolling
        for(int i=0; i<outer_loop_limit; i++)
        {

            TYPE_T acc_i=0;
            #pragma unroll
            for(int j=0;j<W;j++)
            {
                x[j]=read_channel_intel(CHANNEL_VECTOR_X);
                y[j]=read_channel_intel(CHANNEL_VECTOR_Y);
                acc_i+=x[j]*y[j];

            }
            #ifdef DOUBLE_PRECISION
                shift_reg[SHIFT_REG] = shift_reg[0]+acc_i;
                //Shift every element of shift register
                #pragma unroll
                for(int j = 0; j < SHIFT_REG; ++j)
                    shift_reg[j] = shift_reg[j + 1];
            #else
                acc_o+=acc_i;
            #endif

        }

        #ifdef DOUBLE_PRECISION
            //reconstruct the result using the partial results in shift register
            #pragma unroll
            for(int i=0;i<SHIFT_REG;i++)
                acc_o+=shift_reg[i];
        #endif
    }
    else //no computation: result is zero
        acc_o=0.0f;
    //write to the sink
    write_channel_intel(CHANNEL_OUT,acc_o);
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

    Reads a scalar vector of type TYPE_T  and writes it into memory
    The name of the kernel can be redefined by means of preprocessor MACROS.

*/

__kernel void WRITE_SCALAR(__global TYPE_T *restrict out)
{
        *out = read_channel_intel(CHANNEL_OUT);
}

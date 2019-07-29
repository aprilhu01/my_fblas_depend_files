/**
    FBLAS: BLAS implementation for Intel FPGA
    Copyright (c) 2019 ETH-Zurich. All rights reserved.
    See LICENSE for license information.

    Tutorial: the scope of this tutorial is to show how write a simple host program
    that exploits FBLAS routine. In this case it will use SCAL and DOT.
    The program generates two vector x and y randomly.
    Then the even-position elements of x (x[0], x[2]..) are scaled by the factor alpha.
    Finally the dot product between x and y is computed.
 */

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <iostream>
#include <string>
#include <fblas_environment.hpp>
#include "AOCLUtils/opencl.h"
#include "AOCLUtils/aocl_utils.h"
#include "CL/opencl.h"

using namespace std;
#define BLOCKING //comment this for unblocking routine calls



void generate_vector (float *x, int n)
{
    for(int i=0;i<n;i++)
        x[i]= float(rand()) / float(RAND_MAX) * 20.0f - 10.0f; 
        //x[i]= static_cast <double> (rand()) / (static_cast <double> (RAND_MAX/10.0));
}


int main(int argc, char *argv[])
{
    //command line argument parsing
    if(argc<7)
    {
        cerr << "Usage: "<< argv[0]<<" -b <binary file> -j <json file> -n <length of the vectors> "<<endl;
        exit(-1);
    }

    int c;
    int n;
    //double alpha;
    std::string program_path, json_path;
    while ((c = getopt (argc, argv, "n:j:b:")) != -1)
        switch (c)
        {
            case 'n':
                n=atoi(optarg);
                break;
            case 'b':
                program_path=std::string(optarg);
                break;
            case 'j':
                json_path=std::string(optarg);
                break;
            default:
                cerr << "Usage: "<< argv[0]<<" -b <binary file> -j <json file> -n <length of the vectors> -a <alpha>"<<endl;
                exit(-1);
        }

    //create data
    float *x,*y;
    float res,cpu_res;
    posix_memalign ((void **)&x, IntelFPGAOCLUtils::AOCL_ALIGNMENT, n*sizeof(float));
    posix_memalign ((void **)&y, IntelFPGAOCLUtils::AOCL_ALIGNMENT, n*sizeof(float));
    generate_vector(x,n);
    generate_vector(y,n);

    //create FBLAS environment
    FBLASEnvironment fb(program_path,json_path);

    //get context and device
    cl::Context context=fb.get_context();
    cl::Device device=fb.get_device();
    cl::CommandQueue queue;
    IntelFPGAOCLUtils::createCommandQueue(context,device,queue);

    //create buffer over fpga
    cl::Buffer fpga_x(context, CL_MEM_READ_ONLY|CL_CHANNEL_1_INTELFPGA, n *sizeof(float));
    cl::Buffer fpga_y(context, CL_MEM_READ_ONLY|CL_CHANNEL_2_INTELFPGA, n * sizeof(float));
    cl::Buffer fpga_res(context, CL_MEM_READ_WRITE|CL_CHANNEL_3_INTELFPGA,  sizeof(float));

    //copy data
    queue.enqueueWriteBuffer(fpga_x,CL_TRUE,0,n*sizeof(float),x);
    queue.enqueueWriteBuffer(fpga_y,CL_TRUE,0,n*sizeof(float),y);
   
    const double start_time = aocl_utils::getCurrentTimestamp();
#if defined(BLOCKING)
    fb.sdot("sdot",n,fpga_x,1,fpga_y,1,fpga_res);
    queue.enqueueReadBuffer(fpga_res,CL_TRUE,0,sizeof(float),&res);
#else
    std::vector<cl::Event> dot_event;
    cl::Event e;

    fb.sdot("sdot",n,fpga_x,1,fpga_y,1,fpga_res,&nullptr,&e);
    dot_event.push_back(e);

    queue.enqueueReadBuffer(fpga_res,CL_TRUE,0,sizeof(float),&res,&dot_event);

#endif

    //calculate time
    const double end_time = aocl_utils::getCurrentTimestamp();
    const double total_time = end_time - start_time;
    printf("\nTime: %0.3f ms\n", total_time * 1e3);
   
    //check
    cpu_res=0;
    for(int i=0;i<n;i++)
    {
            cpu_res+=x[i]*y[i];
    }
  
    const float dif=(cpu_res-res)*(cpu_res-res);
    const float ref=res*res;
    const float error=sqrtf(dif)/sqrtf(ref);
    //if(error>1e-6)
    //    cout << "Error: " <<cpu_res<<" != " <<res<<endl;
    //else
    //    cout << "Result is correct: "<< res<<endl;
    cout << "When vector size is "<<n<<": "<<endl;
    cout << "Result is: "<<res<<endl;
    cout << "Relative error is: "<<error <<endl;
}

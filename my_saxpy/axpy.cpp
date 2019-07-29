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
    if(argc<9)
    {
        cerr << "Usage: "<< argv[0]<<" -b <binary file> -j <json file> -n <length of the vectors> -a <alpha> "<<endl;
        exit(-1);
    }

    int c;
    int n;
    double alpha;
    std::string program_path, json_path;
    cout<<"Step-4......"<<endl;
    while ((c = getopt (argc, argv, "n:j:b:a:")) != -1)
        switch (c)
        {
            case 'n':
                n=atoi(optarg);
                break;
            case 'b':
                program_path=std::string(optarg);
                break;
            case 'a':
                alpha=atof(optarg);
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
    float *res,*cpu_res;
    cout<<"Step-3......"<<endl;
    posix_memalign ((void **)&x, IntelFPGAOCLUtils::AOCL_ALIGNMENT, n*sizeof(float));
    posix_memalign ((void **)&y, IntelFPGAOCLUtils::AOCL_ALIGNMENT, n*sizeof(float));
    posix_memalign ((void **)&res, IntelFPGAOCLUtils::AOCL_ALIGNMENT, n*sizeof(float));
    posix_memalign ((void **)&cpu_res, IntelFPGAOCLUtils::AOCL_ALIGNMENT, n*sizeof(float));
    generate_vector(x,n);
    generate_vector(y,n);

    //create FBLAS environment
    FBLASEnvironment fb(program_path,json_path);
    cout<<"Step-2......"<<endl;

    //get context and device
    cl::Context context=fb.get_context();
    cl::Device device=fb.get_device();
    cl::CommandQueue queue;
    IntelFPGAOCLUtils::createCommandQueue(context,device,queue);
    cout<<"Step-1......"<<endl;

    //create buffer over fpga
    cl::Buffer fpga_x(context, CL_MEM_READ_WRITE|CL_CHANNEL_1_INTELFPGA, n *sizeof(float));
    cl::Buffer fpga_y(context, CL_MEM_READ_WRITE|CL_CHANNEL_2_INTELFPGA, n * sizeof(float));
   // cl::Buffer fpga_res(context, CL_MEM_WRITE_ONLY|CL_CHANNEL_3_INTELFPGA,  sizeof(float));
    cout<<"Step0......"<<endl;

    //copy data
    queue.enqueueWriteBuffer(fpga_x,CL_TRUE,0,n*sizeof(float),x);
    queue.enqueueWriteBuffer(fpga_y,CL_TRUE,0,n*sizeof(float),y);
   
    const double start_time = aocl_utils::getCurrentTimestamp();
#if defined(BLOCKING)
    //compute the axpy product
    fb.saxpy("saxpy",n,alpha,fpga_x,1,fpga_y,1);
    cout<<"Step1......"<<endl;
    queue.enqueueReadBuffer(fpga_y,CL_TRUE,0,n*sizeof(float),res);
    const double end_time =aocl_utils::getCurrentTimestamp();
#else
    std::vector<cl::Event> axpy_event;
    cl::Event e;
    cout<<"Step2......"<<endl;
    fb.saxpy("saxpy",n,alpha,fpga_x,1,fpga_y,1,&nullptr,&e);
    axpy_event.push_back(e);

    queue.enqueueReadBuffer(fpga_y,CL_TRUE,0,sizeof(float),res,&axpy_event);
    const double end_time = aocl_utils::getCurrentTimestamp();

#endif

    //calculate time

    double total_time = end_time - start_time;
    printf("\nTime: %0.3f ms\n", total_time * 1e3);
   
    //checu
    float dif=0.0f;
    float ref=0.0f;
    cout<<"verifying..."<<endl;
    for(int i=0;i<n;i++)
    {
        cpu_res[i]=x[i]*alpha + y[i];
        dif+=(cpu_res[i]-res[i])*(cpu_res[i]-res[i]);
        ref+=res[i]*res[i];
    }
  
    const float error=sqrtf(dif)/sqrtf(ref);
    //if(error>1e-6)
    //    cout << "Error: " <<cpu_res<<" != " <<res<<endl;
    //else
    //    cout << "Result is correct: "<< res<<endl;
    cout << "When vector size is "<<n<<": "<<endl;
    cout << "Error is: "<<dif<<endl;
    cout << "Relative error is: "<<error <<endl;
    cout << "Total time is: "<<total_time<<" ms" <<endl;
}

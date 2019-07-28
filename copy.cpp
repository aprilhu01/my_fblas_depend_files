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
        x[i] =float(rand()) / float(RAND_MAX) * 20.0f - 10.0f;
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
    float cpu_res,dif,ref;
    posix_memalign ((void **)&x, IntelFPGAOCLUtils::AOCL_ALIGNMENT, n*sizeof(float));
    posix_memalign ((void **)&y, IntelFPGAOCLUtils::AOCL_ALIGNMENT, n*sizeof(float));
    generate_vector(x,n);
    //generate_vector(y,n);

   //create FBLAS environment
    FBLASEnvironment fb(program_path,json_path);

    //get context and device
    cl::Context context=fb.get_context();
    cl::Device device=fb.get_device();
    cl::CommandQueue queue;
    IntelFPGAOCLUtils::createCommandQueue(context,device,queue);

    //create buffer over fpga
    cl::Buffer fpga_x(context, CL_MEM_READ_WRITE|CL_CHANNEL_1_INTELFPGA, n *sizeof(float));
    cl::Buffer fpga_y(context, CL_MEM_READ_WRITE|CL_CHANNEL_2_INTELFPGA, n * sizeof(float));

    //copy data
    queue.enqueueWriteBuffer(fpga_x,CL_TRUE,0,n*sizeof(float),x);
    queue.enqueueWriteBuffer(fpga_y,CL_TRUE,0,n*sizeof(float),y);
    
    const double start_time = aocl_utils::getCurrentTimestamp();
    
#if defined(BLOCKING)
    //copy x to y
    fb.scopy("scopy",n,fpga_x,1,fpga_y,1);
    
    queue.enqueueReadBuffer(fpga_y,CL_TRUE,0,n*sizeof(float),y);
    const double end_time = aocl_utils::getCurrentTimestamp();
#else
    std::vector<cl::Event> copy_event;// copy_event_3;
    cl::Event e;
    fb.scopy("scopy",n,fpga_x,1,fpga_y,1,nullptr,&e);
    copy_event.push_back(e);

    queue.enqueueReadBuffer(fpga_y,CL_TRUE,0,n*sizeof(float),y,&copy_event);
    const double end_time = aocl_utils::getCurrentTimestamp();
#endif

    //calculate time
    const double total_time = end_time - start_time;
    printf("\nTime: %0.3f ms\n", total_time * 1e3);

    //copy back the result

    //check
    ref=0.0f;
    dif=0.0f;
    for(int i=0;i<n;i++)
    {
            //cout << "x[ "<<i<<"] = " <<x[i]<<endl;
            //cout << "y[ "<<i<<"] = " <<y[i]<<endl;
            dif+=(x[i]-y[i])*(x[i]-y[i]);
            ref+=y[i]*y[i];
    }
    const float error = sqrtf(dif)/sqrtf(ref);
    if(error>=1e-6)
        cout << "Error: " <<error<<" >= requirment " <<endl;
    else
        cout << "Okay"<<endl;

}

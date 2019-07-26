/*sgemv.cpp*/

//void FBLASEnvironment::gemv(std::string routine_name, FblasTranspose transposed, unsigned int N, unsigned int M, T alpha, cl::Buffer A,
//                            unsigned int lda, cl::Buffer x, int incx, T beta, cl::Buffer y, int incy, std::vector<cl::Event> *events_wait_list, cl::Event *event)
/*********************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <iostream>
#include <string>
#include <fblas_environment.hpp>

using namespace std;
#define BLOCKING //comment this for unblocking routine calls



void generate_vector (float *x, int n)
{
    for(int i=0;i<n;i++)
        x[i]= static_cast <double> (rand()) / (static_cast <double> (RAND_MAX/10.0));
}

template <typename T>
void generate_matrix(T *A,int N,int M)
{
    for(int i=0;i<N;i++)
    {
	//fill column[i+1]
        for(int j=0;j<M;j++)
            A[i*M+j] = static_cast <float> (rand()) / (static_cast <float> (RAND_MAX/MAX_NUMB));
    }
}

int main(int argc, char *argv[])
{

    //command line argument parsing
    if(argc<11)
    {
        cerr << "Usage: "<< argv[0]<<" -b <binary file> -j <json file> -n <columns of matrix> -m <rows of matrix> -a <alpha>"<<endl;
        exit(-1);
    }

    int c;
    //n is width of matix and m is height
    int n,m;
    double alpha;
    std::string program_path, json_path;
    while ((c = getopt (argc, argv, "n:m:j:b:a:")) != -1)
        switch (c)
        {
            case 'm':
                m=atoi(optarg);
                break;            
            case 'n':
                n=atoi(optarg);
                break;
            case 'a':
                alpha=atof(optarg);
                break;
            case 'b':
                program_path=std::string(optarg);
                break;
            case 'j':
                json_path=std::string(optarg);
                break;
            default:
                cerr << "Usage: "<< argv[0]<<" -b <binary file> -j <json file> -n <column of the matix> -m <row of the matix> -a <alpha>"<<endl;
                exit(-1);
        }

    //create data
    float *x,*y,*A;
    float *res,*cpu_res;

    posix_memalign ((void **)&x, IntelFPGAOCLUtils::AOCL_ALIGNMENT, n*sizeof(float));
    posix_memalign ((void **)&y, IntelFPGAOCLUtils::AOCL_ALIGNMENT, m*sizeof(float));
    posix_memalign ((void **)&A, IntelFPGAOCLUtils::AOCL_ALIGNMENT, n*m*sizeof(float));
    posix_memalign ((void **)&cpu_res, IntelFPGAOCLUtils::AOCL_ALIGNMENT, m*sizeof(float));
    posix_memalign ((void **)&res, IntelFPGAOCLUtils::AOCL_ALIGNMENT, m*sizeof(float));


    generate_matrix<float>(x,1,n); //x has n rows and 1 column
    generate_matrix<float>(y,m,1); //y has 1 row and m columns
    generate_matrix<float>(A,n,m); //A has n columns and m rows

    //create FBLAS environment
    FBLASEnvironment fb(program_path,json_path);

    //get context and device
    cl::Context context=fb.get_context();
    cl::Device device=fb.get_device();
    cl::CommandQueue queue;
    IntelFPGAOCLUtils::createCommandQueue(context,device,queue);

    //create buffer over fpga
    cout<<"creating buffer..."<<endl;
    cl::Buffer fpga_A(context, CL_MEM_READ_ONLY|CL_CHANNEL_1_INTELFPGA, n * m*sizeof(float))
    cl::Buffer fpga_x(context, CL_MEM_READ_ONLY|CL_CHANNEL_2_INTELFPGA, m *sizeof(float));
    cl::Buffer fpga_y(context, CL_MEM_READ_WRITE|CL_CHANNEL_3_INTELFPGA, n * sizeof(float));

    //copy data
    cout<<"Copying..."<<endl;
    queue.enqueueWriteBuffer(fpga_x,CL_TRUE,0,m*sizeof(float),x);
    queue.enqueueWriteBuffer(fpga_y,CL_TRUE,0,m*sizeof(float),y);
    queue.enqueueWriteBuffer(fpga_A,CL_TRUE,0,n*m*sizeof(float),A);
#if defined(BLOCKING)
    //A * x + 0*y
    cout<<"Running..."<<endl;
    fb.gemv("sgemv", FBLAS_NO_TRANSPOSED, n, m, alpha, fpga_A, m, fpga_x, 1, 0, fpga_y, 1);
    queue.enqueueReadBuffer(fpga_res,CL_TRUE,0,m*sizeof(float),res);
#else
    std::vector<cl::Event> gemv_event;
    cl::Event e;
    fb.gemv("sgemv", FBLAS_NO_TRANSPOSED, n, m, alpha, fpga_A, n*m, fpga_x, 1, 0, fpga_y, 1, &e,nullptr);
    gemv_event.push_back(e);

    queue.enqueueReadBuffer(fpga_res,CL_TRUE,0,m*sizeof(float),res,&gemv_event);

#endif


    //copy back the result

    //check
    cout<<"verifying..."<<endl;
    float dif = 0.0f;
    float ref = 0.0f;
    float error = 0.0f;
    for(unsigned i = 0; i < m; i++)
    {
	    float sum = 0.0f;
	    for(unsigned j = 0; j < n; j++){
		sum += A[n * i + j]* x[j];
		}
	    cpu_res[i] = sum;
	}
	for(unsigned i = 0; i <m;i++)
	{
	    const float o = res[i];
        const float r = cpu_res[i];
        const float d = o - r;
        diff += d * d;
        ref += r * r;
    }
    error=sqrtf(dif)/sqrtf(ref);
    cout<<"the error is"<<error<<endl;


    
}

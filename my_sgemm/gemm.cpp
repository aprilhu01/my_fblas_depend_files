/*sgemm.cpp*/
// void FBLASEnvironment::sgemm(std::string routine_name, FblasTranspose transA, FblasTranspose transB, unsigned int N, unsigned int M, unsigned int K,
//            float alpha, cl::Buffer A, unsigned int lda, cl::Buffer B, unsigned int ldb, float beta, cl::Buffer C, unsigned int ldc,
//            std::vector<cl::Event> * events_wait_list, cl::Event * event )
/*********************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <iostream>
#include <string>
#include <fblas_environment.hpp>
#include "AOCLUtils/opencl.h"
#include "AOCLUtils/aocl_utils.h"
#include "CL/opencl.h"
#include "CL/opencl.h"

using namespace std;
#define BLOCKING //comment this for unblocking routine calls


void generate_vector (float *x, int n)
{
    for(int i=0;i<n;i++)
        x[i]= static_cast <double> (rand()) / (static_cast <double> (RAND_MAX/10.0));
}

template <typename T>
void generate_matrix(T *A,int column,int row)
{
    //A[col(i+1),row(j+1)]=A[m*i+j]
    for(int i=0;i<column;i++)
    {
	//fill column[i+1]
        for(int j=0;j<row;j++)
            A[i*row+j] = static_cast <float> (rand()) / (static_cast <float> (RAND_MAX/10.0));
    }
}

int main(int argc, char *argv[])
{

    //command line argument parsing
    if(argc<15)
    {
        cerr << "Usage: "<< argv[0]<<" -b <binary file> -j <json file> -n <column of matix B> -m <row of the matix A> -k <column of the matix A> -B<beta> -a <alpha>"<<endl;
        exit(-1);
    }

    int c; 
    //m -> rows of A  n -> column of B  k -> COL(A)=ROW(B)
    unsigned int n,m,k;  
    double alpha,beta;
    std::string program_path, json_path;
    while ((c = getopt (argc, argv, "n:m:k:j:b:a:B:")) != -1)
        switch (c)
        {
            case 'm':
                m=atoi(optarg);
                break;            
            case 'n':
                n=atoi(optarg);
                break;
            case 'k':
                k=atoi(optarg);
                break;
            case 'B':
                beta=atoi(optarg);
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
                cerr << "Usage: "<< argv[0]<<" -b <binary file> -j <json file> -n <column of matix B> -m <row of the matix A> -k <column of the matix A> -B<beta> -a <alpha>"<<endl;
                exit(-1);
        }
    cout<<"m= "<<m<<"    n= "<<n<<"    k= "<<k<<"    alpha= "<<alpha<<"   beta= "<<beta<<endl;
    
    //create data
    float *A,*B,*C;
    float *res,*cpu_res;

    posix_memalign ((void **)&A, IntelFPGAOCLUtils::AOCL_ALIGNMENT, k*m*sizeof(float));
    posix_memalign ((void **)&B, IntelFPGAOCLUtils::AOCL_ALIGNMENT, n*k*sizeof(float));
    posix_memalign ((void **)&C, IntelFPGAOCLUtils::AOCL_ALIGNMENT, m*n*sizeof(float));
    posix_memalign ((void **)&cpu_res, IntelFPGAOCLUtils::AOCL_ALIGNMENT, m*n*sizeof(float));
    posix_memalign ((void **)&res, IntelFPGAOCLUtils::AOCL_ALIGNMENT, m*n*sizeof(float));


    generate_matrix<float>(A,k,m); //A has m rows and k columnS
    generate_matrix<float>(B,n,k); //B has k rows and n columns
    generate_matrix<float>(C,n,m); //C has m rows and n columns
    
    /*for(int i=0;i<k;i++)
    {
        for(int j=0;j<m;j++)
        {
            cout<<"A["<<i<<"*"<<m<<"+"<<j<<"] = "<<A[i*m+j]<<endl;
	}
	for(int j=0;j<n;j++)
	{
	    cout<<"B["<<j<<"*"<<k<<"+"<<i<<"] = "<<B[j*k+i]<<endl;
        }
    }*/

    //create FBLAS environment
    FBLASEnvironment fb(program_path,json_path);

    //get context and device
    cl::Context context=fb.get_context();
    cl::Device device=fb.get_device();
    cl::CommandQueue queue;
    IntelFPGAOCLUtils::createCommandQueue(context,device,queue);

    //create buffer over fpga
    cout<<"creating buffer..."<<endl;
    cl::Buffer fpga_A(context, CL_MEM_READ_ONLY|CL_CHANNEL_1_INTELFPGA, m*k*sizeof(float));
    cl::Buffer fpga_B(context, CL_MEM_READ_ONLY|CL_CHANNEL_2_INTELFPGA, k*n*sizeof(float));
    cl::Buffer fpga_C(context, CL_MEM_READ_WRITE|CL_CHANNEL_3_INTELFPGA, m*n*sizeof(float));

    //copy data
    cout<<"Copying..."<<endl;
    queue.enqueueWriteBuffer(fpga_A,CL_TRUE,0,m*k*sizeof(float),A);
    queue.enqueueWriteBuffer(fpga_B,CL_TRUE,0,k*n*sizeof(float),B);
    queue.enqueueWriteBuffer(fpga_C,CL_TRUE,0,m*n*sizeof(float),C);
	
    const double start_time = aocl_utils::getCurrentTimestamp();
#if defined(BLOCKING)
    //alpha*A * B + beta*C
    cout<<"Running..."<<endl;
    fb.sgemm("sgemm", FBLAS_NO_TRANSPOSED, FBLAS_NO_TRANSPOSED, n,  m, k, alpha, fpga_A, m, fpga_B, k, beta, fpga_C, m);
    //copy back the result
    queue.enqueueReadBuffer(fpga_C,CL_TRUE,0,m*n*sizeof(float),res);
    const double end_time = aocl_utils::getCurrentTimestamp();
#else
    std::vector<cl::Event> gemm_event;
    cl::Event e;
    fb.sgemm("sgemm", FBLAS_NO_TRANSPOSED, FBLAS_NO_TRANSPOSED, n,  m, k, alpha, fpga_A, m, fpga_B, k, beta, fpga_C, m, nullptr, &e)
    gemm_event.push_back(e);

    queue.enqueueReadBuffer(fpga_C,CL_TRUE,0,m*n*sizeof(float),res,&gemm_event);
    const double end_time = aocl_utils::getCurrentTimestamp();

#endif

    const double total_time = end_time - start_time;
    printf("\nTime: %0.3f ms\n", total_time * 1e3);

    //check
    cout<<"verifying..."<<endl;
    float dif = 0.0f;
    float ref = 0.0f;
    float error = 0.0f;

    //specila attention here!
    for (int l = 0; l < k; l++) {
        for (int i = 0; i < m; i++) {
            float temp = alpha * A[m * l + i];
      
            if (temp != 0.0) {
                for (int j = 0; j < n; j++) {
                    cpu_res[m * j + i] += temp * B[j* k + l];
		    //cout<<"B["<<j<<"*"<<k<<"+"<<l<<"]       = "<<B[j*k+l]<<endl;
		    //cout<<"temp= "<<temp<<endl;
                }
                //cout<<endl;
            }
        }
    }
    
    for(unsigned i = 0; i <n;i++)
    {
        for(unsigned j = 0; j<m;j++)
        {
            const float o = res[i*m+j];
            const float r = cpu_res[i*m+j];
            const float d = o - r;
            dif += d * d;
            ref += r * r;
	    //cout<<"res["<<i<<"*"<<m<<"+"<<j<<"]     = "<<res[i*m+j]<<endl;
            //cout<<"cpu_res["<<i<<"*"<<m<<"+"<<j<<"] = "<<cpu_res[i*m+j]<<endl;
        }
	   
    }
    error=sqrtf(dif)/sqrtf(ref);
    cout<<"the error is"<<error<<endl;


    
}

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
    //A(i,j)=A[m*i+j]
    for(int i=0;i<column;i++)
    {
	//fill column[i+1]
        for(int j=0;j<row;j++)
            A[i*row+j] = static_cast <float> (rand()) / (static_cast <float> (RAND_MAX/MAX_NUMB));
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
    int n,m,k;  
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
#if defined(BLOCKING)
    //alpha*A * B + beta*y
    cout<<"Running..."<<endl;
    fb.sgemm(context, FBLAS_NO_TRANSPOSED, FBLAS_NO_TRANSPOSED, n,  m, k, alpha, fpga_A, m, fpga_B, k, beta, C, m)
    //copy back the result
    queue.enqueueReadBuffer(fpga_res,CL_TRUE,0,m*sizeof(float),res);
#else
    std::vector<cl::Event> gemm_event;
    cl::Event e;
    fb.sgemm(context, FBLAS_NO_TRANSPOSED, FBLAS_NO_TRANSPOSED, n,  m, k, alpha, fpga_A, m, fpga_B, k, beta, fpga_C, m, nullptr, &e)
    gemm_event.push_back(e);

    queue.enqueueReadBuffer(fpga_C,CL_TRUE,0,m*sizeof(float),res,&gemm_event);

#endif



    //check
    cout<<"verifying..."<<endl;
    float dif = 0.0f;
    float ref = 0.0f;
    float error = 0.0f;

    //specila attention here!
    for (int x = 0; x < k; x++) {
        for (int i = 0; i < n; i++) {
            float temp = alpha * A[m * i + k];
            if (temp != 0.0) {
                for (int j = 0; j < m; j++) {
                    cpu_res[m * i + j] += temp * B[k * x + j];
                }
            }
        }
    }

    
	for(unsigned i = 0; i <m;i++)
	{
	    const float o = res[i];
        const float r = cpu_res[i];
        const float d = o - r;
        dif += d * d;
        ref += r * r;
    }
    error=sqrtf(dif)/sqrtf(ref);
    cout<<"the error is"<<error<<endl;


    
}

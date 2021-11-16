#include<stdio.h>
#include<time.h>
#include<math.h>
#include<stdlib.h>
#include<cuda.h>
#define SZ 16384
#define L 1024
#define T 32

__global__ void distribute(float *A11,float *A12,float *A21,float *A22,float *A,int K){

    int x = threadIdx.x + blockIdx.x*blockDim.x;
    int y = threadIdx.y + blockIdx.y*blockDim.y;

    if(y<K && x<K){
        A11[y*K + x] = A[y*K*2 + x]; //Because N = 2*K and A[y][x] = A[y*N + x] = A[y*K*2 + x] therefore we multiply with 2 to map correctly.
        A12[y*K + x] = A[y*K*2 + x + K];
        A21[y*K + x] = A[(y + K)*K*2 + x];
        A22[y*K + x] = A[(y + K)*K*2 + x + K];
    }
     __syncthreads();
 }
 __global__ void add(float *A, float *B, float *C, int K){
    int x = threadIdx.x + blockIdx.x*blockDim.x;
    int y = threadIdx.y + blockIdx.y*blockDim.y;

    if(y<K && x<K){
        C[y*K + x] = A[y*K + x] + B[y*K + x];
    }
     __syncthreads();
 }
 __global__ void sub(float *A, float *B, float *C, int K){
    int x = threadIdx.x + blockIdx.x*blockDim.x;
    int y = threadIdx.y + blockIdx.y*blockDim.y;

    if(y<K && x<K){
        C[y*K + x] = A[y*K + x] - B[y*K + x];
    }
     __syncthreads();
 }
 __global__ void gather(float *A11,float *A12,float *A21,float *A22,float *A,int K){
    int x = threadIdx.x + blockIdx.x*blockDim.x;
    int y = threadIdx.y + blockIdx.y*blockDim.y;

    if(y<K && x<K){
        A[y*K*2 + x] = A11[y*K + x];
        A[y*K*2 + x + K] = A12[y*K + x];
        A[(y + K)*K*2 + x] = A21[y*K + x];
        A[(y + K)*K*2 + x + K] = A22[y*K + x];
    }
     __syncthreads();
 }
 __global__ void trivial_parallel(float *A, float* B, float* C, int N){
    int x = threadIdx.x + blockIdx.x*blockDim.x;
    int y = threadIdx.y + blockIdx.y*blockDim.y;

    if(y<N && x<N){
        int i;
        float tot = 0;
        for(i=0;i<N;i++){
            tot+=A[y*N + i]*B[i*N + x];
        }
        C[y*N + x] = tot;
    }
     __syncthreads();
 }
void strassen(float *A, float *B, float *C,int N){

    float *A_G,*B_G,*C_G;

    int size_b = N*N*sizeof(float);
   

    cudaMalloc((void**)&A_G, size_b);
    cudaMalloc((void**)&B_G, size_b);
    cudaMalloc((void**)&C_G, size_b);

    cudaMemcpy(A_G,A,size_b,cudaMemcpyHostToDevice);
    cudaMemcpy(B_G,B,size_b,cudaMemcpyHostToDevice);


    if(N<=L){
        dim3 dimblock(T,T);
        dim3 dimgrid((N+T-1)/T,(N+T-1)/T); 
        trivial_parallel<<<dimgrid,dimblock>>>(A_G,B_G,C_G,N);
    }
     
    else{
        int K = N/2; //Split into 4 parts so K = N/2 or N = 2*K

        int size_s = K*K*sizeof(float);

        float *A11,*A12,*A21,*A22,*B11,*B12,*B21,*B22,*C11,*C12,*C21,*C22,*T1,*T2; // Declare Device var
        
        cudaMalloc((void**)&A11, size_s);
        cudaMalloc((void**)&A12, size_s);
        cudaMalloc((void**)&A21, size_s);
        cudaMalloc((void**)&A22, size_s);

        cudaMalloc((void**)&B11, size_s);
        cudaMalloc((void**)&B12, size_s);
        cudaMalloc((void**)&B21, size_s);
        cudaMalloc((void**)&B22, size_s);

        cudaMalloc((void**)&C11, size_s);
        cudaMalloc((void**)&C12, size_s);
        cudaMalloc((void**)&C21, size_s);
        cudaMalloc((void**)&C22, size_s);

        cudaMalloc((void **)&T1, size_s);
        cudaMalloc((void **)&T2, size_s);

        
        dim3 dimblock(T,T);
        dim3 dimgrid((K+T-1)/T,(K+T-1)/T); 

        //Utilize only 2 temporary variables T1,T2 to reduce temporary storage per level significantly and avoid unnecessary data transfer.
        //Refer Sahni's paper.

        distribute<<<dimgrid,dimblock>>>(A11,A12,A21,A22,A_G,K);
        distribute<<<dimgrid,dimblock>>>(B11,B12,B21,B22,B_G,K);

        sub<<<dimgrid,dimblock>>>(A21,A11,C12,K);
        add<<<dimgrid,dimblock>>>(B11,B12,C21,K);
        strassen(C12,C21,C22,K);

        sub<<<dimgrid,dimblock>>>(A12,A22,C12,K);
        add<<<dimgrid,dimblock>>>(B21,B22,C21,K);
        strassen(C12,C21,C11,K);

        add<<<dimgrid,dimblock>>>(A11,A22,C12,K);
        add<<<dimgrid,dimblock>>>(B11,B22,C21,K);
        strassen(C12,C21,T1,K);

        add<<<dimgrid,dimblock>>>(T1,C11,C11,K);
        add<<<dimgrid,dimblock>>>(T1,C22,C22,K);
        add<<<dimgrid,dimblock>>>(A21,A22,T2,K);
        strassen(T2,B11,C21,K);

        sub<<<dimgrid,dimblock>>>(C22,C21,C22,K);
        sub<<<dimgrid,dimblock>>>(B21,B11,T1,K);
        strassen(A22,T1,T2,K);

        add<<<dimgrid,dimblock>>>(C21,T2,C21,K);
        add<<<dimgrid,dimblock>>>(C11,T2,C11,K);
        sub<<<dimgrid,dimblock>>>(B12,B22,T1,K);
        strassen(A11,T1,C12,K);

        add<<<dimgrid,dimblock>>>(C22,C12,C22,K);
        add<<<dimgrid,dimblock>>>(A11,A12,T2,K);
        strassen(T2,B22,T1,K);

        add<<<dimgrid,dimblock>>>(C12,T1,C12,K);
        sub<<<dimgrid,dimblock>>>(C11,T1,C11,K);

        gather<<<dimgrid,dimblock>>>(C11,C12,C21,C22,C_G,K);
        

        cudaFree(A11); 
        cudaFree(A12); 
        cudaFree(A21); 
        cudaFree(A22); 
        cudaFree(B11); 
        cudaFree(B12); 
        cudaFree(B21); 
        cudaFree(B22); 
        cudaFree(T1);
        cudaFree(T2);	
    }
    cudaMemcpy(C, C_G, size_b, cudaMemcpyDeviceToHost);
    cudaFree(A_G);
	cudaFree(B_G);
	cudaFree(C_G);

}
int main(){
    int size = SZ*SZ*sizeof(float);

    float *A,*B,*C,*check;

    A = (float*)malloc(size);
    B = (float*)malloc(size);
    C = (float*)malloc(size);
    check = (float*)malloc(size);

    int i,j,k;

    for(i=0;i<SZ;i++){
        for(j=0;j<SZ;j++){
            A[i * SZ + j] = 20*((float)rand() / (float)RAND_MAX);
			B[i * SZ + j] = 20*((float)rand() / (float)RAND_MAX); //Random Floating values between 0...20
			C[i * SZ + j] = 0;
			check[i * SZ + j] = 0;
        }
    }
    /*
    for(i = 0; i < SZ; i++) {
		for(j = 0; j < SZ; j++) {
			for(k = 0; k < SZ; k++) {
				check[i * SZ + j] += A[i * SZ + k] * B[k * SZ + j];
			}
		}
    }
    */

    //TODO : Make clock more accurate.
    clock_t t_f;

    t_f = clock();
         strassen(A, B, C, SZ);
    t_f = clock() - t_f;


    /*
    for(i=0;i<SZ*SZ;++i){
        printf("%f %f \n",C[i],check[i]);
    }
    */
    
   // printf("--------------------------------------------------------------------------------------------------------------------------------------------\n");

    double time_taken = ((double)t_f)/CLOCKS_PER_SEC;

    printf("time(s) = %lf \n", time_taken);

    /*
        N       Leaf     Time(s)
        <1024     -       0.00001
        1024     128      1.25000
        2048     128      2.25000
        8192     128      98.77000
        8192     512      15.99000
        8192     1024      8.74000
        16384    1024     16.43000
        16384    2048     30.93000
        16384    4096     54.79000
    */
}
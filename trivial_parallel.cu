#include<stdio.h>
#include<time.h>
#include<math.h>
#include<stdlib.h>
#include<cuda.h>

__global__ void mult(int *d_a, int *d_b, int *d_c, int N){
    int x = blockIdx.x*blockDim.x + threadIdx.x;
    int y = blockIdx.y*blockDim.y + threadIdx.y;

    if((y*N + x) < N*N){
        int i,val = 0;
        for(i=0;i<N;++i){
                val+=d_a[y*N + i]*d_b[i*N + x];
        }
        d_c[y*N + x] = val;
    }
}
void fill(int *x, int N){
    int i;
    for(i=0;i<N*N;++i){
        x[i] = rand()%500;
    }
}
int main(void){
    int N;

    for(N=1;N<=16384;N*=2){

        srand( time(NULL) );

         int *h_a,*h_b,*h_c,*d_a,*d_b,*d_c;

        int size = N*N*sizeof(int);
        h_a = (int*)malloc(size);
        h_b = (int*)malloc(size);
        h_c = (int*)malloc(size);

        fill(h_a,N);
        fill(h_b,N);


        clock_t t;
        t = clock();

        cudaMalloc((void**)&d_a, size);
        cudaMalloc((void**)&d_b, size);
        cudaMalloc((void**)&d_c, size);

        cudaMemcpy(d_a,h_a,size,cudaMemcpyHostToDevice);
        cudaMemcpy(d_b,h_b,size,cudaMemcpyHostToDevice);


        dim3 dimblock(32,32);
        dim3 dimgrid((N+31)/32,(N+31)/32);

        mult<<<dimgrid,dimblock>>>(d_a,d_b,d_c,N);

        cudaMemcpy(h_c,d_c,size,cudaMemcpyDeviceToHost);

        t = clock()-t;

        double tt = ((double)t)/CLOCKS_PER_SEC; // in second

        cudaFree(d_a);
        cudaFree(d_b);
        cudaFree(d_c);


        free(h_a);
        free(h_b);
        free(h_c);

        printf("%f %d \n",tt,N);
    }

}
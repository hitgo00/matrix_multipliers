#include<stdio.h>
#include<time.h>
#include<math.h>
#include<stdlib.h>
#define CLK CLOCK_MONOTONIC

struct timespec diff(struct timespec start, struct timespec end){
	struct timespec temp;
	if((end.tv_nsec-start.tv_nsec)<0){
		temp.tv_sec = end.tv_sec-start.tv_sec-1;
		temp.tv_nsec = 1000000000+end.tv_nsec-start.tv_nsec;
	}
	else{
		temp.tv_sec = end.tv_sec-start.tv_sec;
		temp.tv_nsec = end.tv_nsec-start.tv_nsec;
	}
	return temp;
}
float **allocate_matrix(int Rows, int Cols){
    float **matrix = (float **)malloc(Rows * sizeof(float *)); 
    int row;
    for (row = 0; row < Rows; row++) {
        matrix[row] = (float*)malloc(Cols * sizeof(float));
    }

    return matrix;
}
void free_matrix(float **matrix, float Rows){
    int row;
    for (row = 0; row < Rows; row++) {
         free(matrix[row]);
    }
    free(matrix);
}
void fill_matrix(float **matrix, float Rows){
    srand(time(0));
    int i,j;
    for(i = 0; i<Rows; ++i){
        for(j = 0; j<Rows ;++j){
            matrix[i][j] = rand()%50;
        }
    }
}
void show(float **matrix, float Rows){
    int i,j;
    for(i = 0; i < Rows; ++i){
        for(j = 0; j < Rows; ++j){
            printf("%d ", matrix[i][j]);
        }
        printf("\n");
    }
    printf("\n");
}
float **add(float **A, float **B, int N){
    float **ret = allocate_matrix(N,N);
    int i,j;
     for(i = 0; i < N; ++i){
         for(j = 0; j < N; ++j){
             ret[i][j] = A[i][j] + B[i][j];
         }
     }
     return ret;
}
float **subtract(float **A, float **B, int N){
    float **ret = allocate_matrix(N,N);
    int i,j;
     for(i = 0; i < N; ++i){
         for(j = 0; j < N; ++j){
             ret[i][j] = A[i][j] - B[i][j];
         }
     }
     return ret;
}
float **trivial(float **A, float **B, int N){
    float **ret = allocate_matrix(N,N);
    int i,j,k;
    for(i = 0; i < N; ++i){
        for(j = 0; j < N; ++j){
            for(k = 0; k < N; ++k){
                ret[i][j]+=A[i][k]*B[k][j];
            }
        }
    }
    return ret;
}
float **strassen(float **A, float **B, int N){
    //Base case
    if(N<=100){
        int **ret = allocate_matrix(N,N);
        ret = trivial(A,B,N);
        return ret;
    }

    float **ret = allocate_matrix(N,N);
    int K = N/2;
    int i,j;

    float **A11 = allocate_matrix(K,K);
    float **A12 = allocate_matrix(K,K);
    float **A21 = allocate_matrix(K,K);
    float **A22 = allocate_matrix(K,K);

    float **B11 = allocate_matrix(K,K);
    float **B12 = allocate_matrix(K,K);
    float **B21 = allocate_matrix(K,K);
    float **B22 = allocate_matrix(K,K);

    for(i = 0; i < K; ++i){
        for(j = 0; j < K; ++j){
            A11[i][j] = A[i][j];
            A12[i][j] = A[i][K+j];
            A21[i][j] = A[K+i][j];
            A22[i][j] = A[K+i][K+j];

            B11[i][j] = B[i][j];
            B12[i][j] = B[i][K+j];
            B21[i][j] = B[K+i][j];
            B22[i][j] = B[K+i][K+j];
        }
    }
    float **P1 = strassen(A11, subtract(B12, B22, K), K);
    float **P2 = strassen(add(A11, A12, K), B22, K);
    float **P3 = strassen(add(A21, A22, K), B11, K);
    float **P4 = strassen(A22, subtract(B21, B11, K), K);
    float **P5 = strassen(add(A11, A22, K), add(B11, B22, K), K);
    float **P6 = strassen(subtract(A12, A22, K), add(B21, B22, K), K);
    float **P7 = strassen(subtract(A11, A21, K), add(B11, B12, K), K);

    float **ret11 = subtract(add(add(P5, P4, K), P6, K), P2, K);
    float **ret12 = add(P1, P2, K);
    float **ret21 = add(P3, P4, K);
    float **ret22 = subtract(subtract(add(P5, P1, K), P3, K), P7, K);

    for(i=0; i<K; i++){
        for(j=0; j<K; j++) {
            ret[i][j] = ret11[i][j];
            ret[i][j+K] = ret12[i][j];
            ret[K+i][j] = ret21[i][j];
            ret[K+i][K+j] = ret22[i][j];
        }
    }
    for(i = 0; i < K; ++i){
        free(A11[i]);
        free(A12[i]);
        free(A21[i]);
        free(A22[i]);
        free(B11[i]);
        free(B12[i]);
        free(B21[i]);
        free(B22[i]);
        free(P1[i]);
        free(P2[i]);
        free(P3[i]);
        free(P4[i]);
        free(P5[i]);
        free(P6[i]);
        free(P7[i]);
        free(ret11[i]);
        free(ret12[i]);
        free(ret21[i]);
        free(ret22[i]);
    }
    free(A11);
    free(A12);
    free(A21);
    free(A22);
    free(B11);
    free(B12);
    free(B21);
    free(B22);
    free(P1);
    free(P2);
    free(P3);
    free(P4);
    free(P5);
    free(P6);
    free(P7);
    free(ret11);
    free(ret12);
    free(ret21);
    free(ret22);

    return ret;
}

int main(void){
    int N;
    for(N = 1; N <= 2048; N*=2){

        float **A = allocate_matrix(N,N);
        float **B = allocate_matrix(N,N);

        fill_matrix(A,N);
        fill_matrix(B,N);

        struct timespec t_strassen_start, t_trivial_start, t_strassen_end, t_trivial_end;

        clock_gettime(CLK, &t_strassen_start);
        float **C_strassen = strassen(A,B,N);
        clock_gettime(CLK, &t_strassen_end);
		
		t_strassen_end= diff(t_strassen_start,t_strassen_end);
        double time_taken_strassen = t_strassen_end.tv_sec * 1e9;
		time_taken_strassen= (time_taken_strassen+ (t_strassen_end.tv_nsec)) *1e-9;
        
        clock_gettime(CLK, &t_trivial_start);
        float **C_trivial = trivial(A,B,N);
        clock_gettime(CLK, &t_trivial_end);

        /*
        show(A,N);
        show(B,N);
        show(C_strassen,N);
        show(C_trivial,N);
        */
		t_trivial_end=diff= (t_trivial_start,t_trivial_end);
        double time_taken_trivial = t_trivial_end.tv_sec * 1e9;  // in second
		time_taken_trivial = (time_taken_trivial + (t_trivial_end.tv_nsec )) * 1e-9; 

        printf("%0.12f %0.12f %d \n",time_taken_trivial,time_taken_strassen,N);
    }

    

}
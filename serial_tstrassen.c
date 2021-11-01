#include<stdio.h>
#include<time.h>
#include<math.h>
#include<stdlib.h>


int **allocate_matrix(int Rows, int Cols){
    int **matrix = (int **)malloc(Rows * sizeof(int *)); 
    int row;
    for (row = 0; row < Rows; row++) {
        matrix[row] = (int *)malloc(Cols * sizeof(int));
    }

    return matrix;
}
void free_matrix(int **matrix, int Rows){
    int row;
    for (row = 0; row < Rows; row++) {
         free(matrix[row]);
    }
    free(matrix);
}
void fill_matrix(int **matrix, int Rows){
    srand(time(0));
    int i,j;
    for(i = 0; i<Rows; ++i){
        for(j = 0; j<Rows ;++j){
            matrix[i][j] = rand()%50;
        }
    }
}
void show(int **matrix, int Rows){
    int i,j;
    for(i = 0; i < Rows; ++i){
        for(j = 0; j < Rows; ++j){
            printf("%d ", matrix[i][j]);
        }
        printf("\n");
    }
    printf("\n");
}
int **add(int **A, int **B, int N){
    int **ret = allocate_matrix(N,N);
    int i,j;
     for(i = 0; i < N; ++i){
         for(j = 0; j < N; ++j){
             ret[i][j] = A[i][j] + B[i][j];
         }
     }
     return ret;
}
int **subtract(int **A, int **B, int N){
    int **ret = allocate_matrix(N,N);
    int i,j;
     for(i = 0; i < N; ++i){
         for(j = 0; j < N; ++j){
             ret[i][j] = A[i][j] - B[i][j];
         }
     }
     return ret;
}
int **trivial(int **A, int **B, int N){
    int **ret = allocate_matrix(N,N);
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
int **strassen(int **A, int **B, int N){
    //Base case
    if(N<=100){
        int **ret = allocate_matrix(N,N);
        ret = trivial(A,B,N);
        return ret;
    }

    int **ret = allocate_matrix(N,N);
    int K = N/2;
    int i,j;

    int **A11 = allocate_matrix(K,K);
    int **A12 = allocate_matrix(K,K);
    int **A21 = allocate_matrix(K,K);
    int **A22 = allocate_matrix(K,K);

    int **B11 = allocate_matrix(K,K);
    int **B12 = allocate_matrix(K,K);
    int **B21 = allocate_matrix(K,K);
    int **B22 = allocate_matrix(K,K);

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
    int **P1 = strassen(A11, subtract(B12, B22, K), K);
    int **P2 = strassen(add(A11, A12, K), B22, K);
    int **P3 = strassen(add(A21, A22, K), B11, K);
    int **P4 = strassen(A22, subtract(B21, B11, K), K);
    int **P5 = strassen(add(A11, A22, K), add(B11, B22, K), K);
    int **P6 = strassen(subtract(A12, A22, K), add(B21, B22, K), K);
    int **P7 = strassen(subtract(A11, A21, K), add(B11, B12, K), K);

    int **ret11 = subtract(add(add(P5, P4, K), P6, K), P2, K);
    int **ret12 = add(P1, P2, K);
    int **ret21 = add(P3, P4, K);
    int **ret22 = subtract(subtract(add(P5, P1, K), P3, K), P7, K);

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

        int **A = allocate_matrix(N,N);
        int **B = allocate_matrix(N,N);

        fill_matrix(A,N);
        fill_matrix(B,N);

        clock_t t_strassen, t_trivial;

        t_strassen = clock();
        int **C_strassen = strassen(A,B,N);
        t_strassen = clock() - t_strassen;

        double time_taken_strassen = ((double)t_strassen)/CLOCKS_PER_SEC; // in second

        
        t_trivial = clock();
        int **C_trivial = trivial(A,B,N);
        t_trivial = clock() - t_trivial;

        /*
        show(A,N);
        show(B,N);
        show(C_strassen,N);
        show(C_trivial,N);
        */

        double time_taken_trivial = ((double)t_trivial)/CLOCKS_PER_SEC;

        printf("%f %f %d \n",time_taken_trivial,time_taken_strassen,N);
    }

    

}
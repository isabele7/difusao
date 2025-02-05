#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>

#define N 2000       
#define T 500        
#define D 0.1        
#define DELTA_T 0.01
#define DELTA_X 1.0

void free_2d_array(double **arr, int rows) {
    for (int i = 0; i < rows; i++) {
        free(arr[i]);
    }
    free(arr);
}

double** create_2d_array(int rows, int cols) {
    double **arr = (double **)malloc(rows * sizeof(double *));
    if (arr == NULL) {
        fprintf(stderr, "Memory allocation failed for array\n");
        exit(1);
    }

    for (int i = 0; i < rows; i++) {
        arr[i] = (double *)calloc(cols, sizeof(double));
        if (arr[i] == NULL) {
            fprintf(stderr, "Memory allocation failed for row %d\n", i);
            free_2d_array(arr, i);
            exit(1);
        }
    }

    return arr;
}

void diff_eq_openmp(double **C, double **C_new) {
    double difmedio;
    double max_diff = 0.0;

    for (int t = 0; t < T; t++) {
        difmedio = 0.0;

        #pragma omp parallel
        {
            #pragma omp for reduction(+:difmedio) collapse(2)
            for (int i = 1; i < N - 1; i++) {
                for (int j = 1; j < N - 1; j++) {
                    C_new[i][j] = C[i][j] + D * DELTA_T * (
                        (C[i+1][j] + C[i-1][j] + C[i][j+1] + C[i][j-1] - 4 * C[i][j]) / (DELTA_X * DELTA_X)
                    );

                    difmedio += fabs(C_new[i][j] - C[i][j]);
                }
            }
            #pragma omp for collapse(2)
            for (int i = 1; i < N - 1; i++) {
                for (int j = 1; j < N - 1; j++) {
                    C[i][j] = C_new[i][j];
                }
            }
        }
        max_diff = difmedio/((N-2)*(N-2));
        if ((t % 100) == 0)
            printf("OpenMP - Iteração %d - Diferença: %g\n", t, max_diff);
    }
}

int main() {
    int num_threads;
    double start_time, end_time, seq_time, parallel_time;
    double speedup, eficiencia;

    #pragma omp parallel
    {
        #pragma omp single
        num_threads = omp_get_num_threads();
    }
    double **C = create_2d_array(N, N);
    double **C_new = create_2d_array(N, N);

    C[N/2][N/2] = 1.0;

    start_time = omp_get_wtime();

    seq_time = 1.0; 

    start_time = omp_get_wtime();
    diff_eq_openmp(C, C_new);
    end_time = omp_get_wtime();
    parallel_time = end_time - start_time;

    speedup = seq_time / parallel_time;
    eficiencia = speedup / num_threads;

    printf("Threads: %d\n", num_threads);
    printf("Tempo de execução paralelo: %f segundos\n", parallel_time);
    printf("Speedup: %f\n", speedup);
    printf("Eficiência: %f\n", eficiencia);

    FILE *file = fopen("openmp_diffusion.txt", "w");
    if (file == NULL) {
        fprintf(stderr, "Error opening file for writing\n");
        return 1;
    }

    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            fprintf(file, "%f ", C[i][j]);
        }
        fprintf(file, "\n");
    }
    fclose(file);

    free_2d_array(C, N);
    free_2d_array(C_new, N);

    return 0;
}
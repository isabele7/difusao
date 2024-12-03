#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>

#define N 2000        // Grid size
#define T 500         // Time iterations
#define D 0.1         // Diffusion coefficient
#define DELTA_T 0.01
#define DELTA_X 1.0

// Function to free 2D array
void free_2d_array(double **arr, int rows) {
    for (int i = 0; i < rows; i++) {
        free(arr[i]);
    }
    free(arr);
}

// Function to allocate and initialize 2D array
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

// Versão Sequencial da Equação de Difusão
void diff_eq_sequential(double **C, double **C_new) {
    double difmedio;

    for (int t = 0; t < T; t++) {
        difmedio = 0.0;

        for (int i = 1; i < N - 1; i++) {
            for (int j = 1; j < N - 1; j++) {
                C_new[i][j] = C[i][j] + D * DELTA_T * (
                    (C[i+1][j] + C[i-1][j] + C[i][j+1] + C[i][j-1] - 4 * C[i][j]) / (DELTA_X * DELTA_X)
                );

                difmedio += fabs(C_new[i][j] - C[i][j]);
            }
        }

        // Atualizar matriz para a próxima iteração
        for (int i = 1; i < N - 1; i++) {
            for (int j = 1; j < N - 1; j++) {
                C[i][j] = C_new[i][j];
            }
        }

        if ((t % 100) == 0)
            printf("Sequencial - Iteracao %d - Diferenca: %g\n", t, difmedio/((N-2)*(N-2)));
    }
}

// Versão OpenMP da Equação de Difusão
void diff_eq_openmp(double **C, double **C_new) {
    double difmedio;

    for (int t = 0; t < T; t++) {
        difmedio = 0.0;

        #pragma omp parallel
        {
            // Computar novas concentrações e rastrear diferença
            #pragma omp for reduction(+:difmedio) collapse(2)
            for (int i = 1; i < N - 1; i++) {
                for (int j = 1; j < N - 1; j++) {
                    C_new[i][j] = C[i][j] + D * DELTA_T * (
                        (C[i+1][j] + C[i-1][j] + C[i][j+1] + C[i][j-1] - 4 * C[i][j]) / (DELTA_X * DELTA_X)
                    );

                    difmedio += fabs(C_new[i][j] - C[i][j]);
                }
            }

            // Atualizar array original na mesma região paralela
            #pragma omp for collapse(2)
            for (int i = 1; i < N - 1; i++) {
                for (int j = 1; j < N - 1; j++) {
                    C[i][j] = C_new[i][j];
                }
            }
        }

        if ((t % 100) == 0)
            printf("OpenMP - Iteracao %d - Diferenca: %g\n", t, difmedio/((N-2)*(N-2)));
    }
}

int main() {
    double start_time, end_time;
    double seq_time, parallel_time;
    double speedup, efficiency;

    // Alocar memória
    double **C_seq = create_2d_array(N, N);
    double **C_new_seq = create_2d_array(N, N);

    double **C_omp = create_2d_array(N, N);
    double **C_new_omp = create_2d_array(N, N);

    // Inicializar matriz
    C_seq[N/2][N/2] = 1.0;
    C_omp[N/2][N/2] = 1.0;

    // Tempo sequencial
    start_time = omp_get_wtime();
    diff_eq_sequential(C_seq, C_new_seq);
    end_time = omp_get_wtime();
    seq_time = end_time - start_time;

    printf("Tempo Sequencial: %f segundos\n", seq_time);

    // Testar com diferentes números de threads
    int thread_counts[] = {2, 4, 8};
    for (int t = 0; t < 3; t++) {
        int threads = thread_counts[t];
        omp_set_num_threads(threads);

        // Reinicializar matriz
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++) {
                C_omp[i][j] = 0.0;
                C_new_omp[i][j] = 0.0;
            }
        }
        C_omp[N/2][N/2] = 1.0;

        // Tempo paralelo
        start_time = omp_get_wtime();
        diff_eq_openmp(C_omp, C_new_omp);
        end_time = omp_get_wtime();
        parallel_time = end_time - start_time;

        // Métricas
        speedup = seq_time / parallel_time;
        efficiency = speedup / threads;

        printf("\n--- Resultados com %d threads ---\n", threads);
        printf("Tempo Paralelo: %f segundos\n", parallel_time);
        printf("Speedup: %f\n", speedup);
        printf("Eficiencia: %f\n", efficiency);
    }

    // Liberar memória
    free_2d_array(C_seq, N);
    free_2d_array(C_new_seq, N);
    free_2d_array(C_omp, N);
    free_2d_array(C_new_omp, N);

    return 0;
}

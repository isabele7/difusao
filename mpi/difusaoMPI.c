#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <mpi.h>

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

        for (int i = 1; i < N - 1; i++) {
            for (int j = 1; j < N - 1; j++) {
                C[i][j] = C_new[i][j];
            }
        }

        if ((t % 100) == 0)
            printf("Sequencial - Iteracao %d - Diferenca: %g\n", t, difmedio/((N-2)*(N-2)));
    }
}

void diff_eq_mpi(double **C, double **C_new, int rank, int size) {
    double difmedio, global_difmedio;
    int rows_per_proc = (N - 2) / size;  
    int start_row = 1 + rank * rows_per_proc;
    int end_row = start_row + rows_per_proc;

    if (rank == size - 1) {
        end_row = N - 1;
    }

    double *send_up = (double *)malloc(N * sizeof(double));
    double *send_down = (double *)malloc(N * sizeof(double));
    double *recv_up = (double *)malloc(N * sizeof(double));
    double *recv_down = (double *)malloc(N * sizeof(double));

    for (int t = 0; t < T; t++) {
        difmedio = 0.0;
        if (rank > 0) {
            MPI_Sendrecv(C[start_row], N, MPI_DOUBLE, rank-1, 0,
                        recv_up, N, MPI_DOUBLE, rank-1, 0,
                        MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
        if (rank < size - 1) {
            MPI_Sendrecv(C[end_row-1], N, MPI_DOUBLE, rank+1, 0,
                        recv_down, N, MPI_DOUBLE, rank+1, 0,
                        MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
        for (int i = start_row; i < end_row; i++) {
            for (int j = 1; j < N - 1; j++) {
                double up = (i == start_row && rank > 0) ? recv_up[j] : C[i-1][j];
                double down = (i == end_row-1 && rank < size-1) ? recv_down[j] : C[i+1][j];

                C_new[i][j] = C[i][j] + D * DELTA_T * (
                    (down + up + C[i][j+1] + C[i][j-1] - 4 * C[i][j]) / (DELTA_X * DELTA_X)
                );

                difmedio += fabs(C_new[i][j] - C[i][j]);
            }
        }
        for (int i = start_row; i < end_row; i++) {
            for (int j = 1; j < N - 1; j++) {
                C[i][j] = C_new[i][j];
            }
        }

        MPI_Allreduce(&difmedio, &global_difmedio, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

        if ((t % 100) == 0 && rank == 0)
            printf("MPI - Iteracao %d - Diferenca: %g\n", t, global_difmedio/((N-2)*(N-2)));
    }

    free(send_up);
    free(send_down);
    free(recv_up);
    free(recv_down);
}

int main(int argc, char *argv[]) {
    int rank, size;
    double start_time, end_time;
    double seq_time, parallel_time;
    double speedup, efficiency;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    double **C = create_2d_array(N, N);
    double **C_new = create_2d_array(N, N);

    if (rank == 0) {
        C[N/2][N/2] = 1.0;

        double **C_seq = create_2d_array(N, N);
        double **C_new_seq = create_2d_array(N, N);
        C_seq[N/2][N/2] = 1.0;

        printf("Iniciando versão Sequencial\n");
        start_time = MPI_Wtime();
        diff_eq_sequential(C_seq, C_new_seq);
        end_time = MPI_Wtime();
        seq_time = end_time - start_time;
        printf("Tempo Sequencial: %f segundos\n", seq_time);

        free_2d_array(C_seq, N);
        free_2d_array(C_new_seq, N);
    }

    MPI_Barrier(MPI_COMM_WORLD);

    if (rank == 0)
        printf("Iniciando versão MPI com %d processos\n", size);

    start_time = MPI_Wtime();
    diff_eq_mpi(C, C_new, rank, size);
    end_time = MPI_Wtime();
    parallel_time = end_time - start_time;

    if (rank == 0) {
        speedup = seq_time / parallel_time;
        efficiency = speedup / size;

        printf("\n--- Resultados com %d processos ---\n", size);
        printf("Tempo Paralelo: %f segundos\n", parallel_time);
        printf("Speedup: %f\n", speedup);
        printf("Eficiência: %f\n", efficiency);
    }

    free_2d_array(C, N);
    free_2d_array(C_new, N);

    MPI_Finalize();
    return 0;
}
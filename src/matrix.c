#include "matrix.h"
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

// Include SSE intrinsics
#if defined(_MSC_VER)
#include <intrin.h>
#elif defined(__GNUC__) && (defined(__x86_64__) || defined(__i386__))
#include <immintrin.h>
#include <x86intrin.h>
#endif

/* Below are some intel intrinsics that might be useful
 * void _mm256_storeu_pd (double * mem_addr, __m256d a)
 * __m256d _mm256_set1_pd (double a)
 * __m256d _mm256_set_pd (double e3, double e2, double e1, double e0)
 * __m256d _mm256_loadu_pd (double const * mem_addr)
 * __m256d _mm256_add_pd (__m256d a, __m256d b)
 * __m256d _mm256_sub_pd (__m256d a, __m256d b)
 * __m256d _mm256_fmadd_pd (__m256d a, __m256d b, __m256d c)
 * __m256d _mm256_mul_pd (__m256d a, __m256d b)
 * __m256d _mm256_cmp_pd (__m256d a, __m256d b, const int imm8)
 * __m256d _mm256_and_pd (__m256d a, __m256d b)
 * __m256d _mm256_max_pd (__m256d a, __m256d b)
*/

/* Generates a random double between low and high */
double rand_double(double low, double high) {
    double range = (high - low);
    double div = RAND_MAX / range;
    return low + (rand() / div);
}

/* Generates a random matrix */
void rand_matrix(matrix *result, unsigned int seed, double low, double high) {
    srand(seed);
    for (int i = 0; i < result->rows; i++) {
        for (int j = 0; j < result->cols; j++) {
            set(result, i, j, rand_double(low, high));
        }
    }
}

/*
 * Allocates space for a matrix struct pointed to by the double pointer mat with
 * `rows` rows and `cols` columns. You should also allocate memory for the data array
 * and initialize all entries to be zeros. `parent` should be set to NULL to indicate that
 * this matrix is not a slice. You should also set `ref_cnt` to 1.
 * You should return -1 if either `rows` or `cols` or both have invalid values. Return -2 if any
 * call to allocate memory in this function fails. Remember to set the error messages in numc.c.
 * Return 0 upon success.
 */
int allocate_matrix(matrix **mat, int rows, int cols) {
    /* TODO: YOUR CODE HERE */
    if (rows <= 0 || cols <= 0) {
        return -1;
    }
    *mat = malloc(sizeof(matrix));
    if (*mat == NULL) {
        return -2;
    }
    (*mat)->rows = rows;
    (*mat)->cols = cols;
    (*mat)->parent = NULL;
    (*mat)->ref_cnt = 1;
    (*mat)->data = malloc(rows*cols*sizeof(double));
    if ((*mat)->data == NULL) {
        return -2;
    }
    // initialize all entries to be zeros
    #pragma omp parallel for
    for (int i = 0; i < rows*cols; i++)
        (*mat)->data[i] = 0;
    return 0;
}

/*
 * Allocates space for a matrix struct pointed to by `mat` with `rows` rows and `cols` columns.
 * Its data should point to the `offset`th entry of `from`'s data (you do not need to allocate memory)
 * for the data field. `parent` should be set to `from` to indicate this matrix is a slice of `from`.
 * You should return -1 if either `rows` or `cols` or both have invalid values. Return -2 if any
 * call to allocate memory in this function fails.
 * Remember to set the error messages in numc.c.
 * Return 0 upon success.
 */
int allocate_matrix_ref(matrix **mat, matrix *from, int offset, int rows, int cols) {
    /* TODO: YOUR CODE HERE */
    if (rows <= 0 || cols <= 0) {
        return -1;
    }
    *mat = malloc(sizeof(matrix));
    if (*mat == NULL) {
        return -2;
    }
    (*mat)->rows = rows;
    (*mat)->cols = cols;
    (*mat)->parent = from;
    (*mat)->ref_cnt = 1;
    (*mat)->data = from->data + offset;
    from->ref_cnt += 1;  // ???
    return 0;
}

/*
 * You need to make sure that you only free `mat->data` if `mat` is not a slice and has no existing slices,
 * or that you free `mat->parent->data` if `mat` is the last existing slice of its parent matrix and its parent matrix has no other references
 * (including itself). You cannot assume that mat is not NULL.
 */
void deallocate_matrix(matrix *mat) {
    /* TODO: YOUR CODE HERE */
    // if `mat` is not a slice and has no existing slices
    if (mat == NULL) {
        return;
    }
    if (mat->parent == NULL && mat->ref_cnt == 1) {
        free(mat->data);
        free(mat);
        return;
    }
    // deallocate a matrix, but has slices
    if (mat->parent == NULL && mat->ref_cnt > 1) {
        mat->ref_cnt -= 1;
        return;
    }
    // mat is a slice, and its parent has more ref, just free the slice itself  
    if (mat->parent->ref_cnt > 1) {
        //free(mat->data);
        mat->parent->ref_cnt -= 1;
        free(mat);
        return;
    }
    // the last existing slice
    if (mat->parent->ref_cnt == 1) {
        free(mat->parent->data);
        free(mat->parent);
        free(mat);
        return;
    }
}

/*
 * Returns the double value of the matrix at the given row and column.
 * You may assume `row` and `col` are valid.
 */
double get(matrix *mat, int row, int col) {
    int col_num = mat->cols;
    return mat->data[row*col_num + col];
    /* TODO: YOUR CODE HERE */
}

/*
 * Sets the value at the given row and column to val. You may assume `row` and
 * `col` are valid
 */
void set(matrix *mat, int row, int col, double val) {
    /* TODO: YOUR CODE HERE */
    int col_num = mat->cols;
    mat->data[row*col_num + col] = val;
}

/*
 * Sets all entries in mat to val
 */
void fill_matrix(matrix *mat, double val) {
    int elements_num = mat->cols * mat->rows;
    #pragma omp parallel for
    for (int i = 0; i < elements_num / 4 * 4; i += 4) {
        __m256d set_val = _mm256_set1_pd(val);
        _mm256_storeu_pd(mat->data + i, set_val);
    }
    #pragma omp parallel for
    for (int i = elements_num / 4 * 4; i < elements_num; i++) {
        mat->data[i] = val;
    }
    /* TODO: YOUR CODE HERE */
}

/*
 * Store the result of adding mat1 and mat2 to `result`.
 * Return 0 upon success and a nonzero value upon failure.
 */
int add_matrix(matrix *result, matrix *mat1, matrix *mat2) {
    if (mat1->rows != mat2->rows || mat1->cols != mat2->cols) {
        return 1;
    }
    __m256d tmp;
    int elements_num = mat1->rows * mat1->cols;
    #pragma omp parallel for
    for (int i = 0; i < elements_num / 4 * 4; i += 4) {
        __m256d load_mat1 = _mm256_loadu_pd(mat1->data + i);
        __m256d load_mat2 = _mm256_loadu_pd(mat2->data + i);
        tmp = _mm256_add_pd(load_mat1, load_mat2);
        _mm256_storeu_pd(result->data + i, tmp);
    }
    #pragma omp parallel for
    for (int i = elements_num / 4 * 4; i < elements_num; i++) {
        result->data[i] = mat1->data[i] + mat2->data[i];
    }
    return 0;
    /* TODO: YOUR CODE HERE */
}

/*
 * (OPTIONAL)
 * Store the result of subtracting mat2 from mat1 to `result`.
 * Return 0 upon success and a nonzero value upon failure.
 */
int sub_matrix(matrix *result, matrix *mat1, matrix *mat2) {
    /* TODO: YOUR CODE HERE */
    if (mat1->rows != mat2->rows || mat1->cols != mat2->cols) {
        return 1;
    }
    __m256d tmp;
    int elements_num = mat1->rows * mat1->cols;
    for (int i = 0; i < elements_num / 4 * 4; i += 4) {
        __m256d load_mat1 = _mm256_loadu_pd(mat1->data + i);
        __m256d load_mat2 = _mm256_loadu_pd(mat2->data + i);
        tmp = _mm256_sub_pd(load_mat1, load_mat2);
        _mm256_storeu_pd(result->data + i, tmp);
    }
    for (int i = elements_num / 4 * 4; i < elements_num; i++) {
        result->data[i] = mat1->data[i] - mat2->data[i];
    }
    return 0;
}

/*
 * Store the result of multiplying mat1 and mat2 to `result`.
 * Return 0 upon success and a nonzero value upon failure.
 * Remember that matrix multiplication is not the same as multiplying individual elements.
 */
int mul_matrix(matrix *result, matrix *mat1, matrix *mat2) {
    /* TODO: YOUR CODE HERE */
    int rows1 = mat1->rows;
    int cols1 = mat1->cols;
    int rows2 = mat2->rows;
    int cols2 = mat2->cols;
    if (cols1 != rows2 || rows1 <= 0 || rows2 <= 0 || cols1 <= 0 || cols2 <= 0) {
        return 1;
    }

    double *mat1_data = mat1->data;
    double *mat2_data = mat2->data;


   // if col or row is less then 4, just multiply
    if (cols1 < 4) {
        double *temp_result = malloc(rows1*cols2*sizeof(double));
        if (temp_result == NULL) {
            return -2;
        }
        #pragma omp parallel for
        for (int i = 0; i < rows1; i++) {
            for (int j = 0; j < cols2; j++) {
                double sum_ij = 0;
                for (int k = 0; k < cols1; k++) {
                    // result[i][j]         data1[i][k]           data2[k][j]
                    sum_ij += mat1_data[i*cols1+k] * mat2_data[k*cols2+j];
                    //printf("data1[i][k]:%f * data2[k][j]:%f, sum=%d\n", get(mat1, i, k), get(mat2, k, j), sum_ij);
                }
                temp_result[i*cols1+j] = sum_ij;
            }
        }

        
        double *prev_data = result->data;
        result->data = temp_result;
        free(prev_data);

        return 0;
        
    } else if (cols1 < 16 && cols1 >= 4) {
        int transposed_row = cols2;
        int transposed_col = rows2;
        double *transposed_data = malloc(cols2*rows2*sizeof(double));
        if (transposed_data == NULL) {
            return -2;
        }
        //allocate_matrix(&transposed, transposed_row, transposed_col);

        #pragma omp parallel for
        for (int i = 0; i < rows2; i++) {
            for (int j = 0; j < cols2; j++) {
                transposed_data[j*rows2 + i] = mat2_data[i*cols2 + j];
            }
        }

        double *temp_data = malloc((rows1*cols2)*sizeof(double));
        if (temp_data == NULL) {
            return -2;
        }

        #pragma omp parallel for
        for (int i = 0; i < rows1; i++) {
            for (int j = 0; j < transposed_row; j++) {
                __m256d sum = _mm256_set1_pd(0);
                double result_4ij[4] = {0, 0, 0, 0};
                for (int k = 0; k < cols1 / 4 * 4; k += 4) {
                    sum = _mm256_fmadd_pd(_mm256_loadu_pd(mat1_data + i*cols1+k), _mm256_loadu_pd(transposed_data + j*transposed_col+k), sum);
                }
                _mm256_storeu_pd(result_4ij, sum);
                for (int k = cols1 / 4 * 4; k < cols1; k++) {
                    result_4ij[0] += mat1_data[i*cols1+k] * transposed_data[j*transposed_col+k];
                }
                double result_ij = result_4ij[0] + result_4ij[1] + result_4ij[2] + result_4ij[3];
                temp_data[i*cols2+j] = result_ij;
                //result->data[i*cols2+j] = result_ij;
            }  
        }

        double *prev_data = result->data;
        result->data = temp_data;
        free(prev_data);

        free(transposed_data);
        return 0;
    } else {
        int transposed_row = cols2;
        int transposed_col = rows2;
        double *transposed_data = malloc(cols2*rows2*sizeof(double));
        if (transposed_data == NULL) {
            return -2;
        }
        //allocate_matrix(&transposed, transposed_row, transposed_col);

        #pragma omp parallel for
        for (int i = 0; i < rows2; i++) {
            for (int j = 0; j < cols2; j++) {
                transposed_data[j*rows2 + i] = mat2_data[i*cols2 + j];
            }
        }

        double *temp_data = malloc((rows1*cols2)*sizeof(double));
        if (temp_data == NULL) {
            return -2;
        }

        #pragma omp parallel for
        for (int i = 0; i < rows1; i++) {
            for (int j = 0; j < transposed_row; j++) {
                __m256d sum = _mm256_set1_pd(0);
                double result_4ij[4] = {0, 0, 0, 0};
                for (int k = 0; k < cols1 / 16 * 16; k += 16) {
                    sum = _mm256_fmadd_pd(_mm256_loadu_pd(mat1_data + i*cols1+k), _mm256_loadu_pd(transposed_data + j*transposed_col+k), sum);
                    
                    // load_mat1 = _mm256_loadu_pd(mat1->data + i*cols1+k + 4);
                    // load_transposed = _mm256_loadu_pd(transposed_data + j*transposed_col+k + 4);
                    sum = _mm256_fmadd_pd(_mm256_loadu_pd(mat1_data + i*cols1+k + 4), _mm256_loadu_pd(transposed_data + j*transposed_col+k + 4), sum);

                    // load_mat1 = _mm256_loadu_pd(mat1->data + i*cols1+k + 8);
                    // load_transposed = _mm256_loadu_pd(transposed_data + j*transposed_col+k + 8);
                    sum = _mm256_fmadd_pd(_mm256_loadu_pd(mat1_data + i*cols1+k + 8), _mm256_loadu_pd(transposed_data + j*transposed_col+k + 8), sum);

                    // load_mat1 = _mm256_loadu_pd(mat1->data + i*cols1+k + 12);
                    // load_transposed = _mm256_loadu_pd(transposed_data + j*transposed_col+k + 12);
                    sum = _mm256_fmadd_pd(_mm256_loadu_pd(mat1_data + i*cols1+k + 12), _mm256_loadu_pd(transposed_data + j*transposed_col+k + 12), sum);

                    
                }
                _mm256_storeu_pd(result_4ij, sum);
                for (int k = cols1 / 16 * 16; k < cols1; k++) {
                    result_4ij[0] += mat1_data[i*cols1+k] * transposed_data[j*transposed_col+k];
                }
                double result_ij = result_4ij[0] + result_4ij[1] + result_4ij[2] + result_4ij[3];
                temp_data[i*cols2+j] = result_ij;
                //result->data[i*cols2+j] = result_ij;
            }  
        }

        double *prev_data = result->data;
        result->data = temp_data;
        free(prev_data);

        free(transposed_data);
        return 0;
    }

}

/*
 * Store the result of raising mat to the (pow)th power to `result`.
 * Return 0 upon success and a nonzero value upon failure.
 * Remember that pow is defined with matrix multiplication, not element-wise multiplication.
 */
int pow_matrix(matrix *result, matrix *mat, int pow) {
    int row = mat->rows;
    int col = mat->cols;
    if (row != col) {
        return 2;
    }
    if (pow < 0) {
        return 1;
    }
    
    int i = 0;
    int j = 0;
    #pragma omp parallel for private(i, j)
    for (i = 0; i < row; i++) {
        for (j = 0; j < col; j++) {
            if (i == j) {
                result->data[i*col+j] = 1;
            } else {
                result->data[i*col+j] = 0; 
            }
        }
    }
    
    if (pow == 0) {
        return 0;
    }


    int bits[20] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    int num = 0;
    for (num = 0; pow != 0; num++) {
        bits[num] = pow % 2;
        pow /= 2;
    }

/*
    printf("%d\n", bits[0]);
    printf("%d\n", bits[1]);
    printf("%d\n", num);


    printf("-----------indentity-------------\n");
    printf("%f\n", get(result, 0, 0));
    printf("%f\n", get(result, 0, 1));
    printf("%f\n", get(result, 1, 0));
    printf("%f\n", get(result, 1, 1));
    
*/
    
    for (num--; num >= 0; num--) {
        if (num == 0) {
            if (bits[num] == 1) {
                mul_matrix(result, result, mat);
            }
        } else {
            if (bits[num] == 0) {
                mul_matrix(result, result, result);
                /*
                printf("----------lsb, num=%d---------\n", num);
                printf("%f\n", get(result, 0, 0));
                printf("%f\n", get(result, 0, 1));
                printf("%f\n", get(result, 1, 0));
                printf("%f\n", get(result, 1, 1));
                */

            } else {
                mul_matrix(result, result, mat);

                /*
                printf("----------*mat, , num=%d---------\n", num);
                printf("%f\n", get(result, 0, 0));
                printf("%f\n", get(result, 0, 1));
                printf("%f\n", get(result, 1, 0));
                printf("%f\n", get(result, 1, 1));
                */
                
                mul_matrix(result, result, result);

                /*
                printf("----------square, , num=%d---------\n", num);
                printf("%f\n", get(result, 0, 0));
                printf("%f\n", get(result, 0, 1));
                printf("%f\n", get(result, 1, 0));
                printf("%f\n", get(result, 1, 1));
                */
            }
        }
        
    }

    /*
    printf("-----------after-------------\n");
    printf("%f\n", get(result, 0, 0));
    printf("%f\n", get(result, 0, 1));
    printf("%f\n", get(result, 1, 0));
    printf("%f\n", get(result, 1, 1));
    */



    return 0;

    
    
    
    /*
    pow_matrix(result, mat, pow / 2);
    
    if (pow % 2 == 0) {
        mul_matrix(result, result, result);
        return 0;
    } else {
        mul_matrix(result, result, result);
        mul_matrix(result, result, mat);
        return 0;
    }
    */


    /*
    if (mat->rows != mat->cols) {
        return 2;
    }
    //TO DO: Can pow be 0?
    if (pow < 0) {
        return 1;
    }
    if (pow == 0) {
        #pragma omp parallel for
        for (int i = 0; i < result->rows; i++) {
            for (int j = 0; j < result->cols; j++) {
                if (i == j) {
                    result->data[i*result->cols+j] = 1;
                }
            }
        }
        return 0;
    }
    int mat_size = mat->rows * mat->cols;
    #pragma omp parallel for
    for (int i = 0; i < mat_size; i++) {
        result->data[i] = mat->data[i];
    }
    for (int i = 1; i < pow; i++) {
        mul_matrix(result, result, mat); 
    }

    return 0;
    */
}

/*
 * (OPTIONAL)
 * Store the result of element-wise negating mat's entries to `result`.
 * Return 0 upon success and a nonzero value upon failure.
 */
int neg_matrix(matrix *result, matrix *mat) {
    /* TODO: YOUR CODE HERE */ 
    __m256d zeros = _mm256_set1_pd(0);
    int elements_num = mat->rows * mat->cols;

    __m256d load_mat;
    __m256d tmp;

    for (int i = 0; i < elements_num / 4 * 4; i += 4) {
        load_mat = _mm256_loadu_pd(mat->data + i);
        tmp = _mm256_sub_pd(zeros, load_mat);
        _mm256_storeu_pd(result->data + i, tmp);
    }
    for (int i = elements_num / 4 * 4; i < elements_num; i++) {
        result->data[i] = 0 - mat->data[i];
    }
    return 0;
}

/*
 * Store the result of taking the absolute value element-wise to `result`.
 * Return 0 upon success and a nonzero value upon failure.
 */
int abs_matrix(matrix *result, matrix *mat) {
    /* TODO: YOUR CODE HERE */
    __m256d zeros = _mm256_set1_pd(0);
    int elements_num = mat->rows * mat->cols;

    __m256d load_mat;
    __m256d tmp;

    #pragma omp parallel for
    for (int i = 0; i < elements_num / 4 * 4; i += 4) {
        load_mat = _mm256_loadu_pd(mat->data + i);
        tmp = _mm256_sub_pd(zeros, load_mat);
        tmp = _mm256_max_pd(load_mat, tmp);
        _mm256_storeu_pd(result->data + i, tmp);
    }
    #pragma omp parallel for
    for (int i = elements_num / 4 * 4; i < elements_num; i++) {
        if (mat->data[i] < 0) {
            result->data[i] = 0 - mat->data[i];
        } else {
            result->data[i] = mat->data[i];
        }
    }
    return 0;
}

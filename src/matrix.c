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
    for (int i = 0; i < elements_num / 4 * 4; i += 4) {
        __m256d set_val = _mm256_set1_pd(val);
        _mm256_storeu_pd(mat->data + i, set_val);
    }
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
    for (int i = 0; i < elements_num / 4 * 4; i += 4) {
        __m256d load_mat1 = _mm256_loadu_pd(mat1->data + i);
        __m256d load_mat2 = _mm256_loadu_pd(mat2->data + i);
        tmp = _mm256_add_pd(load_mat1, load_mat2);
        _mm256_storeu_pd(result->data + i, tmp);
    }
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
    if (rows1 != cols2 || rows1 < 0 || rows2 < 0 || cols1 < 0 || cols2 < 0) {
        return 1;
    }
    matrix *temp_result;
    allocate_matrix(&temp_result, rows1, cols2);
    for (int i = 0; i < rows1; i++) {
        for (int j = 0; j < cols2; j++) {
            int sum_ij = 0;
            for (int k = 0; k < cols1; k++) {
                // result[i][j]         data1[i][k]           data2[k][j]
                sum_ij += mat1->data[i*cols1+k] * mat2->data[k*cols2+j];
                //printf("data1[i][k]:%f * data2[k][j]:%f, sum=%d\n", get(mat1, i, k), get(mat2, k, j), sum_ij);
            }
            temp_result->data[i*cols1+j] = sum_ij;
        }
    }
    free(result->data);
    result->data = temp_result->data;
    free(temp_result);
    return 0;
}

/*
 * Store the result of raising mat to the (pow)th power to `result`.
 * Return 0 upon success and a nonzero value upon failure.
 * Remember that pow is defined with matrix multiplication, not element-wise multiplication.
 */
int pow_matrix(matrix *result, matrix *mat, int pow) {
    if (pow < 1) {
        return 1;
    }
    if (pow == 1) {
        result->data = mat->data;
        return 0;
    }
    mul_matrix(result, mat, mat);
    for (int i = 1; i < pow - 1; i++) {
        mul_matrix(result, result, mat);
    }


    return 0;
    /* TODO: YOUR CODE HERE */
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

    for (int i = 0; i < elements_num / 4 * 4; i += 4) {
        load_mat = _mm256_loadu_pd(mat->data + i);
        tmp = _mm256_sub_pd(zeros, load_mat);
        tmp = _mm256_max_pd(load_mat, tmp);
        _mm256_storeu_pd(result->data + i, tmp);
    }
    for (int i = elements_num / 4 * 4; i < elements_num; i++) {
        if (mat->data[i] < 0) {
            result->data[i] = 0 - mat->data[i];
        } else {
            result->data[i] = mat->data[i];
        }
    }
    return 0;
}

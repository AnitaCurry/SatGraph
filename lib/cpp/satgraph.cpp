#include <Python.h>
#include <stdio.h>
#include <stdint.h>
#include <omp.h>
#include <sched.h>
int OMPNUM = 2;

extern "C" {
void multiply_float (int32_t   size,
                     float   * vector_1,
                     float   * vector_2) {
  int32_t i = 0; 
  omp_set_dynamic(0);
  omp_set_num_threads(OMPNUM);
#pragma omp parallel for schedule(static)
  for (i = 0; i < size; i++) {
    vector_1[i] *= vector_2[i];
  }
}

void divide_float_int32 (int32_t   size,
                       float   * vector_1,
                       int32_t * vector_2) {
  int32_t i = 0; 
  omp_set_dynamic(0);
  omp_set_num_threads(OMPNUM);
#pragma omp parallel for schedule(static)
  for (i = 0; i < size; i++) {
    vector_1[i] /= vector_2[i];
  }
}

void ssp_min_float (int32_t * indices,        // sparse matrix indices
                    int32_t * indptr,         // sparse matrix indptr
                    int32_t   size_indptr,    // size of indptr
                    float   * vertex_value,   // changed vertex (row) val
                    float   * value) {        // vertex value of this matrix
  int32_t i   = 0;
  int32_t j   = 0;
  // int32_t tmp = 0;
  float   min = 0;
  omp_set_dynamic(0);
  omp_set_num_threads(OMPNUM);
//#pragma omp parallel for private(j, tmp, min) schedule(dynamic, 10000)
#pragma omp parallel for private(j, min) schedule(dynamic, 10000)
  for (i = 0; i < size_indptr-1; i++) {
    min = value[i];
    for (j = 0; j < indptr[i+1] - indptr[i]; j++) {
      // tmp = indices[indptr[i] + j];
      if (min > vertex_value[indices[indptr[i]+j]] + 1)
        min = vertex_value[indices[indptr[i] + j]];
    }
    value[i] = min;
  }
}


void pr_dot_product_float(int32_t * indices,           // sparse matrix indices
                          int32_t * indptr,            // sparse matrix indptr
                          int32_t   size_indptr,       // size of indptr
                          int32_t * act_vertex_id,     // active vertex ids (col)
                          int32_t   size_act_vertex,   // size of active vertex
                          float   * vertex,            // vertex data
                          float   * value,             // results
                          int32_t   vertex_num) {
  int32_t i   = 0;
  int32_t k   = 0;
  // int32_t tmp = 0;
  float   rel = 0;
  omp_set_dynamic(0);
  omp_set_num_threads(OMPNUM);
//#pragma omp parallel for private(k, tmp, rel) schedule(dynamic, 10000)
#pragma omp parallel for private(k, rel) schedule(dynamic, 10000)
  for (i=0; i < size_act_vertex; i++) {
    rel = 0;
    for (k = 0; k < indptr[act_vertex_id[i]+1] - indptr[act_vertex_id[i]]; k++) {
      rel += vertex[indices[indptr[act_vertex_id[i]] + k]];
      // rel += vertex[tmp];
    }
    value[act_vertex_id[i]] = rel*0.85 + 1.0/vertex_num;
  }
}

}

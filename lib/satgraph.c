#include <stdio.h>
#include <stdbool.h>
#include <stdint.h>
#include <omp.h>
#include <sched.h>
int OMPNUM = 2;

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

void multiply_double (int32_t   size,
                      double  * vector_1,
                      double  * vector_2) {
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

void divide_double_int32 (int32_t   size,
                          double  * vector_1,
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
                    bool    * act_vertex_id,     // active vertex ids (col)
                    float   * vertex_value,   // changed vertex (row) val
                    float   * value) {        // vertex value of this matrix
  int32_t i   = 0;
  int32_t j   = 0;
  // int32_t tmp = 0;
  float   min = 0;
  omp_set_dynamic(0);
  omp_set_num_threads(OMPNUM);
#pragma omp parallel for private(j, min) schedule(dynamic, 10000)
//#pragma omp parallel for private(j, min) schedule(dynamic)
  for (i = 0; i < size_indptr-1; i++) {
    min = value[i];
    for (j = 0; j < indptr[i+1] - indptr[i]; j++) {
      // tmp = indices[indptr[i] + j];
      if (min > vertex_value[indices[indptr[i]+j]] + 1)
        min = vertex_value[indices[indptr[i] + j]] + 1;
    }
    value[i] = min - value[i];
  }
}

void ssp_min_double (int32_t * indices,        // sparse matrix indices
                     int32_t * indptr,         // sparse matrix indptr
                     int32_t   size_indptr,    // size of indptr
                     bool    * act_vertex_id,     // active vertex ids (col)
                     double  * vertex_value,   // changed vertex (row) val
                     double  * value) {        // vertex value of this matrix
  int32_t i   = 0;
  int32_t j   = 0;
  // int32_t tmp = 0;
  double   min = 0;
  omp_set_dynamic(0);
  omp_set_num_threads(OMPNUM);
#pragma omp parallel for private(j, min) schedule(dynamic, 10000)
//#pragma omp parallel for private(j, min) schedule(dynamic)
  for (i = 0; i < size_indptr-1; i++) {
    min = value[i];
    for (j = 0; j < indptr[i+1] - indptr[i]; j++) {
      // tmp = indices[indptr[i] + j];
      if (min > vertex_value[indices[indptr[i]+j]] + 1)
        min = vertex_value[indices[indptr[i] + j]] + 1;
    }
    value[i] = min - value[i];
  }
}



void pr_dot_product_float(int32_t * indices,           // sparse matrix indices
                          int32_t * indptr,            // sparse matrix indptr
                          int32_t   size_indptr,       // size of indptr
                          bool    * act_vertex_id,     // active vertex ids (col)
                          int32_t   size_vertex,   // size of active vertex
                          int32_t * outgoing,
                          float   * vertex,            // vertex data
                          float   * value,             // results
                          int32_t   vertex_num) {
  int32_t i   = 0;
  int32_t k   = 0;
  int32_t tmp = 0;
  float   rel = 0;
  omp_set_dynamic(0);
  omp_set_num_threads(OMPNUM);
#pragma omp parallel for private(k, tmp, rel) schedule(dynamic, 10000)
//#pragma omp parallel for private(k, tmp, rel) schedule(dynamic)
  for (i=0; i < size_vertex; i++) {
    rel = 0;
    for (k = 0; k < indptr[i+1] - indptr[i]; k++) {
      tmp = indices[indptr[i] + k];
      rel += vertex[tmp]/outgoing[tmp];
      // rel += vertex[tmp];
    }
    value[i] = rel*0.85 + 1.0/vertex_num - value[i];
  }
}


void pr_dot_product_double(int32_t * indices,           // sparse matrix indices
                           int32_t * indptr,            // sparse matrix indptr
                           int32_t   size_indptr,       // size of indptr
                           bool    * act_vertex_id,     // active vertex ids (col)
                           int32_t   size_vertex,   // size of active vertex
                           int32_t * outgoing,
                           double  * vertex,            // vertex data
                           double  * value,             // results
                           int32_t   vertex_num) {
  int32_t i   = 0;
  int32_t k   = 0;
  int32_t tmp = 0;
  double   rel = 0;
  omp_set_dynamic(0);
  omp_set_num_threads(OMPNUM);
#pragma omp parallel for private(k, tmp, rel) schedule(dynamic, 10000)
//#pragma omp parallel for private(k, tmp, rel) schedule(dynamic)
  for (i=0; i < size_vertex; i++) {
    rel = 0;
    for (k = 0; k < indptr[i+1] - indptr[i]; k++) {
      tmp = indices[indptr[i] + k];
      rel += vertex[tmp]/outgoing[tmp];
      // rel += vertex[tmp];
    }
    value[i] = rel*0.85 + 1.0/vertex_num - value[i];
  }
}

void component_float (int32_t * indices,        // sparse matrix indices
                      int32_t * indptr,         // sparse matrix indptr
                      int32_t   size_indptr,    // size of indptr
                      bool    * act_vertex_id,     // active vertex ids (col)
                      float   * vertex_value,   // changed vertex (row) val
                      float   * value) {        // vertex value of this matrix
  int32_t i   = 0;
  int32_t j   = 0;
  float   max = 0;
  omp_set_dynamic(0);
  omp_set_num_threads(OMPNUM);
#pragma omp parallel for private(j, max) schedule(dynamic, 10000)
  for (i = 0; i < size_indptr-1; i++) {
    max = value[i];
    for (j = 0; j < indptr[i+1] - indptr[i]; j++) {
      if (act_vertex_id[indices[indptr[i]+j]] && max < vertex_value[indices[indptr[i]+j]])
        max = vertex_value[indices[indptr[i] + j]];
    }
    value[i] = max - value[i];
  }
}


void component_double (int32_t * indices,        // sparse matrix indices
                       int32_t * indptr,         // sparse matrix indptr
                       int32_t   size_indptr,    // size of indptr
                       bool    * act_vertex_id,     // active vertex ids (col)
                       double  * vertex_value,   // changed vertex (row) val
                       double  * value) {        // vertex value of this matrix
  int32_t i   = 0;
  int32_t j   = 0;
  double  max = 0;
  omp_set_dynamic(0);
  omp_set_num_threads(OMPNUM);
#pragma omp parallel for private(j, max) schedule(dynamic, 10000)
  for (i = 0; i < size_indptr-1; i++) {
    max = value[i];
    for (j = 0; j < indptr[i+1] - indptr[i]; j++) {
      // if (act_vertex_id[indices[indptr[i]+j]] && max < vertex_value[indices[indptr[i]+j]])
      if (max < vertex_value[indices[indptr[i]+j]])
        max = vertex_value[indices[indptr[i] + j]];
    }
    value[i] = max - value[i];
  }
}


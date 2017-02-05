#include <stdio.h>
#include <stdint.h>

void multiply_min_float (int32_t * indices,        // sparse matrix indices
                         int32_t * indptr,         // sparse matrix indptr
                         int32_t   size_indptr,    // size of indptr
                         int32_t * vertex_id,      // changed vertex (row) id
                         float   * vertex_value,   // changed vertex (row) val
                         int32_t   size_vertex,    // size of changed vertex
                         float   * value) {        // vertex value of this matrix
  int32_t i   = 0;
  int32_t j   = 0;
  int32_t k   = 0;
  int32_t tmp = 0;
  float   min = 0;

  for (; i < size_indptr-1; i++) {
    min = value[i];
    j = 0;
    k = 0;
    for (; j < indptr[i+1] - indptr[i]; j++) {
      tmp = indices[indptr[i] + j];
      for (; k < size_vertex; k++) {
        if (tmp < vertex_id[k]) 
          break;
        if (tmp == vertex_id[k]) {
          if (min > vertex_value[k])
            min = vertex_value[k];
          break;
        }
      }
      if (k == size_vertex)
        break;
    }
    value[i] = min;
  }
}


void dot_product_float(int32_t * indices,           // sparse matrix indices
                       int32_t * indptr,            // sparse matrix indptr
                       int32_t   size_indptr,       // size of indptr
                       int32_t * act_vertex_id,     // active vertex ids (col)
                       int32_t   size_act_vertex,   // size of active vertex
                       float   * vertex,            // vertex data
                       float   * value) {           // results
  int32_t act = 0;
  int32_t i   = 0;
  int32_t j   = 0;
  int32_t k   = 0;
  int32_t tmp = 0;
  float   rel = 0;

  for (; i < size_indptr-1; i++) {
    act = 0;
    for (; j < size_act_vertex; j++) {
      if (i == act_vertex_id[j]) {
        act = 1;
        j++;
        break;
      }
    }
    if (act == 0) 
      continue;
    rel = 0;
    for (k = 0; k < indptr[i+1] - indptr[i]; k++) {
      tmp = indices[indptr[i] + k];
      rel += vertex[tmp];
    }
    value[i] = rel;
  }
}

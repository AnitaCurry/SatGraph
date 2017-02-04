#include <stdio.h>
#include <stdint.h>

void sssp_unweight(int32_t * indices, 
                   int32_t * indptr,
                   int32_t   size_indptr,
                   int32_t * vertex_id,
                   int32_t * vertex_value,
                   int32_t   size_vertex,
                   int32_t * value) {
  int32_t i   = 0;
  int32_t j   = 0;
  int32_t k   = 0;
  int32_t tmp = 0;
  int32_t min = 0;

  for (i = 0; i < size_indptr; i++) {
    min = value[i];
    for (j = 0; j < indptr[i+1] - indptr[i]; j++) {
      tmp = indices[indptr[i] + j];
      for (k = 0; k < size_vertex; k++) {
        if (tmp == vertex_id[k]) {
          if (min < vertex_value[k]) {
            min = vertex_value[k];
          }
        }
      }
    }
    value[i] = min;
  }
}

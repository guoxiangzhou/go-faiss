#include "faiss_cgo.h"
#include <stdio.h>
#include <stdlib.h>

#include <time.h>

FaissIndex* CreateIndex(int dimension, const char* description, FaissMetricType metric_type) {
  FaissIndex* index = NULL;
  faiss_index_factory(&index, dimension, description, metric_type);
  return index;
}

void InsertWithID(FaissIndex* index, float* vectors,int num, long* ids) {
  faiss_Index_add_with_ids(index, num, vectors, ids);
}

void Insert(FaissIndex* index, float* vectors, int num) {
  faiss_Index_add(index, num, vectors);
}

void Train(FaissIndex* index, float* vectors, int num) {
  faiss_Index_train(index, num, vectors);
}

void Search(FaissIndex* index, float* vectors, int nq, int topk, long* ids, float* distances) {
  faiss_Index_search(index, nq, vectors, topk, distances, ids);
}

int GetTotal(FaissIndex* index) {
  return faiss_Index_ntotal(index);
}

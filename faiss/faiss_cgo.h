#include "include/error_c.h"
#include "include/index_io_c.h"
#include "include/index_factory_c.h"
#include "include/Index_c.h"
#include "include/IndexFlat_c.h"
#include "include/AutoTune_c.h"
#include "include/clone_index_c.h"

FaissIndex* CreateIndex(int dimension, const char* description, FaissMetricType metric_type);

void InsertWithID(FaissIndex* index, float* vectors, int num, long* ids);

void Insert(FaissIndex* index, float* vectors, int num);

void Train(FaissIndex* index, float* vectors, int num);

void Search(FaissIndex* index, float* vectors, int nq, int topk, long* ids, float* distances);

int GetTotal(FaissIndex* index);

//WriteIndex2File

//ReadIndexFromFile

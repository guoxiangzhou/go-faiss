package faiss

//#cgo CFLAGS: -I./include
//#cgo LDFLAGS: -lfaiss_c
//
//#include "faiss_cgo.h"
import "C"

import (
	"math/rand"
	"time"
	"unsafe"
)

type Result struct {
	Distance float32
	ID       int64
}

type Vectors []float32

func GenVectors(dataSize int, dimension int) Vectors {
	rand.Seed(time.Now().UnixNano())
	vectors := make([]float32, dataSize*dimension)
	for i := 0; i < dataSize; i++ {
		for j := 0; j < dimension; j++ {
			d := rand.Float32()
			vectors[dimension*i+j] = d
			vectors[dimension*i] += float32(i) / float32(1000)
		}
	}
	return vectors
}

func GenIDs(dataSize int) []int64 {
	//rand.Seed(time.Now().UnixNano())
	ids := make([]int64, dataSize)
	for i := 0; i < dataSize; i++ {
		ids[i] = int64(i) + 100 //rand.Int31()
	}
	return ids
}

type IndexParam struct {
	Dimension   int
	Description string
	MetricType  string
}

type Index struct {
	Index *C.FaissIndex
}

func (i *Index) Create(param *IndexParam) {
	var metricType C.FaissMetricType
	if param.MetricType == "L2" {
		metricType = C.METRIC_L2
	}
	i.Index = C.CreateIndex(C.int(param.Dimension), C.CString(param.Description), metricType)
}

func InsertVectorsWithID(v Vectors, index *C.FaissIndex, dimension int, ids []int64) error {
	C.InsertWithID(index, (*C.float)(unsafe.Pointer(&v[0])), C.int(dimension), (*C.long)(unsafe.Pointer(&ids[0])))
	return nil
}

func InsertVectors(v Vectors, index *C.FaissIndex, dimension int) error {
	C.Insert(index, (*C.float)(unsafe.Pointer(&v[0])), C.int(dimension))
	return nil
}

func Train(v Vectors, index *C.FaissIndex, dimension int) error {
	C.Train(index, (*C.float)(unsafe.Pointer(&v[0])), C.int(dimension))
	return nil
}

func SearchVectors(v Vectors, index *C.FaissIndex, nq int, topk int, resIDs []int64, resDistances []float32) []Result {
	res := []Result{}
	C.Search(index, (*C.float)(unsafe.Pointer(&v[0])), C.int(nq), C.int(topk),
		(*C.long)(unsafe.Pointer(&resIDs[0])), (*C.float)(unsafe.Pointer(&resDistances[0])))
	for i := 0; i < topk; i++ {
		res = append(res, Result{
			Distance: resDistances[i],
			ID:       resIDs[i],
		})
	}
	return res
}

func GetTotal(index *C.FaissIndex) int {
	return int(C.GetTotal(index))
}

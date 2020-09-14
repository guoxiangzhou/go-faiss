package main

import "C"
import (
	"fmt"

	"github.com/41tair/go-faiss/faiss"
)

func main() {
	dimension := 128
	datasize := 100
	v := faiss.GenVectors(datasize, dimension)

	params := &faiss.IndexParam{
		Dimension:   dimension,
		Description: "IDMap,Flat",
		MetricType: "L2",
	}

	index := faiss.Index{}
	index.Create(params)
	i := index.Index

	faiss.Train(v, i, datasize)

	ids := faiss.GenIDs(datasize)

	err := faiss.InsertVectorsWithID(v, i, datasize, ids)
	if err != nil {
		fmt.Println(err)
	}

	fmt.Println("----------", faiss.GetTotal(i))

	//err = faiss.InsertVectors2(v, i, datasize)
	//if err != nil {
	//	fmt.Println(err)
	//}
	//
	//fmt.Println("xxxxxxxxxxx", faiss.GetTotal(i))

	v2 := faiss.GenVectors(10, datasize)
	resIDs := make([]int64, 10*1000*100)
	resDistances := make([]float32, 10*1000*100)
	res := faiss.SearchVectors(v2, i, 1000, 10, resIDs, resDistances)

	for _, v := range res {
		fmt.Printf("ID: %v Distance: %v \n", v.ID, v.Distance)
	}
}

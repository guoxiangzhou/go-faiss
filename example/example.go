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

	index := &faiss.Index{
		Dimension:   dimension,
		Description: "IDMap,Flat",
		MetricType:  faiss.L2{},
	}
	i, _ := index.Create()

	ids := faiss.GenIDs(datasize)

	err := faiss.InsertVectors(v, i, datasize, ids)
	if err != nil {
		fmt.Println(err)
	}

	fmt.Println("----------", faiss.GetTotal(i))

	err = faiss.InsertVectors(v, i, datasize, ids)
	if err != nil {
		fmt.Println(err)
	}

	fmt.Println("xxxxxxxxxxx", faiss.GetTotal(i))

	v2 := faiss.GenVectors(10, datasize)
	resIDs := make([]int32, 10*1000*100)
	resDistances := make([]float32, 10*1000*100)
	res := faiss.SearchVectors(v2, i, 1000, 10, resIDs, resDistances)

	for _, v := range res {
		fmt.Printf("ID: %v Distance: %v \n", v.ID, v.Distance)
	}
}

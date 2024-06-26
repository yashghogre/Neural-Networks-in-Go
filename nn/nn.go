package nn

import (
	"fmt"
	"math"
)

type Drop struct {
	Value    float32
	Grad     float32
	Prev     []*Drop
	backward func()
}

func NewDrop(value float32) *Drop {
	return &Drop{
		Value:    value,
		backward: func() {},
	}
}

func NewDropArray(value []float32) []*Drop {
	drop_array := make([]*Drop, len(value))

	for i := 0; i < len(value); i++ {
		drop_array[i] = NewDrop(value[i])
	}

	return drop_array
}

func NewDropMatrix(value [][]float32) [][]*Drop {
	tmp_arr := make([][]*Drop, len(value))

	for i, row := range value {
		tmp_arr[i] = make([]*Drop, len(row))
		for j, val := range row {
			tmp_arr[i][j] = NewDrop(val)
		}
	}

	return tmp_arr
}

func DropArrayToFloat(drops []*Drop) []float32 {
	float_arr := make([]float32, len(drops))

	for i := 0; i < len(drops); i++ {
		float_arr[i] = drops[i].Value
	}

	return float_arr
}

func (d *Drop) GetItem() {
	fmt.Println("( Drop: ", d.Value, ")")
}

func (d *Drop) Add(other *Drop) *Drop {
	out := &Drop{Value: (d.Value + other.Value), Prev: []*Drop{d, other}}

	backward := func() {
		d.Grad += out.Grad
		other.Grad += out.Grad
	}
	out.backward = backward
	return out
}

func (d *Drop) Mul(other *Drop) *Drop {
	out := &Drop{Value: (d.Value * other.Value), Prev: []*Drop{d, other}}

	backward := func() {
		d.Grad += out.Grad * other.Value
		other.Grad += out.Grad * d.Value
	}
	out.backward = backward
	return out
}

func (d *Drop) Sub(other *Drop) *Drop {
	out := &Drop{Value: (d.Value - other.Value), Prev: []*Drop{d, other}}

	backward := func() {
		d.Grad += out.Grad
		other.Grad -= out.Grad
	}
	out.backward = backward
	return out
}

func (d *Drop) Div(other *Drop) *Drop {
	out := &Drop{Value: (d.Value / other.Value), Prev: []*Drop{d, other}}

	backward := func() {
		d.Grad += out.Grad * other.Value
		other.Grad += -out.Grad * d.Value / (other.Value * other.Value)
	}
	out.backward = backward
	return out
}

func (d *Drop) Relu() *Drop {

	var tmp float32

	if d.Value < 0 {
		tmp = 0
	} else {
		tmp = d.Value
	}

	out := &Drop{Value: tmp, Prev: []*Drop{d}}

	backward := func() {
		if d.Value > 0 {
			d.Grad += 1 * out.Grad
		}
	}
	out.backward = backward
	return out
}

func (d *Drop) Tanh() *Drop {

	var tmp = float32((math.Exp(float64(d.Value)) - math.Exp(-float64(d.Value))) / (math.Exp(float64(d.Value)) + math.Exp(-float64(d.Value))))
	out := &Drop{Value: tmp, Prev: []*Drop{d}}

	backward := func() {
		d.Grad += out.Grad * float32(1-math.Pow(float64(tmp), 2))
	}
	out.backward = backward
	return out
}

func (d *Drop) Sigmoid() *Drop {

	var tmp = float32(1 / (1 + math.Exp(-float64(d.Value))))
	out := &Drop{Value: tmp, Prev: []*Drop{d}}

	backward := func() {
		d.Grad += out.Grad * (tmp * (1 - tmp))
	}
	out.backward = backward
	return out
}

func (d *Drop) Sub_Grad(lr float32) {
	d.Value = d.Value - (lr * d.Grad)
}

func (d *Drop) Backward() {
	var topo []*Drop
	var visited []*Drop

	contains := func(values []*Drop, Value *Drop) bool {
		for _, item := range values {
			if item == Value {
				return true
			}
		}
		return false
	}

	var build_topo func(*Drop)
	build_topo = func(v *Drop) {
		if !contains(visited, v) {
			visited = append(visited, v)
			for _, child := range v.Prev {
				build_topo(child)
			}
			topo = append(topo, v)
		}
	}
	build_topo(d)

	for i, j := 0, len(topo)-1; i < j; i, j = i+1, j-1 {
		topo[i], topo[j] = topo[j], topo[i]
	}
	d.Grad = 1
	for _, v := range topo {
		v.backward()
	}
}

func MatMul(mat1 [][]float32, mat2 [][]float32) [][]*Drop {

	if len(mat1[0]) != len(mat2) {
		fmt.Println("Can't multiply matrices with improper dimensions. Can't multiply matrix [", len(mat1), "x", len(mat1[0]), "] and matrix [", len(mat2), "x", len(mat2[0]), "]")
	}
	rows := len(mat1)
	cols := len(mat2[0])
	result_mat := make([][]*Drop, rows)

	for i := range result_mat {
		result_mat[i] = make([]*Drop, cols)
	}

	mat1_drop := NewDropMatrix(mat1)
	mat2_drop := NewDropMatrix(mat2)

	for i := 0; i < rows; i++ {
		for j := 0; j < cols; j++ {
			sum := float32(0)
			for k := 0; k < len(mat1_drop[0]); k++ {
				sum += mat1_drop[i][k].Value * mat2_drop[k][j].Value
			}
			result_mat[i][j] = NewDrop(sum)
		}
	}

	return result_mat
}

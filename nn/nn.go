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

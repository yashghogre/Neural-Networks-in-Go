package main

import (
	"fmt"
	"math"
)

type Drop struct {
	value     float32
	grad      float32
	prev      []*Drop
	_backward func()
}

func (d *Drop) getItem() {
	fmt.Println("( Drop: ", d.value, ")")
}

func (d *Drop) add(other *Drop) *Drop {
	out := &Drop{value: (d.value + other.value), prev: []*Drop{d, other}}

	_backward := func() {
		d.grad += out.grad
		other.grad += out.grad
	}
	out._backward = _backward
	return out
}

func (d *Drop) mul(other *Drop) *Drop {
	out := &Drop{value: (d.value * other.value), prev: []*Drop{d, other}}

	_backward := func() {
		d.grad += out.grad * other.value
		other.grad += out.grad * d.value
	}
	out._backward = _backward
	return out
}

func (d *Drop) sub(other *Drop) *Drop {
	out := &Drop{value: (d.value - other.value), prev: []*Drop{d, other}}

	_backward := func() {
		d.grad += out.grad
		other.grad -= out.grad
	}
	out._backward = _backward
	return out
}

func (d *Drop) div(other *Drop) *Drop {
	out := &Drop{value: (d.value / other.value), prev: []*Drop{d, other}}

	_backward := func() {
		d.grad = out.grad * other.value
		other.grad = -out.grad * d.value / (other.value * other.value)
	}
	out._backward = _backward
	return out
}

func (d *Drop) relu() *Drop {

	var tmp float32

	if d.value < 0 {
		tmp = 0
	} else {
		tmp = d.value
	}

	out := &Drop{value: tmp, prev: []*Drop{d}}

	_backward := func() {
		if d.value > 0 {
			d.grad += 1 * out.grad
		}
	}
	out._backward = _backward
	return out
}

func (d *Drop) tanh() *Drop {

	var tmp = float32((math.Exp(float64(d.value)) - math.Exp(-float64(d.value))) / (math.Exp(float64(d.value)) + math.Exp(-float64(d.value))))
	out := &Drop{value: tmp, prev: []*Drop{d}}

	_backward := func() {
		d.grad += out.grad * float32(1-math.Pow(float64(tmp), 2))
	}
	out._backward = _backward
	return out
}

func (d *Drop) sigmoid() *Drop {

	var tmp = float32(1 / (1 + math.Exp(-float64(d.value))))
	out := &Drop{value: tmp, prev: []*Drop{d}}

	_backward := func() {
		d.grad += out.grad * (tmp * (1 - tmp))
	}
	out._backward = _backward
	return out
}

func main() {
	a := &Drop{value: 4}
	// b := &Drop{value: 5}
	// c := &Drop{value: 2}

	// result_0 := a.mul(b)
	// result_1 := result_0.add(c)
	// result_1.grad = 1

	// result_1._backward()
	// result_0._backward()

	// fmt.Println(a.grad)
	// fmt.Println((*result_0.prev[0]).value)
	// fmt.Println(c.grad)

	// print((a.relu()))
	var d = a.tanh()
	d.grad = 10
	d._backward()
	fmt.Println(a)
}

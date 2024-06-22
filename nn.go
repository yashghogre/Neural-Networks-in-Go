package main

import (
	"fmt"
	// "math"
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

func main() {
	a := &Drop{value: 4}
	b := &Drop{value: 5}
	c := &Drop{value: 2}

	result_0 := a.mul(b)
	result_1 := result_0.add(c)
	result_1.grad = 1

	result_1._backward()
	result_0._backward()

	fmt.Println(a.grad)
	fmt.Println(b.grad)
	fmt.Println(c.grad)
}

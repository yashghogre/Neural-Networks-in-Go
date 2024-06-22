package main

import (
	"fmt"
	// "math"
)

type Drop struct {
	value     float32
	grad      float32
	prev      []Drop
	_backward func()
}

func (d Drop) getItem() {
	fmt.Println("( Drop: ", d.value, ")")
}

func (d *Drop) add(other *Drop) Drop {
	out := Drop{value: (d.value + other.value), prev: []Drop{*d, *other}}

	_backward := func() {
		d.grad += out.grad
		other.grad += out.grad
	}
	out._backward = _backward
	return out
}

func (d *Drop) mul(other *Drop) Drop {
	out := Drop{value: (d.value * other.value), prev: []Drop{*d, *other}}

	_backward := func() {
		d.grad = out.grad * other.value
		other.grad = out.grad * d.value
	}
	out._backward = _backward
	return out
}

func (d *Drop) sub(other *Drop) Drop {
	out := Drop{value: (d.value - other.value), prev: []Drop{*d, *other}}

	_backward := func() {
		d.grad += out.grad
		other.grad += out.grad
	}
	out._backward = _backward
	return out
}

func (d *Drop) div(other *Drop) Drop {
	out := Drop{value: (d.value / other.value), prev: []Drop{*d, *other}}

	_backward := func() {
		d.grad = out.grad * other.value
		other.grad = out.grad * d.value
	}
	out._backward = _backward
	return out
}

// func (d *Drop) pow(other *Drop) Drop {
// 	out := Drop{value: float32(math.Pow(float64(d.value), float64(other.value))), prev: []Drop{*d, *other}}

// 	_backward := func() {
// 		d.grad =
// 	}
// 	return out
// }

func main() {
	a := Drop{value: 4}
	b := Drop{value: 5}
	c := Drop{value: 2}

	var result_0 = (a.mul(&b))
	var result_1 = result_0.add(&c)
	result_1.grad = 5

	result_1._backward()

	fmt.Println(result_0.grad)
}

package main

import (
	"fmt"
	"nn/nn"
)

func main() {

	// Data Declaration and definition
	x1 := nn.NewDrop(4)
	x2 := nn.NewDrop(5)

	w1 := nn.NewDrop(0.5)
	w2 := nn.NewDrop(0.5)

	b := nn.NewDrop(1)
	var lr float32 = 100

	// Forward Pass
	x1w1 := x1.Mul(w1)
	x2w2 := x2.Mul(w2)
	m := x1w1.Add(x2w2)
	n := m.Add(b)
	o := n.Tanh()

	fmt.Printf("\n\n")
	fmt.Println("----------------Grads and Results before Gradient Descent------------------")
	fmt.Println("Value w1: ", w1.Value)
	fmt.Println("Value w2: ", w2.Value)
	fmt.Println("Value b: ", b.Value)
	fmt.Println("Value final: ", o.Value)
	fmt.Printf("\n")

	// Backward Pass and Grdient Descent
	o.Backward()
	w1.Sub_Grad(lr)
	w2.Sub_Grad(lr)
	b.Sub_Grad(lr)

	// Another Forward Pass
	x1w1 = x1.Mul(w1)
	x2w2 = x2.Mul(w2)
	m = x1w1.Add(x2w2)
	n = m.Add(b)
	o = n.Tanh()

	fmt.Println("----------------Grads and Results after Gradient Descent------------------")
	fmt.Println("Value w1: ", w1.Value)
	fmt.Println("Value w2: ", w2.Value)
	fmt.Println("Value b: ", b.Value)
	fmt.Println("Value final: ", o.Value)
	fmt.Printf("\n\n")

}

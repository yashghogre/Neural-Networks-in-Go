package main

import (
	"fmt"
	"nn/nn"
)

func main() {

	// Data Declaration and definition
	// x1 := nn.NewDrop(4)
	// x2 := nn.NewDrop(5)

	// w1 := nn.NewDrop(0.5)
	// w2 := nn.NewDrop(0.5)

	// b := nn.NewDrop(1)
	// var lr float32 = 100

	// // Forward Pass
	// x1w1 := x1.Mul(w1)
	// x2w2 := x2.Mul(w2)
	// m := x1w1.Add(x2w2)
	// n := m.Add(b)
	// o := n.Tanh()

	// fmt.Printf("\n\n")
	// fmt.Println("----------------Grads and Results before Gradient Descent------------------")
	// fmt.Println("Value w1: ", w1.Value)
	// fmt.Println("Value w2: ", w2.Value)
	// fmt.Println("Value b: ", b.Value)
	// fmt.Println("Value final: ", o.Value)
	// fmt.Printf("\n")

	// // Backward Pass and Grdient Descent
	// o.Backward()
	// w1.Sub_Grad(lr)
	// w2.Sub_Grad(lr)
	// b.Sub_Grad(lr)

	// // Another Forward Pass
	// x1w1 = x1.Mul(w1)
	// x2w2 = x2.Mul(w2)
	// m = x1w1.Add(x2w2)
	// n = m.Add(b)
	// o = n.Tanh()

	// fmt.Println("----------------Grads and Results after Gradient Descent------------------")
	// fmt.Println("Value w1: ", w1.Value)
	// fmt.Println("Value w2: ", w2.Value)
	// fmt.Println("Value b: ", b.Value)
	// fmt.Println("Value final: ", o.Value)
	// fmt.Printf("\n\n")

	// n1 := nn.NewNeuron(4)
	// var inputs = []float32{2, 4, 1, 4}

	// n1_forward := n1.Forward_Neuron(inputs)
	// for i := range len(inputs) {
	// 	fmt.Println(n1.W[i])
	// }
	// fmt.Println(n1.B)
	// fmt.Println(n1_forward)

	var x = []float32{3, 4, 2}
	l := nn.NewLayer(3, 3)
	fmt.Println(l.N[0])
	var out = l.Forward_Layer(x)
	fmt.Println(out)
	l.Layer_Outputs(x)
}

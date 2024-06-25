package nn

import (
	// "fmt"
	"math/rand"
)

type Neuron struct {
	W []*Drop
	B *Drop
}

func NewNeuron(n_in int) *Neuron {
	tmp_arr := make([]float32, n_in)
	tmp_w := make([]*Drop, n_in)

	for i := 0; i < n_in; i++ {
		tmp_arr[i] = rand.Float32()
	}

	for i := 0; i < n_in; i++ {
		tmp_n := NewDrop(tmp_arr[i])
		tmp_w[i] = tmp_n
	}

	tmp_b := NewDrop(rand.Float32())

	return &Neuron{W: tmp_w, B: tmp_b}
}

func (n *Neuron) Forward_Neuron(inputs []float32) *Drop {
	inp_arr := NewDropArray(inputs)
	result := NewDrop(0)

	for i := 0; i < len(inputs); i++ {
		result = result.Add(inp_arr[i].Mul(n.W[i]))
	}

	result = result.Add(n.B)
	result = result.Relu()

	return result
}

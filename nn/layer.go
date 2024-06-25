package nn

import "fmt"

type Layer struct {
	N []*Neuron
}

func NewLayer(n_in int, n_neu int) *Layer {
	n_arr := make([]*Neuron, n_neu)

	for i := 0; i < n_neu; i++ {
		n_arr[i] = NewNeuron(n_in)
	}

	return &Layer{N: n_arr}
}

func (l *Layer) Forward_Layer(inputs []float32) []*Drop {
	out_arr := make([]*Drop, len(l.N))

	for i := 0; i < len(l.N); i++ {
		out_arr[i] = (l.N[i]).Forward_Neuron(inputs)
	}

	return out_arr
}

func (l *Layer) Layer_Outputs(inputs []float32) {

	out_arr := l.Forward_Layer(inputs)
	for i := 0; i < len(l.N); i++ {
		fmt.Println("Output from Neuron ", i+1, ": ", out_arr[i])
	}
}

package nn

// import "fmt"

type MLP struct {
	Layers []*Layer
}

func NewMLP(n_in int, n_neu []int) *MLP { // n_in is the no of inputs for first layer and n_neu is the list of no of neurons in each layer
	var layers_arr []*Layer

	tmp_arr := []int{n_in}
	sz := append(tmp_arr, n_neu...)

	for i := 0; i < len(n_neu); i++ {
		layers_arr = append(layers_arr, NewLayer(sz[i], sz[i+1]))
	}

	return &MLP{Layers: layers_arr}
}

func (mlp *MLP) Forward_MLP(inputs []float32) []*Drop {
	var drop_arr = NewDropArray(inputs)

	for i := 0; i < len(mlp.Layers); i++ {
		float_arr := DropArrayToFloat(drop_arr)
		drop_arr = (mlp.Layers[i]).Forward_Layer(float_arr)
	}

	return drop_arr
}

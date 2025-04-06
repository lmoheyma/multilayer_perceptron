# Multilayer perceptron

The multilayer perceptron is a feedforward network (meaning that the data flows from the input layer to the output layer) defined by the presence of one or more hidden layers as well as an interconnection of all the neurons of one layer to the next.

![multilayer perceptron](img/multilayer_perceptron.png)

The diagram above represents a network containing 4 dense layers (also called fully connected layers). Its inputs consist of 4 neurons and its output of 2 (perfect for binary classification). The weights of one layer to the next are represented by two dimensional matrices noted $W_{l_jl_{j+1}}$. The matrix $W_{l_0l_1}$ is of size (3, 4) for example, as it contains the weights of the connections between the layer $l_0$ and the layer $l_1$.

The bias is often represented as a special neuron which has no inputs and with an output always equal to 1. Like a perceptron it is connected to all the neurons of the following layer (the bias neurons are noted $b^{l_j}$ on the diagram above). The bias is generally useful as it allows to “control the behavior” of a layer.

# Perceptron

The perceptron is the type of neuron that the _multilayer perceptron_ is composed of. They are defined by the presence of one or more input connections, an activation function and a single output. Each connection contains a weight (also called parameter) which is learned during the training phase.

![perceptron](img/perceptron.png)

Two steps are necessary to get the output of a neuron. The first one consists in computing the weighted sum of the outputs of the previous layer with the weights of the input connections of the neuron, which gives

$$weigted\;sum=\sum_{k=0}^{n-1}(x_k \cdot w_k)+bias$$

The second step consists in applying an activation function on this weighted sum, the output of this function being the output of the perceptron, and can be understood as the threshold above which the neuron is activated (activation functions can take a lot of shapes, you are free to chose whichever one you want depending on the model to train, here are some of the most frequently used ones to give you an idea : sigmoid, hyperboloid tangent, rectified linear unit).
try:
    from base import Layer
    from utils import export_class
except ImportError:
    from .base import Layer
    from .utils import export_class
import numpy as np


@export_class
class Dense(Layer):
    """
    A Dense Layer of nodes in the Neural Network Model
    Attributes
    ----------
    input  : numpy.ndarray
        The input matrix to be used by the layer
    output : numpy.ndarray
        The output matrix produced by the layer
    weights  : numpy.ndarray
        The weight matrix to be used by the layer
    biases : numpy.ndarray
        The bias matrix produced by the layer
    Methods
    -------
    forward(input: numpy.ndarray) -> numpy.ndarray:
        Passes the output to the next layer in forward propagation
    backward(output_gradient: numpy.ndarray,
              learning_rate: float = 0.01) -> numpy.ndarray:
        Passes the output to the previous layer in backward propagation
    """

    def __init__(self, input_nodes: np.ndarray, output_nodes: np.ndarray):
        """
        Constructs all the necessary attributes for the Dense object
        Parameters
        ----------
            input_node: numpy.ndarray
                Number of input nodes on the dense layer
                Used to initialise the Weights Matrix
            output_node: numpy.ndarray
                Number of input nodes on the dense layer
                Used to initialise the Weights and Bias Matrix
        """
        self.weights = np.random.randn(output_nodes, input_nodes)
        self.biases = np.random.randn(output_nodes, 1)
        self.input_nodes = input_nodes
        self.output_nodes = output_nodes

    def forward(self, input: np.ndarray) -> np.ndarray:
        self.input = input
        return (self.weights @ self.input) + self.biases

    def backward(self, output_gradient: np.ndarray,
                 learning_rate: float = 0.01) -> np.ndarray:
        self.output = self.weights.T @ output_gradient
        self.weights -= learning_rate * (output_gradient @ self.input.T)
        self.biases -= learning_rate * output_gradient
        return self.output

    def __str__(self):
        return f'Dense layer of shape ({self.input_nodes}, {self.output_nodes})'

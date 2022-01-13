import numpy as np
try:
    from .base import Layer
    from .utils import export_class
except ImportError:
    from base import Layer
    from utils import export_class


@export_class
class Reshape(Layer):
    def __init__(self, input_shape, output_shape, /):
        self.input_shape = input_shape
        self.output_shape = output_shape

    def forward(self, input, /) -> np.ndarray:
        return np.reshape(input, self.output_shape)

    def backward(self, output_gradient, /, *, learning_rate: float = 0.01) -> np.ndarray:
        return np.reshape(output_gradient, self.input_shape)

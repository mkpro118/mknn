import numpy as np
from math import inf as infinity
try:
    from base import CostFunction, Layer, Activation
    from dense import Dense
    from activators import Softmax, TanH, exports as activators_exports
    from cost import MeanSquaredError, exports as cost_exports
    from exceptions import (UnfittedModelError,
                            InvalidCostFunctionError,
                            InvalidLearningRateError,
                            InvalidModelNetworkError,
                            InvalidSavedModelError)
    from utils import export_class
except ImportError:
    from .base import CostFunction, Layer, Activation
    from .dense import Dense
    from .activators import Softmax, TanH, exports as activators_exports
    from .cost import MeanSquaredError, exports as cost_exports
    from .exceptions import (UnfittedModelError,
                             InvalidCostFunctionError,
                             InvalidLearningRateError,
                             InvalidModelNetworkError,
                             InvalidSavedModelError)
    from .utils import export_class


@export_class
class Model:
    """
    Objects of this class are trainable Neural Network Models
    Attributes
    ----------
        network       : list
        X             : numpy.ndarray
        Y             : numpy.ndarray
        cost_function : CostFunctions  = MeanSquaredError()
        learning_rate : float          = 0.01
    Methods
    -------
        fit(X: np.ndarray, Y: np.ndarray) -> None:
            Assigns the training input and known output sets
        train(epochs: int = 100, verbose: bool = True) -> None:
            Trains the Model based on the network and cost functions
            it was initialised with
        predict(X: np.ndarray) -> np.ndarray:
            Predcits the output from the given input
    """

    def __init__(self, network: list, /, *,
                 cost_function: CostFunction = MeanSquaredError(),
                 learning_rate: float = 0.01,
                 threshold: float = 0.,
                 epsilon: float = 1e-4,
                 name: str = 'model'):
        """
        Constructs all necessary attributes for the Model object
        Parameters
        ----------
            network: list
                The network of Layers to be used for the model
            X: numpy.ndarray
                The training set of inputs
            Y: numpy.ndarray
                The training set of known outputs
            cost_function: CostFunctions  = MeanSquaredError()
                The cost function object to be used to calculate errors
                Defaults to Mean Squared Error
            learning_rate: float = 0.01
                Determines how the model performs gradient descent
                Defaults to 0.01
        """
        self.X = None
        self.Y = None
        if not isinstance(cost_function, CostFunction):
            raise InvalidCostFunctionError("Cost Function specified must be a object of derived class of mynn.base.CostFunction")
        if not isinstance(learning_rate, (float, int)):
            raise InvalidLearningRateError()
        if not all([isinstance(_, (Layer, Dense)) for _ in network[::2]]):
            raise InvalidModelNetworkError('Must have alternating Dense and Activation Layers')
        if not all([isinstance(_, (Activation, Softmax)) for _ in network[1::2]]):
            raise InvalidModelNetworkError('Must have alternating Dense and Activation Layers')

        self.network = network
        self.cost_function = cost_function
        self.learning_rate = abs(learning_rate)
        self.threshold = abs(threshold)
        self.epsilon = abs(epsilon)
        self.name = name

    def fit(self, X: np.ndarray, Y: np.ndarray) -> None:
        """
        Fits the model with training sets
        Parameters
        ----------
            X: numpy.ndarray
                The training set of inputs
            Y: numpy.ndarray
                The training set of known outputs
        """
        if not isinstance(X, np.ndarray) or not isinstance(Y, np.ndarray):
            raise TypeError("Training sets must be a numpy.ndarray object")
        self.X = X
        self.Y = Y

    def train(self, epochs: int = 10_000, verbose: bool = True) -> None:
        """
        Trains the fitted Model based on the network and cost functions
        Parameters
        ----------
            epochs: int = 100
                Number of training iterations for the model
                Defaults to 10_000
            verbose: bool = True
                Displays a message showing the error after each
                training iteration
        """
        if self.X is None or self.Y is None:
            raise UnfittedModelError("""Model is not trainable without fitting the training sets.
            Use model.fit(X_train, Y_train) to fit the model with training input and output sets.""")
        prev_error = infinity
        for _ in range(1, epochs + 1):
            error = 0.
            for x, y in zip(self.X, self.Y):
                # Forward Propagation
                output = self.predict(x)

                # Error Calculation
                error += self.cost_function.cost(y, output)

                # Backward Propagation
                grad = self.cost_function.derivative(y, output)
                for layer in reversed(self.network):
                    grad = layer.backward(grad, self.learning_rate)

            error /= len(self.X)
            if error > prev_error:
                self.learning_rate = max(1e-6, self.learning_rate / 10)
            prev_error = error
            if verbose:
                print(f"{_}/{epochs}, {error = : .10f}, learning_rate = {self.learning_rate}")
            if abs(abs(error) - self.threshold) < self.epsilon:
                return

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predicts the output for a given input ``X``
        """
        prediction = X
        for layer in self.network:
            prediction = layer.forward(prediction)

        return prediction

    def save(self, /, filename: str = None) -> None:
        if not filename:
            filename = f'{self.name}.json'
        weights = {}
        biases = {}
        for i, v in enumerate(self.network[::2], 1):
            weights[f'layer{i}'] = v.weights.tolist()
            biases[f'layer{i}'] = v.biases.tolist()
        data = {
            "name": self.name,
            "learning_rate": self.learning_rate,
            "threshold": self.threshold,
            "epsilon": self.epsilon,
            "network": list(map(lambda x: str(x), self.network)),
            "cost_function": str(self.cost_function),
            "weights": weights,
            "biases": biases,
        }

        if not filename.endswith('.json'):
            raise InvalidSavedModelError('Can only save to JSON files!')

        with open(filename, 'w') as f:
            from json import dump
            dump(data, f, indent=2)
            print(f'Saved model at {filename}')

    @staticmethod
    def _get_file_data(filename: str, /, *, w: bool = True) -> dict:
        with open(filename) as f:
            from json import load
            data = load(f)
        weights = data.get('weights', None)
        if not weights:
            if w:
                raise InvalidSavedModelError('No weights found in the file')
            else:
                weights = {}
        biases = data.get('biases', None)
        if not biases:
            if w:
                raise InvalidSavedModelError('No biases found in the file')
            else:
                biases = {}
        cost_function = data.get('cost_function', 'MeanSquaredError')
        _cost_function = cost_exports.get(cost_function, None)
        if not _cost_function:
            raise InvalidSavedModelError(f"We don't recognize that cost function {_cost_function} as of now")
        else:
            cost_function = _cost_function()
        learning_rate = data.get('learning_rate', 0.01)
        threshold = data.get('threshold', 0.)
        epsilon = data.get('epsilon', 1e-4)
        network = data.get('network', None)
        name = data.get('name', 'model')
        return {
            "weights": list(map(lambda x: np.array(weights[x]), weights)),
            "biases": list(map(lambda x: np.array(biases[x]), biases)),
            "cost_function": cost_function,
            "learning_rate": learning_rate,
            "threshold": threshold,
            "epsilon": epsilon,
            "network": network,
            "name": name,
        }

    def load(self, filename: str, /) -> None:
        data = Model._get_file_data(filename)
        weights = data['weights']
        biases = data['biases']
        self.cost_function = data['cost_function']
        self.learning_rate = data['learning_rate']
        self.threshold = data['threshold']
        self.epsilon = data['epsilon']
        self.name = data['name']
        for i, v in enumerate(self.network[::2]):
            v.weights = weights[i]
            v.biases = biases[i]

    @classmethod
    def construct_from_file(cls, filename: str, /):
        data = Model._get_file_data(filename)
        weights = data['weights']
        biases = data['biases']
        cost_function = data['cost_function']
        learning_rate = data['learning_rate']
        threshold = data['threshold']
        epsilon = data['epsilon']
        network = data['network']
        name = data['name']

        if network:
            def get_dense(_) -> Dense:
                nonlocal weights
                nonlocal biases
                i, _ = _
                s, e = _.index('('), _.rindex(')')
                _ = _[s + 1:e]
                _ = _.split(',')
                _ = list(map(int, _))
                _ = Dense(*_)
                _.weights = weights[i]
                _.biases = biases[i]
                return _

            def get_activation(_: str) -> Activation:
                name = _.split()[0].lower()
                for key, value in activators_exports.items():
                    if key.lower() == name:
                        return value()
                else:
                    raise InvalidSavedModelError(f'Invalid Activation Layer {_} !')

            network[::2] = list(map(get_dense, enumerate(network[::2])))
            network[1::2] = list(map(get_activation, network[1::2]))
        else:
            def get_dense(_: str) -> Dense:
                x, y = _.shape
                return Dense(y, x)

            def get_activation(_: str) -> Activation:
                return TanH()

            network[::2] = list(map(get_dense, weights))
            network[1::2] = list(map(get_activation, weights))

        return cls(network,
                   cost_function=cost_function,
                   learning_rate=learning_rate,
                   threshold=threshold,
                   epsilon=epsilon,
                   name=name
                   )

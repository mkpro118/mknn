import numpy as np
try:
    from base import CostFunction
    from utils import export_class
except ImportError:
    from .base import CostFunction
    from .utils import export_class


@export_class
class MeanSquaredError(CostFunction):
    """
    Uses the Mean Squared Error function as the cost
    funtion for the output layer
    MSE is defined as
            (Y - y) ** 2
    f(y) = --------------
              size(Y)
    """

    def cost(self, y_true: np.ndarray,
             y_pred: np.ndarray) -> np.ndarray:
        """
        Calculates the error based on
        the Mean Squared Error cost function
        Parameters
        ----------
        y_true: numpy.ndarray
            The vector of known values in the training set
        y_pred: numpy.ndarray
            The vector of values predicted by the model
        Returns
        -------
            A ``numpy.ndarray`` of the applied cost function vector
        """
        return np.mean(np.power(y_true - y_pred, 2))

    def derivative(self, y_true: np.ndarray,
                   y_pred: np.ndarray) -> np.ndarray:
        """
        Calculates the derivative of error based on
        the Mean Squared Error cost function
        Parameters
        ----------
        y_true: numpy.ndarray
            The vector of known values in the training set
        y_pred: numpy.ndarray
            The vector of values predicted by the model
        Returns
        -------
            A ``numpy.ndarray`` of the applied derivative of cost function vector
        """
        return 2. * (y_pred - y_true) / np.size(y_true)


@export_class
class BinaryCrossEntropy(CostFunction):

    def cost(self, y_true: np.ndarray,
             y_pred: np.ndarray) -> np.ndarray:
        """
        Calculates the error based on the
        Binary Cross Entropy cost function
        Parameters
        ----------
        y_true: numpy.ndarray
            The vector of known values in the training set
        y_pred: numpy.ndarray
            The vector of values predicted by the model
        Returns
        -------
            A ``numpy.ndarray`` of the applied cost function vector
        """
        return np.mean(-y_true * np.log(y_pred) - (1 - y_true) * np.log(1 - y_pred))

    def derivative(self, y_true: np.ndarray,
                   y_pred: np.ndarray) -> np.ndarray:
        """
        Calculates the derivative of error based on the
        Binary Cross Entropy cost function
        Parameters
        ----------
        y_true: numpy.ndarray
            The vector of known values in the training set
        y_pred: numpy.ndarray
            The vector of values predicted by the model
        Returns
        -------
            A ``numpy.ndarray`` of the applied derivative of cost function vector
        """
        return ((1 - y_true) / (1 - y_pred) - y_true / y_pred) / np.size(y_true)


@export_class
class CategoricalCrossEntropy(CostFunction):

    def cost(self, y_true: np.ndarray,
             y_pred: np.ndarray) -> np.ndarray:
        """
        Calculates the error based on the
        Categorical Cross Entropy cost function
        Parameters
        ----------
        y_true: numpy.ndarray
            The vector of known values in the training set
        y_pred: numpy.ndarray
            The vector of values predicted by the model
        Returns
        -------
            A ``numpy.ndarray`` of the applied cost function vector
        """
        return np.mean(-y_true * np.log(y_pred + 1e-15))

    def derivative(self, y_true: np.ndarray,
                   y_pred: np.ndarray) -> np.ndarray:
        """
        Calculates the derivative of error based on the
        Categorical Cross Entropy cost function
        Parameters
        ----------
        y_true: numpy.ndarray
            The vector of known values in the training set
        y_pred: numpy.ndarray
            The vector of values predicted by the model
        Returns
        -------
            A ``numpy.ndarray`` of the applied derivative of cost function vector
        """
        return - y_true / y_pred

import numpy as np
from colors import *

class ActivationFunction:
    def __init__(self, function):
        self.function = function

    def base_function(self, x):
        match self.function:
            case 'sigmoid':
                return 1 / (1 + np.exp(-x))
            case 'relu':
                return (abs(x) + x) / 2
            case 'tanh':
                return np.tanh(x)
            case _:
                raise ValueError(f"{BHRED}Unknown activation function: {RED}{self.function}{BHRED}.{RESET}")

    def prime_function(self, x):
        match self.function:
            case 'sigmoid':
                return x * (1 - x)
            case 'relu':
                return 1. * (x > 0)
            case 'tanh':
                return 1 - x**2
            case _:
                raise ValueError(f"{BHRED}Unknown activation function: {RED}{self.function}{BHRED}.{RESET}")

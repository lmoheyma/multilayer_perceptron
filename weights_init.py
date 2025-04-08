import numpy as np
from colors import *

class WeightInitializer:
    def __init__(self, method):
        self.method = method

    def initialize(self, fan_in, fan_out):
        match self.method:
            case 'he':
                return np.random.randn(fan_in, fan_out) * np.sqrt(2. / fan_in)
            case 'xavier':
                return np.random.randn(fan_in, fan_out) * np.sqrt(2. / (fan_in + fan_out))
            case 'uniform':
                return np.random.uniform(-0.1, 0.1, size=(fan_in, fan_out))
            case _:
                raise ValueError(f"{BHRED}Unknown weight initialization method: {RED}{self.method}{BHRED}.{RESET}")

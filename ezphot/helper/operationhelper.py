#%%
from numba import jit
import inspect
import numpy as np

class OperationHelper:
    def __init__(self):
        pass

    def __repr__(self):
        methods = []
        for name, attr in self.__class__.__dict__.items():
            if isinstance(attr, staticmethod) and not name.startswith("_"):
                func = attr.__func__
                sig = inspect.signature(func)
                doc = inspect.getdoc(func) or "No doc"
                methods.append(f"  • {name}{sig}: {doc}")
        return "[Supported Operations]\n" + "\n".join(methods)

    @staticmethod
    @jit(nopython=True)
    def sum(a, b):
        """Return the sum of a and b."""
        return a + b

    @staticmethod
    @jit(nopython=True)
    def multiply(a, b):
        """Return the product of a and b."""
        return a * b

    @staticmethod
    @jit(nopython=True)
    def divide(a, b):
        """Return a divided by b."""
        return a / b

    @staticmethod
    @jit(nopython=True)
    def subtract(a, b):
        """Return a minus b."""
        return a - b
    
    @staticmethod
    @jit(nopython=True)
    def power(a, b):
        """Return a raised to the power of b."""
        return np.power(a, b)
    
    @staticmethod
    @jit(nopython=True)
    def sqrt(a):
        """Return the square root of a."""
        return np.sqrt(a)

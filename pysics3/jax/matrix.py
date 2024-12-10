from typing import Sequence
from functools import cached_property
import jax.numpy as jnp


class _Matrix:
    def __new__(cls, arr: Sequence[Sequence[float]]):
        obj = super().__new__(cls)
        obj.array = jnp.asarray(arr)
        return obj
    
    def __init__(self, array: Sequence[Sequence[float]]):
        self.array = jnp.asarray(array)
    
    @cached_property
    def shape(self) -> tuple[int, ...]:
        return self.array.shape
    
    @property
    def vec(self) -> jnp.ndarray:
        return self.array.flatten()
    
    def __getitem__(self, key):
        return self.array[key]
    
    def __array__(self, dtype=None):
        return jnp.asarray(self.array, dtype=dtype)
    
    def __add__(self, other):
        if isinstance(other, _Matrix):
            if self.shape == other.shape:
                return Matrix(self.array + other.array)
            else:
                raise ValueError("Cannot add matrices of different shapes")
        else:
            raise TypeError("Cannot add Matrix and non-Matrix objects")
    
    def __sub__(self, other):
        if isinstance(other, _Matrix):
            if self.shape == other.shape:
                return Matrix(self.array - other.array)
            else:
                raise ValueError("Cannot subtract matrices of different shapes")
        else:
            raise TypeError("Cannot subtract Matrix and non-Matrix objects")
    
    def __mul__(self, other):
        if isinstance(other, _Matrix):
            if self.shape[1] == other.shape[0]:
                return Matrix(self.array @ other.array)
            else:
                raise ValueError("Cannot multiply matrices of different shapes")
        elif isinstance(other, (int, float)):
            return Matrix(self.array * other)
        else:
            raise TypeError("Cannot multiply Matrix and non-Matrix objects")
    
    def __rmul__(self, other):
        if isinstance(other, (int, float)):
            return Matrix(self.array * other)
        else:
            raise TypeError("Cannot multiply Matrix and non-Matrix objects")


class Matrix(_Matrix):
    def __new__(cls, arr: Sequence[Sequence[float]]):
        array = jnp.asarray(arr)
        if array.shape == (4, 4):
            return super().__new__(cls, array)
        elif array.shape == (4, 1):
            x, y, z, w = array.T[0]
            if array[3][0] == 0:
                return Vector([x, y, z])
            else:
                return Point([x / w, y / w, z / w])
        elif array.shape == (1, 1):
            return float(array[0][0])
        else:
            raise ValueError("Invalid shape for Matrix")

    def __init__(self, array: Sequence[Sequence[float]]):
        super().__init__(array)
    
    @cached_property
    def T(self) -> "Matrix":
        return Matrix(self.array.T)
    
    @cached_property
    def inv(self) -> "Matrix":
        return Matrix(jnp.linalg.inv(self.array))


class Vector(_Matrix):
    def __new__(cls, arr: Sequence[float]):
        array = jnp.asarray(arr)
        if array.shape == (4, 1):
            return super().__new__(cls, array)
        elif array.shape == (3, ) or array.shape == (4, ):
            return super().__new__(cls, [[array[0]], [array[1]], [array[2]], [0]])
        else:
            raise ValueError("Invalid shape for Vector")

    def __init__(self, array: Sequence[float]):
        super().__init__([[array[0]], [array[1]], [array[2]], [0]])
        self.x = float(self.vec[0])
        self.y = float(self.vec[1])
        self.z = float(self.vec[2])
    
    @property
    def vec(self) -> jnp.ndarray:
        return super().vec[:3]
    
    def __getitem__(self, key):
        return self.array[key][0]
    
    @cached_property
    def norm(self) -> float:
        return jnp.linalg.norm(self.vec)
    
    def normalize(self) -> "Vector":
        return Vector(self.vec / self.norm)
    
    
class Point(_Matrix):
    def __new__(cls, arr):
        array = jnp.asarray(arr)
        if array.shape == (4, 1):
            return super().__new__(cls, array)
        elif array.shape == (3, ):
            x, y, z = array
            return super().__new__(cls, [[x], [y], [z], [1]])
        elif array.shape == (4, ):
            x, y, z, w = array
            return super().__new__(cls, [[x / w], [y / w], [z / w], [1]])
        else:
            raise ValueError("Invalid shape for Point")

    def __init__(self, array: Sequence[float]):
        super().__init__([[array[0]], [array[1]], [array[2]], [1]])
        self.x = float(self.vec[0])
        self.y = float(self.vec[1])
        self.z = float(self.vec[2])
    
    @property
    def vec(self) -> jnp.ndarray:
        return super().vec[:3]
    
    def __getitem__(self, key):
        return self.array[key][0]


def dot(a: Vector, b: Vector) -> float:
    return float(jnp.dot(a.vec, b.vec))


def cross(a: Vector, b: Vector) -> Vector:
    return Vector(jnp.cross(a.vec, b.vec))

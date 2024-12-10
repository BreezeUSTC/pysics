from typing import Sequence

from pysics3.base import BaseObject
from pysics3.render.surface import Surface
from pysics3.constants import I
from pysics3.jax import Matrix, Vector, Point


class Model(BaseObject):

    def __init__(self, surfaces: Sequence[Surface]):
        self.surfaces = surfaces
        self.matrix = I

    @property
    def x(self) -> Matrix:
        return Vector((self.matrix[0][0], self.matrix[1][0], self.matrix[2][0]))
    
    @property
    def y(self) -> Matrix:
        return Vector((self.matrix[0][1], self.matrix[1][1], self.matrix[2][1]))
    
    @property
    def z(self) -> Matrix:
        return Vector((self.matrix[0][2], self.matrix[1][2], self.matrix[2][2]))
    
    @property
    def pos(self) -> Matrix:
        return Point((self.matrix[0][3], self.matrix[1][3], self.matrix[2][3]))
        
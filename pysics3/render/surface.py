from typing import Sequence

from pysics3.base import BaseObject, Surface2D
from pysics3.render.texture import Texture
# from pysics3.core.camera import Camera
from pysics3.jax import Matrix, Point, Vector, cross
from pysics3.constants import MTrs, MRotX, MRotY, MRotZ, MRot


class Surface(BaseObject):

    def __init__(self, texture: Texture, *points: tuple[Point | Sequence[float]]):
        if isinstance(points[0], Point):
            self.points = points
        else:
            self.points = tuple(Point(p) for p in points)
        
        if len(points) != 3:
            raise ValueError("Surface must have 3 points")

        self.texture = texture

    @property
    def verges(self) -> tuple[Vector, Vector, Vector]:
        return (self.points[1] - self.points[0], self.points[2] - self.points[1], self.points[0] - self.points[2])
        
    @property
    def normal(self) -> Matrix:
        return cross(*self.verges[:2]).normalize()
    
    def translate(self, vec: Matrix) -> None:
        for i in range(3):
            self.points[i] = MTrs(vec) * self.points[i]
    
    def rotate_x(self, pos: Matrix, angle: float) -> None:
        for i in range(3):
            self.points[i] = MRotX(pos, angle) * self.points[i]

    def rotate_y(self, pos: Matrix, angle: float) -> None:
        for i in range(3):
            self.points[i] = MRotY(pos, angle) * self.points[i]

    def rotate_z(self, pos: Matrix, angle: float) -> None:
        for i in range(3):
            self.points[i] = MRotZ(pos, angle) * self.points[i]

    def rotate(self, axis: Matrix, pos: Matrix, angle: float) -> None:
        for i in range(3):
            self.points[i] = MRot(axis, pos, angle) * self.points[i]
    
    def project(self, camera) -> Surface2D:
        return tuple(camera.project(point) for point in self.points)

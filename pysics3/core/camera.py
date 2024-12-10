from functools import cached_property
from pygame.surfarray import blit_array

from pysics3.base import BaseObject, Point2D
from pysics3.render.raster import raster
# from pysics3.core.world import World
from pysics3.constants import I, MTrs, MRotX, MRotY, MRotZ, MRot, MPrj
from pysics3.jax import Matrix, Vector, Point, pi


class Camera(BaseObject):

    def __init__(self, world, fov: float = pi / 2, near: float = 0.5, far: float = 500.0):
        self.world = world
        self.screen = world.screen
        self.width= self.world.width
        self.height = self.world.height

        self.fov = fov
        self.near = near
        self.far = far
        self.aspect = self.width / self.height

        self.matrix = I
    
    @cached_property
    def MPrj(self) -> Matrix:
        return MPrj(self.fov, self.width, self.height, self.near, self.far)
    
    @property
    def x(self) -> Vector:
        return Vector(self.matrix.T[0])
    
    @property
    def y(self) -> Vector:
        return Vector(self.matrix.T[1])
    
    @property
    def z(self) -> Vector:
        return Vector(self.matrix.T[2])
    
    @property
    def pos(self) -> Point:
        return Point(self.matrix.T[3])

    def translate(self, vec: Vector) -> None:
        self.matrix = MTrs(vec) * self.matrix

    def rotate_x(self, angle: float) -> None:
        self.matrix = MRotX(self.pos, angle) * self.matrix
    
    def rotate_y(self, angle: float) -> None:
        self.matrix = MRotY(self.pos, angle) * self.matrix

    def rotate_z(self, angle: float) -> None:
        self.matrix = MRotZ(self.pos, angle) * self.matrix

    def rotate(self, axis: Matrix, angle: float) -> None:
        self.matrix = MRot(axis, self.pos, angle) * self.matrix
    
    def project(self, point: Point) -> Point2D:
        return tuple((self.MPrj * self.matrix.inv * point).vec)

    def display(self) -> None:
        blit_array(self.screen, raster(self, self.world.objects, self.world.bgcolor))


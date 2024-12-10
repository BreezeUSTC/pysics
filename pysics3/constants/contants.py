from pysics3.jax.matrix import Matrix, Point, Vector
from typing import Final


origin: Final = Point((0, 0, 0))
x_axis: Final = Vector((1, 0, 0))
y_axis: Final = Vector((0, 1, 0))
z_axis: Final = Vector((0, 0, 1))

I: Final = Matrix([
    [1, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 0, 1, 0],
    [0, 0, 0, 1]])

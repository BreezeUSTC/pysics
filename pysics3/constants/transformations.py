from pysics3.jax.matrix import Matrix, Vector, Point
import jax.numpy as jnp


def matrix_translation(vec: Vector) -> Matrix:
    return Matrix([
        [1, 0, 0, vec.x], 
        [0, 1, 0, vec.y], 
        [0, 0, 1, vec.z],
        [0, 0, 0, 1]])

def matrix_scale(x: float, y: float, z: float) -> Matrix:
    return Matrix([
        [x, 0, 0, 0],
        [0, y, 0, 0],
        [0, 0, z, 0],
        [0, 0, 0, 1]])

def matrix_rotation_x(pos: Point, angle: float) -> Matrix:
    T = Matrix([
        [1, 0, 0, -pos.x],
        [0, 1, 0, -pos.y],
        [0, 0, 1, -pos.z],
        [0, 0, 0, 1]])
    R = Matrix([
        [1, 0, 0, 0],
        [0, jnp.cos(angle), -jnp.sin(angle), 0],
        [0, jnp.sin(angle), jnp.cos(angle), 0],
        [0, 0, 0, 1]])
    return T.inv * R * T

def matrix_rotation_y(pos: Point, angle: float) -> Matrix:
    T = Matrix([
        [1, 0, 0, -pos.x],
        [0, 1, 0, -pos.y],
        [0, 0, 1, -pos.z],
        [0, 0, 0, 1]])
    R = Matrix([
        [jnp.cos(angle), 0, jnp.sin(angle), 0],
        [0, 1, 0, 0],
        [-jnp.sin(angle), 0, jnp.cos(angle), 0],
        [0, 0, 0, 1]])
    return T.inv * R * T

def matrix_rotation_z(pos: Point, angle: float) -> Matrix:
    T = Matrix([
        [1, 0, 0, -pos.x],
        [0, 1, 0, -pos.y],
        [0, 0, 1, -pos.z],
        [0, 0, 0, 1]])
    R = Matrix([
        [jnp.cos(angle), -jnp.sin(angle), 0, 0],
        [jnp.sin(angle), jnp.cos(angle), 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]])
    return T.inv * R * T

def matrix_rotation_axis(axis: Vector, pos: Point, angle: float) -> Matrix:
    axis = axis.normalize()
    c = jnp.cos(angle)
    s = jnp.sin(angle)

    T = Matrix([
        [1, 0, 0, -pos.x],
        [0, 1, 0, -pos.y],
        [0, 0, 1, -pos.z],
        [0, 0, 0, 1]])
    R = Matrix([
        [axis.x * axis.x * (1 - c) + c, axis.x * axis.y * (1 - c) - axis.z * s, axis.x * axis.z * (1 - c) + axis.y * s, 0],
        [axis.y * axis.x * (1 - c) + axis.z * s, axis.y * axis.y * (1 - c) + c, axis.y * axis.z * (1 - c) - axis.x * s, 0],
        [axis.x * axis.z * (1 - c) - axis.y * s, axis.y * axis.z * (1 - c) + axis.x * s, axis.z * axis.z * (1 - c) + c, 0],
        [0, 0, 0, 1]])
    return T.inv * R * T

def matrix_projection_perspective(fov: float, width: int, height: int, near: float, far: float) -> Matrix:
    f = 1 / jnp.tan(fov / 2)
    aspect = width / height
    return Matrix([
        [f / aspect, 0, 0, 0],
        [0, f, 0, 0],
        [0, 0, (far + near) / (near - far), (2 * far * near) / (near - far)],
        [0, 0, -1, 0]])

def matrix_viewport(width: int, height: int) -> Matrix:
    return Matrix([
        [width / 2, 0, 0, width / 2],
        [0, -height / 2, 0, height / 2],
        [0, 0, 1, 0],
        [0, 0, 0, 1]])

def matrix_projection(fov: float, width: int, height: int, near: float, far: float) -> Matrix:
    return matrix_viewport(width, height) * matrix_projection_perspective(fov, width, height, near, far)

def MTrs(vec: Matrix) -> Matrix:
    return matrix_translation(vec)

def MScl(x: float, y: float, z: float) -> Matrix:
    return matrix_scale(x, y, z)

def MRotX(pos: Matrix, angle: float) -> Matrix:
    return matrix_rotation_x(pos, angle)

def MRotY(pos: Matrix, angle: float) -> Matrix:
    return matrix_rotation_y(pos, angle)

def MRotZ(pos: Matrix, angle: float) -> Matrix:
    return matrix_rotation_z(pos, angle)

def MRot(axis: Matrix, pos: Matrix, angle: float) -> Matrix:
    return matrix_rotation_axis(axis, pos, angle)

def MPrjPersp(fov: float, width: int, height: int, near: float, far: float) -> Matrix:
    return matrix_projection_perspective(fov, width, height, near, far)

def MPrj(fov: float, width: int, height: int, near: float, far: float) -> Matrix:
    return matrix_projection(fov, width, height, near, far)

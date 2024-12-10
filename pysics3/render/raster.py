from jax import vmap, tree_map
import jax.numpy as jnp

from pysics3.render import Surface
# from pysics3.core import Camera


def raster(cam, surfaces: list[Surface], bgcolor: tuple[int, int, int] = (255, 255, 255)) -> jnp.ndarray:
    surface_and_texture = [(surface.project(cam), surface.texture) for surface in surfaces]

    def operate_pixel(x: int, y: int) -> tuple[int, int, int]:
        near_surface = ((0, 0, 0), (0, 0, 0), (0, 0, 0))
        near_texture = bgcolor
        near_z = jnp.inf
        
        for surface, texture in surface_and_texture:
            x1, y1, z1 = surface[0][:3]
            x2, y2, z2 = surface[1][:3]
            x3, y3, z3 = surface[2][:3]

            w = ((x2 - x3) * (y - y3) - (y2 - y3) * (x - x3)) / ((x2 - x3) * (y1 - y3) - (y2 - y3) * (x1 - x3))
            u = ((x3 - x1) * (y - y1) - (y3 - y1) * (x - x1)) / ((x3 - x1) * (y2 - y1) - (y3 - y1) * (x2 - x1))
            v = ((x1 - x2) * (y - y2) - (y1 - y2) * (x - x2)) / ((x1 - x2) * (y3 - y2) - (y1 - y2) * (x3 - x2))

            z = w * z1 + u * z2 + v * z3
            color = texture.get_pixel_normalized_blended(x, y)

            condition = jnp.logical_and(jnp.logical_not(jnp.logical_or(jnp.logical_or(w < 0, u < 0), v < 0)), jnp.logical_and(0 <= z, z < near_z))

            near_z = jnp.where(condition, z, near_z)
            near_surface = tree_map(lambda a, b: jnp.where(condition, a, b), surface, near_surface)
            near_texture = tree_map(lambda a, b: jnp.where(condition, a, b), color, near_texture)

        return jnp.where(jnp.isnan(near_z), jnp.asarray(bgcolor), jnp.asarray(near_texture))

    x_coords, y_coords = jnp.meshgrid(jnp.arange(cam.width), jnp.arange(cam.height))
    result = vmap(vmap(operate_pixel, in_axes=0), in_axes=0)(x_coords, y_coords)
    result = jnp.transpose(result, (1, 0, 2))
    return result


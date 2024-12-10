from PIL import Image
from os import PathLike
from jax import jit

from pysics3.base import BaseObject


class Texture(BaseObject):

    path: str | PathLike
    image: Image.Image
    width: int
    height: int
    size: tuple[int, int]
    

    def get_pixel(self, *args) -> tuple[int, int, int]:
        ...
    
    def get_pixel_normalized(self, *args) -> tuple[int, int, int]:
        ...
    
    def get_pixel_blended(self, *args) -> tuple[int, int, int]:
        ...
    
    def get_pixel_normalized_blended(self, *args) -> tuple[int, int, int]:
        ...


class ImageTexture(Texture):

    def __init__(self, path: str | PathLike):
        self.path = path
        self.image = Image.open(path)
        self.width, self.height = self.image.size
        self.pixels = self.image.load()

    def get_pixel(self, x: int, y: int) -> tuple[int, int, int]:
        return self.pixels[x, y]
    
    def get_pixel_normalized(self, u: float, v: float) -> tuple[int, int, int]:
        return self.get_pixel(int(u * self.width), int(v * self.height))
    
    @jit
    def get_pixel_blended(self, x: int, y: int) -> tuple[int, int, int]:
        x1, y1 = x, y
        x2, y2 = x1 + 1, y1 + 1

        if x < 0:
            x1 = 0
            x2 = 1
        elif x + 1 >= self.width:
            x2 = self.width - 1
            x1 = self.width - 2
        else:
            x1, x2 = x, x + 1
        if y < 0:
            y1 = 0
            y2 = 1
        elif y + 1 >= self.height:
            y2 = self.height - 1
            y1 = self.height - 2
        else:
            y1, y2 = y, y + 1

        c1 = self.get_pixel(x1, y1)
        c2 = self.get_pixel(x1, y2)
        c3 = self.get_pixel(x2, y1)
        c4 = self.get_pixel(x2, y2)

        dx1 = x - x1
        dx2 = x2 - x
        dy1 = y - y1
        dy2 = y2 - y

        return tuple(int(c1[i] * dx2 * dy2 + c2[i] * dx1 * dy2 + c3[i] * dx2 * dy1 + c4[i] * dx1 * dy1) for i in range(3))

    def get_pixel_normalized_blended(self, u: float, v: float) -> tuple[int, int, int]:
        return self.get_pixel_blended(int(u * self.width), int(v * self.height))


class ColorTexture(Texture):

    def __init__(self, color: tuple[int, int, int]):
        self.color = color

    def get_pixel(self, x, y) -> tuple[int, int, int]:
        return self.color

    def get_pixel_normalized(self, x, y) -> tuple[int, int, int]:
        return self.color

    def get_pixel_blended(self, x, y) -> tuple[int, int, int]:
        return self.color

    def get_pixel_normalized_blended(self, x, y) -> tuple[int, int, int]:
        return self.color

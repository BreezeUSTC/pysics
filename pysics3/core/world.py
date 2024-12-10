import pygame

from pysics3.base import BaseObject
from pysics3.render.surface import Surface
from pysics3.render.model import Model


class World(BaseObject):

    def __init__(self, screen: pygame.Surface, bgcolor: tuple[int, int, int] = (255, 255, 255)):
        self.screen = screen
        self.width, self.height = screen.get_size()
        self.bgcolor = bgcolor

        self.clock = pygame.time.Clock()

        self.objects = list()

    def update(self, fps: int = 120):
        self.clock.tick(fps)
        pygame.display.update()

    def add_obj(self, *obj: Surface | Model):
        for each in obj:
            if isinstance(each, Surface):
                self.objects.append(each)
            elif isinstance(each, Model):
                self.objects.extend(each.surfaces)
            else:
                raise TypeError(f"Object of type {type(each)} is not supported.")
    
    def del_obj(self, *obj: Surface | Model):
        for each in obj:
            if isinstance(each, Surface):
                self.objects.remove(each)
            elif isinstance(each, Model):
                for surface in each.surfaces:
                    self.objects.remove(surface)
            else:
                raise TypeError(f"Object of type {type(each)} is not supported.")
    
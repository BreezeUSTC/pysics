from pysics3 import *
import pygame
import os
import shutil


def message(text, pos, size, color=(0, 0, 0), anchor_x="w", anchor_y="n"):
    text_font = pygame.font.SysFont("FreeSansBold", size)
    text_surface = text_font.render(text, True, color)
    text_rect = text_surface.get_rect()
    if anchor_x == "w":
        text_rect.left = pos[0]
    elif anchor_x == "c":
        text_rect.centerx = pos[0]
    elif anchor_x == "e":
        text_rect.right = pos[0]
    else:
        raise ValueError("Param 'anchor_x' should be 'w', 'c' or 'e'.")
    if anchor_y == "n":
        text_rect.top = pos[1]
    elif anchor_y == "c":
        text_rect.centery = pos[1]
    elif anchor_y == "s":
        text_rect.bottom = pos[1]
    else:
        raise ValueError("Param 'anchor_y' should be 'n', 'c' or 's'.")
    screen.blit(text_surface, text_rect)


def main():
    if not os.path.exists("C:\\images"):
        os.mkdir("C:\\images")

    record = input("Do you want to record the screen?(Y/N)")
    tick = 0

    global running, paused
    while running:
        keys = pygame.key.get_pressed()
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                if record == "Y" or record == "y":
                    import video
                    video.main(30)
                shutil.rmtree("C:\\images")
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    paused = not paused
            if event.type == pygame.MOUSEMOTION:
                if not paused:
                    camera.rotate(y_axis, -event.rel[0] / 150)
                    camera.rotate(camera.x, -event.rel[1] / 150)
        
        if paused:
            pygame.mouse.set_visible(True)
            message("PAUSED", (width / 2, height / 2), 70, anchor_x="c", anchor_y="c")
        else:
            pygame.mouse.set_visible(False)
            pygame.mouse.set_pos((width / 2, height / 2))

            if keys[pygame.K_UP] or keys[pygame.K_w]:
                camera.translate(-0.5 * cross(camera.x, y_axis))
            if keys[pygame.K_DOWN] or keys[pygame.K_s]:
                camera.translate(0.5 * cross(camera.x, y_axis))
            if keys[pygame.K_LEFT] or keys[pygame.K_a]:
                camera.translate(-0.5 * camera.x)
            if keys[pygame.K_RIGHT] or keys[pygame.K_d]:
                camera.translate(0.5 * camera.x)
            if keys[pygame.K_LSHIFT]:
                camera.translate(-0.5 * y_axis)
            if keys[pygame.K_SPACE]:
                camera.translate(0.5 * y_axis)

            camera.display()

            message(f"FPS: {round(world.clock.get_fps(), 1)}", (10, 10), 30)
            message(f"Pos: {tuple(round(n, 2) for n in (camera.pos.x, camera.pos.y, camera.pos.z))}", (10, 30), 30)
            pygame.draw.line(screen, (0, 0, 0), (width / 2 - 10, height / 2), (width / 2 + 10, height / 2), 2)
            pygame.draw.line(screen, (0, 0, 0), (width / 2, height / 2 - 10), (width / 2, height / 2 + 10), 2)
        
        if record == "Y" or record == "y" and not paused:
            pygame.image.save(screen, f"C:\\images\\{tick}.png")
            tick += 1
        
        world.update()         


if __name__ == "__main__":
    pygame.init()
    size = width, height = 400, 300
    screen = pygame.display.set_mode(size)
    pygame.display.set_caption("pysics-3.2 test")

    world = World(screen)
    camera = Camera(world, fov=pi / 4)
    camera.translate(z_axis * 5)

    p1 = Point((1, 1, 1))
    p2 = Point((-1, 1, 1))
    p3 = Point((-1, 1, -1))
    p4 = Point((1, 1, -1))

    p5 = Point((1, -1, 1))
    p6 = Point((-1, -1, 1))
    p7 = Point((-1, -1, -1))
    p8 = Point((1, -1, -1))

    texture_white = ColorTexture((245, 245, 245))
    texture_yellow = ColorTexture((255, 255, 0))
    texture_red = ColorTexture((255, 0, 0))
    texture_orange = ColorTexture((255, 128, 0))
    texture_blue = ColorTexture((0, 0, 255))
    texture_green = ColorTexture((0, 255, 0))

    world.add_obj(Surface(texture_white, p5, p6, p7))
    world.add_obj(Surface(texture_white, p7, p8, p5))

    world.add_obj(Surface(texture_yellow, p1, p2, p3))
    world.add_obj(Surface(texture_yellow, p3, p4, p1))

    world.add_obj(Surface(texture_red, p1, p5, p8))
    world.add_obj(Surface(texture_red, p8, p4, p1))

    world.add_obj(Surface(texture_orange, p2, p3, p7))
    world.add_obj(Surface(texture_orange, p7, p6, p2))

    world.add_obj(Surface(texture_green, p3, p4, p8))
    world.add_obj(Surface(texture_green, p8, p7, p3))

    world.add_obj(Surface(texture_blue, p2, p6, p5))
    world.add_obj(Surface(texture_blue, p5, p1, p2))

    # texture = ImageTexture("texture.png")
    # world.add_obj(Surface(texture, p1, p2, p3))

    running = True
    paused = False
    main()

import sys
import pygame


class GUI:
    def __init__(self, size):
        self.Particle = 60
        self.size = size
        pygame.init()
        self.screen = pygame.display.set_mode((self.Particle * self.size[0], self.Particle * self.size[1]))
        self.screen.fill((0, 0, 0))
        pygame.display.set_caption('Snake')
        pygame.PYGAME_HIDE_SUPPORT_PROMPT = 1

    def draw(self, pos, color):
        Cords = [pos[0] * self.Particle, pos[1] * self.Particle]
        pygame.draw.rect(self.screen, color, (Cords[0], Cords[1], self.Particle, self.Particle), 0)

    def update_GUI(self, ground):
        if self.size != ground.shape:
            self.size = ground.shape
            del self.screen
            self.screen = pygame.display.set_mode((self.Particle * self.size[1], self.Particle * self.size[0]))
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
        for row in range(self.size[0]):
            for column in range(self.size[1]):
                # last part of the tail - red
                if ground[row][column] == -1:
                    self.draw([column, row], (0, 200, 75))
                    continue
                # apple - red
                if ground[row][column] == -2:
                    self.draw([column, row], (255, 0, 0))
                    continue
                # player 1
                    # body / tail of snake
                if ground[row][column] == 1:
                    self.draw([column, row], (0, 200, 75))
                    continue
                    # head of snake
                if ground[row][column] == 2:
                    self.draw([column, row], (0, 100, 60))
                    continue
        pygame.display.update()

    def reset_GUI(self):
        self.screen.fill((0, 0, 0))

    def show(self, ground):
        self.reset_GUI()
        self.update_GUI(ground)

from os import environ

environ['PYGAME_HIDE_SUPPORT_PROMPT'] = '1'
import pygame


class GUI:
    def __init__(self, size):
        self.particle = 60
        self.size = size
        pygame.init()
        self.screen = pygame.display.set_mode((self.particle * self.size[0], self.particle * self.size[1]))
        self.screen.fill((0, 0, 0))
        pygame.display.set_caption('Snake')
        pygame.PYGAME_HIDE_SUPPORT_PROMPT = 1

    """
    This method is responsible for drawing the single squares of the gui.
    @:param pos: Position of the square.
    @:param color: RGB tuple which defines the color.
    """

    def draw(self, pos, color):
        Cords = [pos[0] * self.particle, pos[1] * self.particle]
        pygame.draw.rect(self.screen, color, (Cords[0], Cords[1], self.particle, self.particle), 0)

    """
    This method is updates every single square of the gui. Hence, it is responsible for updating the gui.
    @:param ground: Playground.
    """

    def update_GUI(self, ground):
        self.reset_GUI()
        if self.size != ground.shape:
            self.size = ground.shape
            del self.screen
            self.screen = pygame.display.set_mode((self.particle * self.size[1], self.particle * self.size[0]))
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
        for row in range(self.size[0]):
            for column in range(self.size[1]):
                # last part of the tail - red
                if ground[row][column] == -1:
                    self.draw([column, row], (0, 225, 120))
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
                if ground[row][column] == 3:
                    self.draw([column, row], (0, 0, 255))
        pygame.display.update()

    """
    This method resets the gui.
    """

    def reset_GUI(self):
        self.screen.fill((0, 0, 0))

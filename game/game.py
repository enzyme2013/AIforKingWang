"""Minigame example using some thorpy elements and coupling them with user input."""

import pygame, random
import thorpy as tp

from game.gameconsts import SCREEN_WIDTH, SCREEN_HEIGHT

clock = pygame.time.Clock()


class Game:
    state = None
    screen = None

    def __int__(self):
        state = None
        self.__initGames()

    def __initGames(self):
        pygame.init()
        screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))

    def refresh_game(self):
        pass

    def start(self):
        playing = True
        while playing:
            clock.tick(60)
            events = pygame.event.get()
            mouse_rel = pygame.mouse.get_rel()
            for e in events:
                if e.type == pygame.QUIT:
                    playing = False
                else:
                    ...  # do your stuff with events
            self.refresh_game()
            pygame.display.flip()
        pygame.quit()

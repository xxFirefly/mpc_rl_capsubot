from sys import exit
from typing import List, Tuple, Optional

import pygame


class Renderer:

    GROUND_LEVEL = 200
    SCREEN_SIZE = (1280, 400)
    WORLD_WIDTH = 2.0
    SCALE = SCREEN_SIZE[0] / WORLD_WIDTH

    CAPSULE_SIZE = (100, 30)
    INNER_BODY_SIZE = (50, 30)

    BLACK = (0, 0, 0)
    WHITE = (255, 255, 255)
    BROWN = (139, 69, 19)
    GREEN = (175, 215, 70)

    def __init__(
        self,
        name: str,
        render_target_region: bool,
    ):
        self.render_target_region = render_target_region
        self.goal_point = None
        self.tolerance = None

        pygame.init()
        pygame.display.set_caption(name)
        self.clock = pygame.time.Clock()
        self.screen = pygame.display.set_mode(self.SCREEN_SIZE)

        # capsule body
        self.capsule_surf = pygame.Surface(self.CAPSULE_SIZE)
        self.capsule_rect = self.capsule_surf.get_rect(
            midbottom=(self.SCREEN_SIZE[0] / 2, self.GROUND_LEVEL)
        )

        # inner body
        self.inner_surf = pygame.Surface(self.INNER_BODY_SIZE)
        self.inner_surf.fill(self.BROWN)
        self.inner_rect = self.inner_surf.get_rect(
            midbottom=(
                self.capsule_rect.midtop[0],
                self.GROUND_LEVEL - self.CAPSULE_SIZE[1],
            )
        )

        # text
        self.font = pygame.font.SysFont("arial", 22)

    def render(self, time: float, state: List[float], fps: int = 60, goal_point: Optional[float] = None) -> None:
        self._pose_update(state)
        self._view_update(state, goal_point, time)
        self.clock.tick(fps)

    def _draw_ground(self) -> None:
        pygame.draw.line(
            self.screen,
            self.GREEN,
            (0, self.GROUND_LEVEL + 4),
            (self.SCREEN_SIZE[0], self.GROUND_LEVEL + 4),
            width=10,
        )

    def _draw_target_region(self) -> None:
        pygame.draw.line(
            self.screen,
            self.GREEN,
            (
                self.SCREEN_SIZE[0] / 2
                + (self.goal_point - self.tolerance / 2) * self.SCALE,
                self.GROUND_LEVEL - 100,
            ),
            (
                self.SCREEN_SIZE[0] / 2
                + (self.goal_point - self.tolerance / 2) * self.SCALE,
                self.GROUND_LEVEL + 100,
            ),
        )

        pygame.draw.line(
            self.screen,
            self.GREEN,
            (
                self.SCREEN_SIZE[0] / 2 + self.goal_point * self.SCALE,
                self.GROUND_LEVEL - 100,
            ),
            (
                self.SCREEN_SIZE[0] / 2 + self.goal_point * self.SCALE,
                self.GROUND_LEVEL + 100,
            ),
        )

        pygame.draw.line(
            self.screen,
            self.GREEN,
            (
                self.SCREEN_SIZE[0] / 2
                + (self.goal_point + self.tolerance / 2) * self.SCALE,
                self.GROUND_LEVEL - 100,
            ),
            (
                self.SCREEN_SIZE[0] / 2
                + (self.goal_point + self.tolerance / 2) * self.SCALE,
                self.GROUND_LEVEL + 100,
            ),
        )

    def _draw_robot(self) -> None:
        self.screen.blit(self.capsule_surf, self.capsule_rect)
        self.screen.blit(self.inner_surf, self.inner_rect)

    @staticmethod
    def _extract_state(state: List[float]) -> Tuple[float, float]:
        x = state[0]
        xi = state[2]
        return x, xi

    def _pose_update(self, state: List[float]) -> None:
        x, xi = self._extract_state(state)
        self.capsule_rect.centerx = self.SCREEN_SIZE[0] / 2 + self.SCALE * x
        self.inner_rect.centerx = self.capsule_rect.midtop[0] + self.SCALE * xi

    def _view_update(self, state: List[float], goal_point: Optional[float], time: float) -> None:
        self.screen.fill(self.WHITE)
        self._draw_ground()
        self._draw_robot()
        self._draw_text_info(state, goal_point, time)
        if self.render_target_region:
            self._draw_target_region()
        pygame.display.update()

    def _draw_text_info(self, state: List[float], goal_point: Optional[float], time: float) -> None:
        # robot x pose
        x, _ = self._extract_state(state)
        pose_text_surf = self.font.render(f"x: {round(x, 5)}", False, self.BLACK)
        pose_text_rect = pose_text_surf.get_rect(topleft=(20, 20))
        self.screen.blit(pose_text_surf, pose_text_rect)

        # goal point
        if goal_point is not None:
            goal_point_text_surf = self.font.render(f"goal point: {round(goal_point, 3)}", False, self.BLACK)
            goal_point_text_rect = goal_point_text_surf.get_rect(topleft=(20, 80))
            self.screen.blit(goal_point_text_surf, goal_point_text_rect)

        # total time
        total_time_text_surf = self.font.render(f"total time: {round(time, 3)}", False, self.BLACK)
        total_time_text_rect = total_time_text_surf.get_rect(topleft=(20, 50))
        self.screen.blit(total_time_text_surf, total_time_text_rect)

    @property
    def goal_point(self):
        return self._goal_point

    @goal_point.setter
    def goal_point(self, goal_point):
        self._goal_point = goal_point

    @property
    def tolerance(self):
        return self._tolerance

    @tolerance.setter
    def tolerance(self, tolerance):
        self._tolerance = tolerance

    @staticmethod
    def quit() -> None:
        exit()

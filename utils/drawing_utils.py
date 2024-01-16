import os

import numpy as np

__all__ = ['create_background']

try:
    import pygame
except (ImportError, ModuleNotFoundError) as e:
    if 'SLURM_CLUSTER_NAME' not in os.environ:
        raise e
    else:
        pygame = None
        print("Import pygame ignored on cluster.")
finally:
    if pygame is not None:
        __all__.append('pygame')


os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"


def create_background(width, height):
    background = pygame.Surface((width, height))
    pygame.draw.rect(background, (255, 255, 204), pygame.Rect(0, 0, width, height))
    return background


def rect_points(center, width, height, ang, scaling, offset):
    points = []
    diag = np.sqrt(width ** 2 + height ** 2) / 2
    ang1 = 2 * np.arctan2(height, width)
    ang2 = 2 * np.arctan2(width, height)
    # Pygame's y axis points downwards. Hence invert y coordinate alone before offset ->
    points.append((offset[0] + scaling * (center[0] + np.cos(ang + ang1 / 2) * diag),
                   offset[1] - scaling * (center[1] + np.sin(ang + ang1 / 2) * diag)))
    points.append((offset[0] + scaling * (center[0] + np.cos(ang + ang1 / 2 + ang2) * diag),
                   offset[1] - scaling * (center[1] + np.sin(ang + ang1 / 2 + ang2) * diag)))
    points.append((offset[0] + scaling * (center[0] + np.cos(ang + ang1 * 1.5 + ang2) * diag),
                   offset[1] - scaling * (center[1] + np.sin(ang + ang1 * 1.5 + ang2) * diag)))
    points.append((offset[0] + scaling * (center[0] + np.cos(ang + ang1 * 1.5 + 2 * ang2) * diag),
                   offset[1] - scaling * (center[1] + np.sin(ang + ang1 * 1.5 + 2 * ang2) * diag)))

    return points

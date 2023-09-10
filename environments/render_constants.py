"""
This file contains the constants used in the render.py file.
"""
SCREEN_HEIGHT = 1000  # Want a screen that is 1000 pixels high and 1000 pixels wide
SCREEN_WIDTH = 1000
DAMPING = 1.0  # Is a value between 0 and 1.0 that determines how much the velocity of a body is dampened each step. A value of 0.9 means that each step the velocity will be reduced 10% (multiplied by 0.9).
GRAVITY = (
    -5
)  # Is the gravity for the space. The default is (0, 0), which means no gravity. The value is in pixels per second squared.
BALL_RADIUS = 10  # Is the radius of the ball in pixels.
SHELF_WIDTH = SCREEN_WIDTH // 10  # Is the width of the shelf in pixels.
BALL_COLLISION_TYPE = 1
SHELF_COLLISION_TYPE = 2
GROUND_COLLISION_TYPE = 3

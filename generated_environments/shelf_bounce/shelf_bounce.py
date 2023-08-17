import random
from typing import Optional

import arcade

from environments.environment import Environment, EnvironmentView
from environments.objects import Ball, Ground, Rectangle
from environments.render_constants import SCREEN_WIDTH as WIDTH, SCREEN_HEIGHT as HEIGHT, BALL_RADIUS, GRAVITY, DAMPING, \
    SHELF_WIDTH, SHELF_COLLISION_TYPE, BALL_COLLISION_TYPE


class ShelfBounceEnv(Environment):
    def __init__(self, render: bool = False, height_limit: float = 0.2, **kwargs):
        """ This is an environment where a ball is dropped from a random height onto an angled shelf and the model has to predict
        x position of the ball when it hits the ground.

        Args:
            render (bool, optional): Whether to render the environment or not. Defaults to True.
            height_limit (float, optional): portion of the screen to drop the ball from. Defaults to 0.2.
        """

        super().__init__(render=render)
        # init initialises the space, objects, text_log, numerical_log
        self.space.gravity = (0, GRAVITY)
        self.space.damping = DAMPING
        self.objects = []
        self.text_log = []
        self.numerical_log = dict(
            shelf_width=None,
            shelf_angle=None,
            shelf_x=None,
            shelf_y=None,
            radius=BALL_RADIUS,
            elasticity=None,
            ball_x0=None,
            ball_y0=None,
            ball_x1=None,
        )
        self.ball = None
        self.shelf = None
        self.width = WIDTH
        self.height = HEIGHT
        self.height_limit = height_limit

        # Add window title if it is not already in kwargs
        if "title" not in kwargs:
            self.title = "Shelf Bounce"
        else:
            self.title = kwargs["title"]

        if "fixed_height" in kwargs:
            self.fixed_height = kwargs["fixed_height"]
            self.drop_height = kwargs["drop_height"]
        else:
            self.fixed_height = False

        if "fixed_angle" in kwargs:
            self.fixed_angle = kwargs["fixed_angle"]
            self.angle = kwargs["angle"]
        else:
            self.fixed_angle = False

        if "fixed_shelf_width" in kwargs:
            self.fixed_shelf_width = kwargs["fixed_shelf_width"]
            self.width = kwargs["shelf_width"]
        else:
            self.fixed_shelf_width = False

        if "fixed_shelf_position" in kwargs:
            self.fixed_shelf_x = kwargs["fixed_shelf_position"]
            self.shelf_x = kwargs["shelf_position"]
        else:
            self.fixed_shelf_x = False

        if "fixed_ball_position" in kwargs:
            self.fixed_ball_x = kwargs["fixed_ball_position"]
            self.ball_x = kwargs["ball_position"]

        # Add ground
        ground = Ground(self.space, width=WIDTH)
        ground.elasticity = 0.9

        self.add_shelf()
        self.add_ball()
        self.render_mode = render
        self.end_episode = False

        if self.render_mode:
            self.window = arcade.Window(WIDTH, HEIGHT, self.title)
            self.window.show_view(EnvironmentView(self))
            # arcade.run()


        else:
            while True:  # Infinite loop for logging mode
                self.update(1 / 60)  # Update every 1/60 seconds
                if self.end_episode:
                    break

    def add_ball(self):
        """Makes a ball and adds it to the environment"""
        # Pick a random x and y coordinate for the ball
        x = self.width // 2
        y = round(random.uniform((1 - self.height_limit) * self.height, self.height), 2)
        elasticity = 0.8

        self.ball = Ball(self.space, radius=BALL_RADIUS, mass=1, x=x, y=y, collision_type=BALL_COLLISION_TYPE,
                         elasticity=elasticity)

        self.objects.append(self.ball)
        # Update the initial conditions in the numerical log and text log
        self.numerical_log["ball_x0"] = x
        self.numerical_log["ball_y0"] = y
        self.numerical_log["elasticity"] = elasticity

        self.text_log.append(f"Ball of radius {BALL_RADIUS} and elasticity {elasticity} is dropped from "
                             f"x = {round(x, 2)} and y = {round(y, 2)}.")

    def add_shelf(self):
        """Makes a rectangular shelf and adds it to the environment"""
        # Pick a random y coordinate for the shelf
        # Want the y coordinate to be between the height limit and the middle of the screen

        # Pick a random x coordinate for the shelf
        x = self.width // 2
        y = round(random.uniform(self.height_limit * self.height, 0.5 * self.height), 2)

        # Pick a random width for the shelf
        width = SHELF_WIDTH

        # Pick a random angle between +/- 45 degrees

        angle = round(random.uniform(-45, 45), 2)

        # Make the shelf
        self.shelf = Rectangle(self.space, width=width, height=BALL_RADIUS, x=x, y=y, angle=angle,
                               collision_type=SHELF_COLLISION_TYPE)

        # Add the shelf to the environment
        self.objects.append(self.shelf)

        # Update the initial conditions in the numerical log and text log
        self.numerical_log["shelf_x"] = float(x)
        self.numerical_log["shelf_y"] = float(y)
        self.numerical_log["shelf_angle"] = angle
        self.numerical_log["shelf_width"] = float(width)
        self.text_log.append(f'Shelf of width {float(width)} and angle {angle} degrees at x={float(x)} y={y}')

    def update(self, delta_time):
        # if self.render_mode:
        self.space.step(delta_time)  # Advance the simulation by 1/60th of a second
        self.elapsed_time += delta_time

        # End the episode if the ball hits the ground
        if self.ball.body.position.y <= BALL_RADIUS:
            self.end_episode = True
            self.log_state()

        if self.end_episode and self.render_mode:
            arcade.close_window()


if __name__ == "__main__":
    env = ShelfBounceEnv(render=True)  # Set render=True for visualization, set render=False for text logging
    if env.render_mode:
        arcade.run()
    print(env.numerical_log)
    print(env.text_log)

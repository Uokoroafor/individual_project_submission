import random

import arcade

from environments.environment import Environment, EnvironmentView
from environments.objects import Ball, Ground, Rectangle
from environments.render_constants import SCREEN_WIDTH as WIDTH, SCREEN_HEIGHT as HEIGHT, BALL_RADIUS, GRAVITY, DAMPING, \
    SHELF_WIDTH, SHELF_COLLISION_TYPE, BALL_COLLISION_TYPE, GROUND_COLLISION_TYPE


class ShelfBounceEnv(Environment):
    def __init__(self, render: bool = False, height_limit: float = 0.2, time_limit=10, **kwargs):
        """ This is an environment where a ball is dropped from a random height onto an angled shelf and the model has to predict
        x position of the ball when it hits the ground.

        Args:
            render (bool, optional): Whether to render the environment or not. Defaults to True.
            height_limit (float, optional): portion of the screen to drop the ball from. Defaults to 0.2.
            time_limit (int, optional): The time limit for the episode. Defaults to 10.
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
            t1=None,
            ball_x1=None,
            ball_y1=None,
        )
        self.ball = None
        self.shelf = None
        self.width = WIDTH
        self.height = HEIGHT
        self.height_limit = height_limit
        self.elapsed_time = 0.0
        self.time_limit = time_limit

        # Add window title if it is not already in kwargs
        if "title" not in kwargs:
            self.title = "Shelf Bounce"
        else:
            self.title = kwargs["title"]

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

        if "fixed_shelf_x" in kwargs:
            self.fixed_shelf_x = kwargs["fixed_shelf_position"]
            self.shelf_x = kwargs["shelf_position"]
        else:
            self.fixed_shelf_x = False

        if "fixed_shelf_y" in kwargs:
            self.fixed_shelf_y = kwargs["fixed_shelf_y"]
            self.shelf_y = kwargs["shelf_y"]
        else:
            self.fixed_shelf_y = False

        if "fixed_ball_y" in kwargs:
            self.fixed_ball_height = kwargs["fixed_ball_y"]
            self.ball_y = kwargs["ball_y"]
        else:
            self.fixed_ball_height = False

        if "fixed_ball_x" in kwargs:
            self.fixed_ball_x = kwargs["fixed_ball_x"]
            self.ball_x = kwargs["ball_x"]
        else:
            self.fixed_ball_x = False

        if "fixed_ball_elasticity" in kwargs:
            self.fixed_ball_elasticity = kwargs["fixed_ball_elasticity"]
            self.ball_elasticity = kwargs["ball_elasticity"]
        else:
            self.fixed_ball_elasticity = False

        # Add ground to the space
        ground = Ground(self.space, width=WIDTH, elasticity=0, collision_type=GROUND_COLLISION_TYPE)

        self.add_shelf()
        self.add_ball()
        self.render_mode = render
        self.end_episode = False
        self.time_up = False

        if self.render_mode:
            self.window = arcade.Window(WIDTH, HEIGHT, self.title)
            arcade.set_background_color(arcade.color.BLACK)
            self.window.show_view(EnvironmentView(self))

        else:
            while True:  # Infinite loop for logging mode
                self.update(1 / 6)  # Update every 1/60 seconds
                if self.end_episode:
                    break

    def add_ball(self):
        """Makes a ball and adds it to the environment"""
        # Pick a random x and y coordinate for the ball
        if self.fixed_ball_x:
            x = self.ball_x
        else:
            x = self.width // 2
        if self.fixed_ball_height:
            y = self.ball_y
        else:
            y = round(random.uniform((1 - self.height_limit) * self.height, self.height), 2)

        if self.fixed_ball_elasticity:
            elasticity = self.ball_elasticity
        else:
            elasticity = 0.5

        self.ball = Ball(self.space, radius=BALL_RADIUS, mass=1, x=x, y=y, collision_type=BALL_COLLISION_TYPE,
                         elasticity=elasticity)

        self.objects.append(self.ball)
        # Update the initial conditions in the numerical log and text log
        self.numerical_log["ball_x0"] = x
        self.numerical_log["ball_y0"] = y
        self.numerical_log["elasticity"] = elasticity

        self.text_log.append(f"Ball of radius {BALL_RADIUS} and elasticity {elasticity} is dropped from "
                             f"x={round(x, 2)} and y={round(y, 2)}")

    def add_shelf(self):
        """Makes a rectangular shelf and adds it to the environment"""
        # Pick a random y coordinate for the shelf
        # Want the y coordinate to be between the height limit and the middle of the screen

        # Pick a random x coordinate for the shelf
        if self.fixed_shelf_x:
            x = self.shelf_x
        else:
            x = self.width // 2

        if self.fixed_shelf_y:
            y = self.shelf_y
        else:
            y = round(random.uniform(self.height_limit * self.height, 0.5 * self.height), 2)

        if self.fixed_angle:
            angle = self.angle
        else:
            angle = round(random.uniform(-45, 45), 2)

        # Pick a random width for the shelf
        width = SHELF_WIDTH

        # Make the shelf
        self.shelf = Rectangle(self.space, width=width, height=BALL_RADIUS, x=x, y=y, angle=angle,
                               collision_type=SHELF_COLLISION_TYPE)

        # Add the shelf to the environment
        self.objects.append(self.shelf)

        # Update the initial conditions in the numerical log and text log
        self.numerical_log["shelf_x"] = round(float(x), 2)
        self.numerical_log["shelf_y"] = round(float(y), 2)
        self.numerical_log["shelf_angle"] = angle
        self.numerical_log["shelf_width"] = float(width)
        self.text_log.append(f'Shelf of width {float(width)} and angle {angle} degrees at x={float(x)} y={y}')

    def update(self, delta_time):
        if self.render_mode:
            delta_time *= 4
        self.space.step(delta_time)  # Advance the simulation by 1/60th of a second
        self.elapsed_time += delta_time

        # End the episode if it is past the time limit and the ball has hit the ground
        if round(self.elapsed_time, 1) >= self.time_limit and not self.time_up:
            self.log_state(self.elapsed_time)
            self.time_up = True
            self.print_message = (f"At Time of {round(self.elapsed_time, 1)}: Ball was at "
                                  f"x={round(self.ball.body.position.x, 2)} y={round(self.ball.body.position.y, 2)}")

        if (self.ball.body.position.y <= BALL_RADIUS and self.time_up) or \
                (self.time_up and not self.render_mode):
            # End the episode if the ball hits the ground in render mode or hits time limit in logging mode
            self.end_episode = True

        if self.end_episode and self.render_mode:
            arcade.close_window()

    def log_state(self, t: float):
        """Log the final state of the environment

        Args:
            t (float): The time at which the state is logged
            """
        # Get the final y coordinate of the ball
        x = self.ball.body.position.x
        y = float(max(self.ball.body.position.y, BALL_RADIUS))

        self.numerical_log["t1"] = round(t, 1)
        self.numerical_log["ball_x1"] = round(x, 2)
        self.numerical_log["ball_y1"] = round(y, 2)
        self.text_log.append(f"At time {round(t, 1)}")
        self.text_log.append(f"Ball is at x={round(x, 2)}")
        self.text_log.append(f"y={round(y, 2)}")


if __name__ == "__main__":
    # env = ShelfBounceEnv(render=True)  # Set render=True for visualization, set render=False for text logging
    # Render Fixed Angle with variable time limit

    # env = ShelfBounceEnv(render=False, fixed_angle=True, angle=30, time_limit=round(random.uniform(5, 15), 1))
    for _ in range(10):
        env = ShelfBounceEnv(render=True, time_limit=10)
        if env.render_mode:
            arcade.run()
        print(env.numerical_log)
        print(env.text_log)

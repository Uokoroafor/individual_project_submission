import random
from typing import List

import arcade
import pymunk

from environments.environment import Environment, EnvironmentView
from environments.objects import Ball, Ground, Rectangle
from environments.render_constants import (
    SCREEN_WIDTH as WIDTH,
    SCREEN_HEIGHT as HEIGHT,
    BALL_RADIUS,
    GRAVITY,
    DAMPING,
    SHELF_WIDTH,
    SHELF_COLLISION_TYPE,
    BALL_COLLISION_TYPE,
    GROUND_COLLISION_TYPE,
)

SHELF_WIDTH *= 1.5


class MultiShelfBounceEnv(Environment):
    def __init__(
        self,
        render: bool = False,
        height_limit: float = 0.2,
        time_limit: float = 10.0,
        num_shelves=2,
        **kwargs,
    ):
        """This is an environment where a ball is dropped from a random height onto an angled shelf and the model has to predict
        x position of the ball when it hits the ground.

        Args:
            render (bool, optional): Whether to render the environment or not. Defaults to True.
            height_limit (float, optional): portion of the screen to drop the ball from. Defaults to 0.2.
            time_limit (float, optional): The time limit for the episode. Defaults to 10.0.
            num_shelves (int, optional): The number of shelves in the environment. Defaults to 2.
        """

        super().__init__(render=render)
        # init initialises the space, objects, text_log, numerical_log
        self.space.gravity = (0, GRAVITY)
        self.space.damping = DAMPING
        self.objects = []
        self.shelf_list = []
        self.num_shelves = num_shelves
        self.text_log = []
        self.numerical_log = dict()
        self.make_numeric_log()

        self.ball = None
        self.width = WIDTH
        self.height = HEIGHT
        self.height_limit = height_limit
        self.elapsed_time = 0.0
        self.time_limit = time_limit

        # Add window title if it is not already in kwargs
        if "title" not in kwargs:
            self.title = "Multi Shelf Bounce"
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
            self.fixed_shelf_x = kwargs["fixed_shelf_x"]
            self.shelf_x = kwargs["shelf_x"]
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

        if self.fixed_shelf_y and self.fixed_shelf_x and self.num_shelves > 2:
            print(
                "Warning: fixed shelf y coordinates will be ignored if there are more than 2 shelves"
            )
            self.fixed_shelf_y = False

        # Add ground
        ground = Ground(
            self.space, width=WIDTH, elasticity=0, collision_type=GROUND_COLLISION_TYPE
        )

        self.make_shelves()
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
            # pick a random point in the middle of the screen
            x = round(random.uniform(WIDTH // 3, WIDTH * 2 // 3), 2)

        if self.fixed_ball_height:
            y = self.ball_y
        else:
            y = round(
                random.uniform((1 - self.height_limit) * self.height, self.height), 2
            )

        if self.fixed_ball_elasticity:
            elasticity = self.ball_elasticity
        else:
            elasticity = 0.5

        self.ball = Ball(
            self.space,
            radius=BALL_RADIUS,
            mass=1,
            x=x,
            y=y,
            collision_type=BALL_COLLISION_TYPE,
            elasticity=elasticity,
        )

        self.objects.append(self.ball)
        # Update the initial conditions in the numerical log and text log
        self.numerical_log["ball_x0"] = x
        self.numerical_log["ball_y0"] = y
        self.numerical_log["elasticity"] = elasticity

        self.text_log.append(
            f"Ball of radius {BALL_RADIUS} and elasticity {elasticity} is dropped from "
            f"x={round(x, 2)} and y={round(y, 2)}"
        )

    def make_shelves(self):
        """Makes a number of shelves and adds them to the environment"""
        # Pick a random x coordinate for the shelf

        # [Want the shelves to be mirrored across the center of the screen]
        x_list = self.make_shelf_x_list()
        y_list = self.make_shelf_y_list()

        for i in range(self.num_shelves):
            x = x_list[i]
            y = y_list[i]

            if self.fixed_angle:
                angle = self.angle if i % 2 == 0 else -self.angle
            else:
                angle = round(random.uniform(-45, 45), 2)
            self.add_shelf(x, y, angle)

    def add_shelf(self, x: float, y: float, angle: float):
        """Makes a rectangular shelf and adds it to the environment

        Args:
            x (float): The x coordinate of the shelf
            y (float): The y coordinate of the shelf
            angle (float): The angle of the shelf
        """
        # Pick a random y coordinate for the shelf
        # Want the y coordinate to be between the height limit and the middle of the screen

        # Pick a random width for the shelf
        width = SHELF_WIDTH

        overlap = True
        while overlap:
            # Make the shelf
            shelf = Rectangle(
                self.space,
                width=width,
                height=BALL_RADIUS,
                x=x,
                y=y,
                angle=angle,
                collision_type=SHELF_COLLISION_TYPE,
            )
            overlap = self.check_shelf_overlaps(shelf)

        self.shelf_list.append(shelf)

        # Add the shelf to the environment
        self.objects.append(shelf)

        shelf_id = len(self.shelf_list) - 1

        # Update the initial conditions in the numerical log and text log
        self.numerical_log[f"shelf{shelf_id}_x"] = round(float(x), 2)
        self.numerical_log[f"shelf{shelf_id}_y"] = round(float(y), 2)
        self.numerical_log[f"shelf{shelf_id}_angle"] = round(angle, 2)
        self.numerical_log[f"shelf{shelf_id}_width"] = float(width)
        self.text_log.append(
            f"Shelf{shelf_id} of width {float(width)} and angle {angle} degrees at x={float(x)} y={y}"
        )

    @staticmethod
    def check_shelf_overlap(shelf1: Rectangle, shelf2: Rectangle) -> bool:
        """Checks if two shelves overlap

        Args:
            shelf1 (Rectangle): The first shelf
            shelf2 (Rectangle): The second shelf
        """
        # Get the coordinates of the corners of both shelves
        contacts = pymunk.Shape.shapes_collide(shelf1.shape, shelf2.shape)
        # If there are no contacts, there is no overlap
        return len(contacts.points) > 0

    def check_shelf_overlaps(self, shelf1: Rectangle) -> bool:
        """
        This checks if a shelf overlaps with any other shelf in the environment
        Args:
            shelf1: The shelf to check for overlaps with

        Returns:
            bool: True if there is an overlap, False otherwise
        """
        if len(self.shelf_list) > 0:
            for shelf2 in self.shelf_list:
                if shelf1 != shelf2:
                    if self.check_shelf_overlap(shelf1, shelf2):
                        return True
        return False

    def make_numeric_log(self):
        """This initialises the numeric log by listing the variables that will be logged. It includes information for
        each of the shelves added"""
        for i in range(self.num_shelves):
            self.numerical_log[f"shelf{i}_width"] = None
            self.numerical_log[f"shelf{i}_angle"] = None
            self.numerical_log[f"shelf{i}_x"] = None
            self.numerical_log[f"shelf{i}_y"] = None

        # Add the ball's initial position
        self.numerical_log["radius"] = BALL_RADIUS
        self.numerical_log["ball_x0"] = None
        self.numerical_log["ball_y0"] = None
        self.numerical_log["elasticity"] = None

        # Add the ball's final position
        self.numerical_log["t1"] = None
        self.numerical_log["ball_x1"] = None
        self.numerical_log["ball_y1"] = None

    def update(self, delta_time):
        if self.render_mode:
            delta_time *= 4
        self.space.step(delta_time)  # Advance the simulation by 1/60th of a second
        self.elapsed_time += delta_time

        # End the episode if it is past the time limit and the ball has hit the ground
        if self.elapsed_time >= self.time_limit and not self.time_up:
            self.log_state(self.elapsed_time)
            self.time_up = True
            self.print_message = (
                f"At Time of {round(self.elapsed_time, 1)}: Ball was at "
                f"x={round(self.ball.body.position.x, 2)} y={round(self.ball.body.position.y, 2)}"
            )

        if self.ball.body.position.y <= BALL_RADIUS and self.time_up:
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

    def make_shelf_x_list(self) -> List[float]:
        """Want to make a list of x coordinates of the shelves. For simplicity, the shelves will be of the same width
        and mirrored across the centre of the screen. x coordinates can be either given or randomly generated. They can also be given as a single value or a list
        """
        # Pick a random x coordinate for the shelf
        # Want the x coordinate to be between the width limit and the middle of the screen
        shelf_x_list = []
        if self.fixed_shelf_x:
            if type(self.shelf_x) == list:
                assert (
                    len(self.shelf_x) == self.num_shelves
                ), "The number of shelves must match the number of x coordinates"
                shelf_x_list = self.shelf_x
                return shelf_x_list
            else:
                x1 = self.shelf_x
                # mirror the shelf across the center of the screen
                x2 = WIDTH - x1
        else:
            # Pick a random x and mirror it
            x1 = round(random.uniform(WIDTH // 3, WIDTH / 2 - SHELF_WIDTH // 3), 2)
            # mirror the shelf across the center of the screen
            x2 = round(WIDTH - x1, 2)

        for i in range(self.num_shelves):
            if i % 2 == 0:
                shelf_x_list.append(x1)
            else:
                shelf_x_list.append(x2)

        return shelf_x_list

    def make_shelf_y_list(self) -> List[float]:
        """Want to make a list of y coordinates of the shelves. For simplicity, the shelves will be of the same width
        and mirrored across the centre of the screen. y coordinates can be either given or randomly generated. They can also be given as a single value or a list
        """
        shelf_y_list = []
        for i in range(self.num_shelves):
            if self.fixed_shelf_y:
                if type(self.shelf_y) == list:
                    assert (
                        len(self.shelf_y) == self.num_shelves
                    ), "The number of shelves must match the number of y coordinates"
                    shelf_y_list = self.shelf_y
                    return shelf_y_list
                else:
                    y = self.shelf_y

            else:
                y = round(
                    random.uniform(self.height_limit * self.height, 0.5 * self.height),
                    2,
                )
            shelf_y_list.append(y)

        return shelf_y_list


if __name__ == "__main__":
    # env = ShelfBounceEnv(render=True)  # Set render=True for visualization, set render=False for text logging
    for _ in range(5):
        # Render Fixed Angle and shelf x with random shelf y and variable time limit
        env = MultiShelfBounceEnv(
            render=True,
            fixed_angle=True,
            angle=-30,
            fixed_shelf_x=True,
            shelf_x=WIDTH // 3,
            fixed_shelf_y=True,
            shelf_y=[0.5 * HEIGHT, 0.5 * HEIGHT],
            time_limit=round(random.uniform(5, 15), 1),
            fixed_ball_y=True,
            ball_y=0.8 * HEIGHT,
            num_shelves=2,
        )

        if env.render_mode:
            arcade.run()
        print(env.numerical_log)
        print(env.text_log)

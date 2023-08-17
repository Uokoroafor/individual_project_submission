import random
from typing import Optional

import arcade

from environments.environment import Environment, EnvironmentView
from environments.objects import Ball, Ground
from environments.render_constants import SCREEN_WIDTH as WIDTH, SCREEN_HEIGHT as HEIGHT, BALL_RADIUS, GRAVITY, DAMPING


class FreefallEnv(Environment):
    def __init__(self, render: bool = True, height_range: float = 0.2, time_limit: int = 5, **kwargs):
        """ This is an environment where a ball is dropped from a random height and the model has to predict the
        height after a certain time.
        Args:
            render (bool, optional): Whether to render the environment or not. Defaults to True.
            height_range (float, optional): portion of the screen to drop the ball from. Defaults to 0.2.
            time_limit (int, optional): The maximum time for the ball to fall. Defaults to 5.
        """

        super().__init__(render=render)
        # init initialises the space, objects, text_log, numerical_log

        self.space.gravity = (0, GRAVITY)
        self.space.damping = DAMPING
        self.objects = []
        self.text_log = []
        # self.text_log.append(f"Gravity is {GRAVITY}.")
        self.numerical_log = dict(
            radius=float(BALL_RADIUS),
            y_0=None,
            t_1=None,
            y_1=None,
        )
        self.ball = None
        # Add window title if it is not already in kwargs
        if "title" not in kwargs:
            self.title = "Freefall"
        else:
            self.title = kwargs["title"]

        if "fixed_height" in kwargs:
            self.fixed_height = kwargs["fixed_height"]
            self.drop_height = kwargs["drop_height"]
        else:
            self.fixed_height = False

        if "allow_bounce" in kwargs:
            self.allow_bounce = kwargs["allow_bounce"]
            if self.allow_bounce:
                self.ball_elasticity = kwargs["elasticity"]
            else:
                self.ball_elasticity = 0.0
        else:
            self.allow_bounce = False
            self.ball_elasticity = 0.0

        self.width = WIDTH
        self.height = HEIGHT
        self.height_limit = height_range
        self.time_limit = time_limit
        self.elapsed_time = 0.0

        # Add ground
        ground = Ground(self.space, width=WIDTH)
        ground.elasticity = 0.0  # Make the ground not bouncy

        self.add_ball()
        self.render_mode = render
        self.end_episode = False

        if self.render_mode:
            self.window = arcade.Window(WIDTH, HEIGHT, self.title)
            self.window.show_view(EnvironmentView(self))
            arcade.run()

        else:
            while True:  # Infinite loop for logging mode
                self.update(1 / 10)  # Update every 1/60 seconds

                if self.end_episode:
                    break

    def update(self, delta_time):
        # if self.render_mode:
        self.space.step(delta_time)  # Advance the simulation by 1/60th of a second
        self.elapsed_time += delta_time

        # End the episode after a set amount of time
        if round(self.elapsed_time, 1) >= self.time_limit:
            self.elapsed_time = round(self.elapsed_time, 1)
            self.log_state(self.elapsed_time)
            self.end_episode = True

        if self.end_episode and self.render_mode:
            arcade.close_window()

    def add_ball(self):
        """Makes a ball and adds it to the environment"""
        # Pick a random x and y coordinate for the ball
        x = round(random.uniform(0, self.width), 2)
        if self.fixed_height:
            y = float(self.drop_height)
        else:
            y = round(random.uniform((1 - self.height_limit) * self.height, self.height), 2)
        self.ball = Ball(self.space, radius=BALL_RADIUS, mass=1, x=x, y=y)

        # Set the elasticity of the ball
        self.ball.shape.elasticity = self.ball_elasticity

        self.objects.append(self.ball)
        # Update the initial conditions in the numerical log and text log
        self.numerical_log["y_0"] = round(y, 2)
        self.text_log.append(f"Ball of radius {float(BALL_RADIUS)} dropped from y={y:.2f}")

    def log_state(self, t):
        y1 = max(self.ball.body.position.y, float(self.ball.radius))

        self.numerical_log["t_1"] = round(t, 1)
        self.numerical_log["y_1"] = round(y1, 2)
        self.text_log.append(f"At time {round(t, 1)}")
        self.text_log.append(f"Ball is at y={y1:.2f}")

    def make_minimal_text(self, split_text: str = "ans: ", ans_key: Optional[str] = None) -> str:
        """Joins up the text in the numerical log to make a minimal text"""
        # Join the column names and values
        if ans_key is None:
            # Take the last value in the numerical log as the answer
            ans_key = list(self.numerical_log.keys())[-1]
        text = ""
        for key, value in self.numerical_log.items():
            if key != ans_key:
                text += f"{key}: {value}, "

        # Add the split text and the answer
        text += f"{split_text}{self.numerical_log[ans_key]}"
        return text

    def make_descriptive_text(self, split_text: str = "ans: ", ans_index: Optional[int] = None) -> str:
        """Joins up the text in the text log to make a descriptive text"""
        # Join the column names and values
        if ans_index is None:
            # Take the last value in the numerical log as the answer
            ans_index = len(self.text_log) - 1
        text = ""
        for i, line in enumerate(self.text_log):
            if i != ans_index:
                text += f"{line} "

        # Add the split text and the answer
        text += f"{split_text}{self.text_log[ans_index]}"
        return text


if __name__ == "__main__":
    # Example Usage

    time_limit = round(random.uniform(1, 10), 2)
    env = FreefallEnv(render=False, time_limit=time_limit, fixed_height=True, drop_height=HEIGHT // 2)
    print(env.numerical_log)
    print(env.text_log)

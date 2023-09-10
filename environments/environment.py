from typing import Optional

import arcade
import pymunk


class Environment:
    def __init__(self, render=True):
        """Abstract Class for an environment that will render and/or log text data

        Args:
            render (bool, optional): Whether to render the environment. Defaults to True.
        """
        self.space = pymunk.Space()
        self.objects = []
        self.text_log = []
        self.numerical_log = dict()
        self.render_mode = render
        self.print_message = None

    def make_minimal_text(self, split_text: str = "ans: ", ans_key: Optional[str] = None) -> str:
        """Joins up the text in the numerical log to make a minimal text

        Args:
            split_text (str, optional): The text to split the text log with. Defaults to "ans: ".
            ans_key (Optional[str], optional): The key of the answer in the numerical log. Defaults to None.

        Returns:
            str: The minimal text
        """
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
        """Joins up the text in the text log to make a descriptive text

        Args:
            split_text (str, optional): The text to split the text log with. Defaults to "ans: ".
            ans_index (Optional[int], optional): The index of the answer in the text log. Defaults to None.

        Returns:
            str: The descriptive text
        """
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

    def update(self, delta_time):
        """Abstract method to update the environment

        Args:
            delta_time (float): The time since the last update
        """
        pass


class EnvironmentView(arcade.View):
    def __init__(self, env: Environment):
        """Abstract Class for an arcade view that will render an environment

        Args:
            env (Environment): The environment to render
        """
        super().__init__()
        self.env = env

    def on_draw(self):
        """Draws the environment"""
        arcade.start_render()
        for obj in self.env.objects:
            obj.draw()
        if self.env.print_message is not None:
            arcade.draw_text(self.env.print_message, 10, 10, arcade.color.WHITE, 20)

    def on_update(self, delta_time):
        """Updates the environment

        Args:
            delta_time (float): The time since the last update
        """
        self.env.update(delta_time)

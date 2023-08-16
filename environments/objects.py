import math
from typing import List, Optional, Tuple

import arcade
import pymunk


class Ball:
    def __init__(self, space: pymunk.Space, x: float, y: float, radius: float, collision_type: int = 1,
                 elasticity: float = 0.8,
                 mass: float = 10.0, colour: arcade.Color = arcade.color.GOLD):
        """ This is a ball object that can be used in any arcade environment.
        It is a wrapper around pymunk's circle object.

        Args:
            space (pymunk.Space): The pymunk space in which the ball will be added
            x (float): The x coordinate of the ball's center
            y (float): The y coordinate of the ball's center
            radius (float): The radius of the ball
            collision_type (int, optional): The collision type of the ball. Defaults to 1.
            elasticity (float, optional): The elasticity of the ball. Defaults to 0.8.
            mass (float, optional): The mass of the ball. Defaults to 10.0.
            colour (arcade.Color, optional): The colour of the ball. Defaults to arcade.color.GOLD.
        """
        moment = pymunk.moment_for_circle(mass, 0, radius, (0, 0))
        self.body = pymunk.Body(mass, moment)
        self.body.position = pymunk.Vec2d(x, y)
        self.shape = pymunk.Circle(self.body, radius)
        self.shape.elasticity = elasticity  # This means that the coin will bounce back with elasticity%
        # of its original speed
        self.radius = radius
        self.shape.collision_type = collision_type
        self.colour = colour
        space.add(self.body, self.shape)

    def draw(self):
        arcade.draw_circle_filled(self.body.position.x, self.body.position.y, self.radius, self.colour)


class Rectangle:
    def __init__(self, space: pymunk.Space, x: float, y: float, width: float, height: float, angle: float = 0,
                 collision_type: int = 1,
                 colour: arcade.Color = arcade.color.BROWN):
        """ This is a rectangle object that can be used in any arcade environment.
        It is a wrapper around pymunk's segment object.

        Args:
            space (pymunk.Space): The pymunk space in which the rectangle will be added
            x (int): The x coordinate of the rectangle's center
            y (int): The y coordinate of the rectangle's center
            width (float): The width of the rectangle
            height (float): The height of the rectangle
            angle (float, optional): The angle of the rectangle. Defaults to 0.
            collision_type (int, optional): The collision type of the rectangle. Defaults to 1.
        """

        self.body = pymunk.Body(body_type=pymunk.Body.STATIC)
        self.body.position = pymunk.Vec2d(x, y)
        self.shape = pymunk.Segment(self.body, pymunk.Vec2d(-width / 2, -height / 2),
                                    pymunk.Vec2d(width / 2, height / 2), 1)
        self.shape.elasticity = 1.0
        self.body.angle = math.radians(angle)
        self.width = width
        self.height = height
        self.angle = angle
        self.shape.collision_type = collision_type
        self.colour = colour
        space.add(self.body, self.shape)

    @property
    def center_x(self):
        return self.body.position.x

    @property
    def center_y(self):
        return self.body.position.y

    @property
    def top(self):
        return self.center_y + self.height / 2

    @property
    def bottom(self):
        return self.center_y - self.height / 2

    def get_corners(self) -> List[Tuple[float, float]]:
        """ Returns the four corners of the rectangle. """
        half_width = self.width / 2
        half_height = self.height / 2

        # Define the four corners of the rectangle.
        corners = [
            (-half_width, -half_height),
            (half_width, -half_height),
            (half_width, half_height),
            (-half_width, half_height)
        ]

        # Rotate each corner around the center of the rectangle.
        corners = [(x * math.cos(math.radians(self.angle)) - y * math.sin(math.radians(self.angle)),
                    x * math.sin(math.radians(self.angle)) + y * math.cos(math.radians(self.angle)))
                   for x, y in corners]

        # Offset each corner by the position of the rectangle.
        corners = [(x + self.center_x, y + self.center_y) for x, y in corners]

        return corners

    def draw(self):
        """ Draws the rectangle. """
        # Offset each corner by the position of the rectangle.
        corners = self.get_corners()

        arcade.draw_polygon_filled(corners, self.colour)


class Ground:

    def __init__(self, space: pymunk.Space, width: int, radius: int = 1, elasticity: float = 1.0):
        """ This is an object that can be used in any arcade environment. It is a pymunk static segment object that
        will be added to the space and used as the ground.

        Args:
            space (pymunk.Space): The space that the ground will be added to
            width (int): The width of the ground
            radius (int, optional): The radius of the ground. Defaults to 1.
            elasticity (int, optional): The elasticity of the ground. Defaults to 1.
        """

        self.body = pymunk.Body(body_type=pymunk.Body.STATIC)
        self.shape = pymunk.Segment(self.body, (0, 0), (width, 0), radius)
        self.shape.elasticity = elasticity
        self.shape.friction = 1.0
        space.add(self.body, self.shape)


class Boundary:

    def __init__(self, space: pymunk.Space, length: int, boundary_type: str, radius: int = 1, elasticity: float = 1.0):
        """ This is an object that can be used in any arcade environment. It is a pymunk static segment object that
        will be added to the space and used as the boundary.

        Args:
            space (pymunk.Space): The space that the boundary will be added to
            length (int): The length of the boundary
            boundary_type (str): The type of boundary. "l", "r", "t" or "b"
            radius (int, optional): The radius of the boundary. Defaults to 1.
            elasticity (float, optional): The elasticity of the boundary. Defaults to 1.
        """

        if boundary_type == "l":
            a = (0, 0)
            b = (0, length)
        elif boundary_type == "r":
            a = (length, 0)
            b = (length, length)
        elif boundary_type == "t":
            a = (0, length)
            b = (length, length)
        elif boundary_type == "b":
            a = (0, 0)
            b = (length, 0)
        else:
            raise ValueError("boundary_type must be 'l', 'r', 't' or 'b'")

        self.body = pymunk.Body(body_type=pymunk.Body.STATIC)
        self.shape = pymunk.Segment(self.body, a, b, radius)
        self.shape.elasticity = elasticity
        self.shape.friction = 1.0
        space.add(self.body, self.shape)

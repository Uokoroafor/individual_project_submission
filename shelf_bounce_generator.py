import random

from environments.render_constants import SCREEN_HEIGHT as HEIGHT
from generated_environments.shelf_bounce.shelf_bounce import ShelfBounceEnv
from utils.env_utils import RandomDict, pick_between_two_ranges, generate_environment_data
from utils.train_utils import set_seed

if __name__ == "__main__":
    # Set the seed for reproducibility
    set_seed(6_345_789)
    # Wilson Pickett - 634-5789 https://www.youtube.com/watch?v=TSGuaVAufV0

    # Set the time limits for training and testing - We will have two test sets.
    # One in the training time limits and one in the test time limits
    time_limits_train = [15, 25]
    time_limits_test = [10, 14.999], [25.001, 30]
    fixed_time = 20

    # Set the height limits for training and testing - We will have two test sets.
    ball_height_limits_train = [0.70, 0.9]
    ball_height_limits_test = [0.60001, 0.6999999], [0.9001, 1]
    ball_fixed_height = round(0.8 * HEIGHT, 2)

    # Set the height limit for the shelf
    shelf_height_limits_train = [0.4, 0.5]
    shelf_height_limits_test = [0.30001, 0.3999999], [0.5001, 0.599999]
    shelf_fixed_height = round(0.45 * HEIGHT, 2)

    # Angle limits
    angle_limits_train = [-45, 45]
    angle_limits_test = [-45, 45]
    fixed_angle = 30

    # Set the width limits for training and testing - We will have two test sets.
    width_limits_train = [0.05, 0.15]
    width_limits_test = [0.01, 0.049999], [0.15001, 0.2]
    fixed_width = 0.1

    Env = ShelfBounceEnv
    # Create Environment Dictionaries

    # Fixed Ball Height and Shelf Height - Variable Angle
    variable_angle_dict_train = RandomDict(render=False, fixed_shelf_y=True, shelf_y=shelf_fixed_height,
                                           fixed_ball_y=True, ball_y=ball_fixed_height, fixed_angle=True,
                                           angle=lambda: round(random.uniform(*angle_limits_train), 1),
                                           time_limit=fixed_time)

    variable_angle_dict_test = RandomDict(render=False, fixed_shelf_y=True, shelf_y=shelf_fixed_height,
                                          fixed_ball_y=True, ball_y=ball_fixed_height, fixed_angle=True,
                                          angle=lambda: round(random.uniform(*angle_limits_test), 1),
                                          time_limit=fixed_time)

    # Fixed Ball Height and Angle - Variable Shelf Height
    variable_shelfheight_dict_train = RandomDict(render=False, fixed_shelf_y=True,
                                                 shelf_y=lambda: round(
                                                     random.uniform(*shelf_height_limits_train) * HEIGHT, 2),
                                                 fixed_ball_y=True, ball_y=ball_fixed_height, fixed_angle=True,
                                                 angle=fixed_angle, time_limit=fixed_time)

    variable_shelfheight_dict_test = RandomDict(render=False, fixed_shelf_y=True, shelf_y=lambda: round(
        pick_between_two_ranges(shelf_height_limits_test[0], shelf_height_limits_test[1]) * HEIGHT, 2),
                                                fixed_ball_y=True, ball_y=ball_fixed_height, fixed_angle=True,
                                                angle=fixed_angle, time_limit=fixed_time)

    # Fixed Shelf Height and Angle - Variable Ball Height
    variable_ballheight_dict_train = RandomDict(render=False, fixed_shelf_y=True, shelf_y=shelf_fixed_height,
                                                fixed_ball_y=True, ball_y=lambda: round(
            random.uniform(*ball_height_limits_train) * HEIGHT, 2), fixed_angle=True, angle=fixed_angle,
                                                time_limit=fixed_time)

    variable_ballheight_dict_test = RandomDict(render=False, fixed_shelf_y=True, shelf_y=shelf_fixed_height,
                                               fixed_ball_y=True, ball_y=lambda: round(
            pick_between_two_ranges(ball_height_limits_test[0], ball_height_limits_test[1]) * HEIGHT, 2),
                                               fixed_angle=True, angle=fixed_angle, time_limit=fixed_time)

    # Fixed Shelf Height Environment - variable ball height and angle
    variable_ballheight_angle_dict_train = RandomDict(render=False, fixed_shelf_y=True, shelf_y=lambda: round(
        random.uniform(*shelf_height_limits_train) * HEIGHT, 2), fixed_ball_y=True, ball_y=ball_fixed_height,
                                                      fixed_angle=True,
                                                      angle=lambda: round(random.uniform(*angle_limits_train), 1),
                                                      time_limit=fixed_time)

    variable_ballheight_angle_dict_test = RandomDict(render=False, fixed_shelf_y=True, shelf_y=lambda: round(
        pick_between_two_ranges(shelf_height_limits_test[0], shelf_height_limits_test[1]) * HEIGHT, 2),
                                                     fixed_ball_y=True, ball_y=ball_fixed_height, fixed_angle=True,
                                                     angle=lambda: round(random.uniform(*angle_limits_test), 1),
                                                     time_limit=fixed_time)

    # Fixed Ball Height Environment - variable shelf height and angle
    variable_shelfheight_angle_dict_train = RandomDict(render=False, fixed_shelf_y=True, shelf_y=round(
        random.uniform(*shelf_height_limits_train) * HEIGHT, 2), fixed_ball_y=True, ball_y=ball_fixed_height,
                                                       fixed_angle=True,
                                                       angle=lambda: round(random.uniform(*angle_limits_train), 1),
                                                       time_limit=fixed_time)

    variable_shelfheight_angle_dict_test = RandomDict(render=False, fixed_shelf_y=True, shelf_y=lambda: round(
        pick_between_two_ranges(shelf_height_limits_test[0], shelf_height_limits_test[1]) * HEIGHT, 2),
                                                      fixed_ball_y=True, ball_y=ball_fixed_height, fixed_angle=True,
                                                      angle=lambda: round(random.uniform(*angle_limits_test), 1),
                                                      time_limit=fixed_time)

    # Fixed Angle Environment - variable shelf height and ball height
    variable_shelfheight_ballheight_dict_train = RandomDict(render=False, fixed_shelf_y=True, shelf_y=lambda: round(
        random.uniform(*shelf_height_limits_train) * HEIGHT, 2), fixed_ball_y=True, ball_y=lambda: round(
        random.uniform(*ball_height_limits_train) * HEIGHT, 2), fixed_angle=True, angle=fixed_angle,
                                                            time_limit=fixed_time)

    variable_shelfheight_ballheight_dict_test = RandomDict(render=False, fixed_shelf_y=True, shelf_y=lambda: round(
        pick_between_two_ranges(shelf_height_limits_test[0], shelf_height_limits_test[1]) * HEIGHT, 2),
                                                           fixed_ball_y=True, ball_y=lambda: round(
            pick_between_two_ranges(ball_height_limits_test[0], ball_height_limits_test[1]) * HEIGHT, 2),
                                                           fixed_angle=True, angle=fixed_angle, time_limit=fixed_time)

    # Variable Ball Height, Shelf Height and Angle
    all_variable_dict_train = RandomDict(render=False, fixed_shelf_y=True, shelf_y=lambda: round(
        random.uniform(*shelf_height_limits_train) * HEIGHT, 2), fixed_ball_y=True, ball_y=lambda: round(
        random.uniform(*ball_height_limits_train) * HEIGHT, 2), fixed_angle=True, angle=lambda: round(
        random.uniform(*angle_limits_train), 1), time_limit=fixed_time)

    all_variable_dict_test = RandomDict(render=False, fixed_shelf_y=True, shelf_y=lambda: round(
        pick_between_two_ranges(shelf_height_limits_test[0], shelf_height_limits_test[1]) * HEIGHT, 2),
                                        fixed_ball_y=True, ball_y=lambda: round(
            pick_between_two_ranges(ball_height_limits_test[0], ball_height_limits_test[1]) * HEIGHT, 2),
                                        fixed_angle=True,
                                        angle=lambda: round(random.uniform(*angle_limits_test), 1),
                                        time_limit=fixed_time)

    num_iters_train = 200_000
    num_iters_test = 40_000
    save_folder = 'data/shelf_bounce/'

    # Generate all files
    generate_environment_data(variable_angle_dict_train, Env, num_iters_train, save_path=save_folder + "variable_angle/",
                              verbose=True)
    generate_environment_data(variable_angle_dict_test, Env, num_iters_test, save_path=save_folder + "variable_angle/oos_",
                              verbose=True)
    generate_environment_data(variable_shelfheight_dict_train, Env, num_iters_train,
                              save_path=save_folder + "variable_shelfheight/", verbose=True)
    generate_environment_data(variable_shelfheight_dict_test, Env, num_iters_test,
                              save_path=save_folder + "variable_shelfheight/oos_", verbose=True)
    generate_environment_data(variable_ballheight_dict_train, Env, num_iters_train,
                              save_path=save_folder + "variable_ballheight/", verbose=True)
    generate_environment_data(variable_ballheight_dict_test, Env, num_iters_test,
                              save_path=save_folder + "variable_ballheight/oos_", verbose=True)
    generate_environment_data(variable_ballheight_angle_dict_train, Env, num_iters_train,
                              save_path=save_folder + "variable_ballheight_angle/",
                              verbose=True)
    generate_environment_data(variable_ballheight_angle_dict_test, Env, num_iters_test,
                              save_path=save_folder + "variable_ballheight_angle/oos_",
                              verbose=True)
    generate_environment_data(variable_shelfheight_angle_dict_train, Env, num_iters_train,
                              save_path=save_folder + "variable_shelfheight_angle/",
                              verbose=True)
    generate_environment_data(variable_shelfheight_angle_dict_test, Env, num_iters_test,
                              save_path=save_folder + "variable_shelfheight_angle/oos_",
                              verbose=True)
    generate_environment_data(variable_shelfheight_ballheight_dict_train, Env, num_iters_train,
                              save_path=save_folder + "variable_shelfheight_ballheight/", verbose=True)
    generate_environment_data(variable_shelfheight_ballheight_dict_test, Env, num_iters_test,
                              save_path=save_folder + "variable_shelfheight_ballheight/oos_", verbose=True)
    generate_environment_data(all_variable_dict_train, Env, num_iters_train,
                              save_path=save_folder + "variable_shelfheight_ballheight_angle/", verbose=True)
    generate_environment_data(all_variable_dict_test, Env, num_iters_test,
                              save_path=save_folder + "variable_shelfheight_ballheight_angle/oos_", verbose=True)

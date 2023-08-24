import random

from environments.render_constants import SCREEN_HEIGHT as HEIGHT, SCREEN_WIDTH as WIDTH
from generated_environments.multi_shelf_bounce.multi_shelf_bounce import MultiShelfBounceEnv
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

    # Set the height limit for the shelf
    shelf_height_limits_train = [0.4, 0.6]
    shelf_height_limits_test = [0.30001, 0.3999999], [0.6001, 0.699999]
    shelf_fixed_height = round(0.5 * HEIGHT, 2)
    shelf_fixed_x = WIDTH // 3

    # Angle limits
    angle_limits_train = [-45, -0.00000001]
    angle_limits_test = [-45, -0.00000001]
    fixed_angle = -30

    # Create the environment
    Env = MultiShelfBounceEnv

    # Create Environment Dictionaries

    # Variable Angle
    variable_angle_dict_train = RandomDict(render=False, fixed_shelf_y=True, shelf_y=shelf_fixed_height,
                                           fixed_shelf_x=True, shelf_x=shelf_fixed_x, fixed_angle=True,
                                           angle=lambda: round(random.uniform(*angle_limits_train), 1),
                                           time_limit=fixed_time)

    variable_angle_dict_test = RandomDict(render=False, fixed_shelf_y=True, shelf_y=shelf_fixed_height,
                                          fixed_shelf_x=True, shelf_x=shelf_fixed_x, fixed_angle=True,
                                          angle=lambda: round(random.uniform(*angle_limits_test), 1),
                                          time_limit=fixed_time)

    # Variable Shelf Height
    variable_shelfheight_dict_train = RandomDict(render=False, fixed_shelf_y=True, shelf_y=lambda: round(
        random.uniform(*shelf_height_limits_train) * HEIGHT, 2),
                                                 fixed_shelf_x=True, shelf_x=shelf_fixed_x, fixed_angle=True,
                                                 angle=fixed_angle, time_limit=fixed_time)

    variable_shelfheight_dict_test = RandomDict(render=False, fixed_shelf_y=True, shelf_y=lambda: round(
        pick_between_two_ranges(shelf_height_limits_test[0], shelf_height_limits_test[1]) * HEIGHT, 2),
                                                fixed_shelf_x=True, shelf_x=shelf_fixed_x, fixed_angle=True,
                                                angle=fixed_angle, time_limit=fixed_time)

    # Variable Time
    variable_time_dict_train = RandomDict(render=False, fixed_shelf_y=True, shelf_y=shelf_fixed_height,
                                          fixed_shelf_x=True, shelf_x=shelf_fixed_x, fixed_angle=True,
                                          angle=fixed_angle,
                                          time_limit=lambda: round(random.uniform(*time_limits_train), 1))

    variable_time_dict_test = RandomDict(render=False, fixed_shelf_y=True, shelf_y=shelf_fixed_height,
                                         fixed_shelf_x=True, shelf_x=shelf_fixed_x, fixed_angle=True,
                                         angle=fixed_angle, time_limit=lambda: round(
            pick_between_two_ranges(time_limits_test[0], time_limits_test[1]), 1))

    # Variable Angle and Time
    variable_time_angle_dict_train = RandomDict(render=False, fixed_shelf_y=True, shelf_y=shelf_fixed_height,
                                                fixed_shelf_x=True, shelf_x=shelf_fixed_x, fixed_angle=True,
                                                angle=lambda: round(random.uniform(*angle_limits_train), 1),
                                                time_limit=lambda: round(random.uniform(*time_limits_train), 1))

    variable_time_angle_dict_test = RandomDict(render=False, fixed_shelf_y=True, shelf_y=shelf_fixed_height,
                                               fixed_shelf_x=True, shelf_x=shelf_fixed_x, fixed_angle=True,
                                               angle=lambda: round(random.uniform(*angle_limits_test), 1),
                                               time_limit=lambda: round(
                                                   pick_between_two_ranges(time_limits_test[0], time_limits_test[1]),
                                                   1))

    # Variable Angle and Shelf Height
    variable_angle_shelfheight_dict_train = RandomDict(render=False, fixed_shelf_y=True, shelf_y=lambda: round(
        random.uniform(*shelf_height_limits_train) * HEIGHT, 2), fixed_shelf_x=True, shelf_x=shelf_fixed_x,
                                                       fixed_angle=True,
                                                       angle=lambda: round(random.uniform(*angle_limits_train), 1),
                                                       time_limit=fixed_time)

    variable_angle_shelfheight_dict_test = RandomDict(render=False, fixed_shelf_y=True, shelf_y=lambda: round(
        pick_between_two_ranges(shelf_height_limits_test[0], shelf_height_limits_test[1]) * HEIGHT, 2),
                                                      fixed_shelf_x=True, shelf_x=shelf_fixed_x, fixed_angle=True,
                                                      angle=lambda: round(random.uniform(*angle_limits_test), 1),
                                                      time_limit=fixed_time)

    # Variable Angle, Shelf Height and Time
    variable_time_angle_shelfheight_dict_train = RandomDict(render=False, fixed_shelf_y=True, shelf_y=lambda: round(
        random.uniform(*shelf_height_limits_train) * HEIGHT, 2), fixed_shelf_x=True, shelf_x=shelf_fixed_x,
                                                            fixed_angle=True,
                                                            angle=lambda: round(random.uniform(*angle_limits_train),
                                                                                1),
                                                            time_limit=lambda: round(
                                                                random.uniform(*time_limits_train), 1))

    variable_time_angle_shelfheight_dict_test = RandomDict(render=False, fixed_shelf_y=True, shelf_y=lambda: round(
        pick_between_two_ranges(shelf_height_limits_test[0], shelf_height_limits_test[1]) * HEIGHT, 2),
                                                           fixed_shelf_x=True, shelf_x=shelf_fixed_x, fixed_angle=True,
                                                           angle=lambda: round(random.uniform(*angle_limits_test), 1),
                                                           time_limit=lambda: round(
                                                               pick_between_two_ranges(time_limits_test[0],
                                                                                       time_limits_test[1]), 1))

    num_iters_train = 20  # 200_000
    num_iters_test = 5  # 40_000
    save_folder = 'data/multi_shelf_bounce/'

    # Generate all files
    generate_environment_data(variable_angle_dict_train, Env, num_iters_train,
                              save_path=save_folder + "variable_angle/",
                              verbose=True)
    generate_environment_data(variable_angle_dict_test, Env, num_iters_test,
                              save_path=save_folder + "variable_angle/oos_",
                              verbose=True)

    generate_environment_data(variable_shelfheight_dict_train, Env, num_iters_train,
                              save_path=save_folder + "variable_shelfheight/", verbose=True)
    generate_environment_data(variable_shelfheight_dict_test, Env, num_iters_test,
                              save_path=save_folder + "variable_shelfheight/oos_", verbose=True)

    generate_environment_data(variable_time_dict_train, Env, num_iters_train,
                              save_path=save_folder + "variable_time/", verbose=True)
    generate_environment_data(variable_time_dict_test, Env, num_iters_test,
                              save_path=save_folder + "variable_time/oos_", verbose=True)

    generate_environment_data(variable_time_angle_dict_train, Env, num_iters_train,
                              save_path=save_folder + "variable_time_angle/", verbose=True)

    generate_environment_data(variable_time_angle_dict_test, Env, num_iters_test,
                              save_path=save_folder + "variable_time_angle/oos_", verbose=True)

    generate_environment_data(variable_angle_shelfheight_dict_train, Env, num_iters_train,
                              save_path=save_folder + "variable_angle_shelfheight/", verbose=True)

    generate_environment_data(variable_angle_shelfheight_dict_test, Env, num_iters_test,
                              save_path=save_folder + "variable_angle_shelfheight/oos_", verbose=True)

    generate_environment_data(variable_time_angle_shelfheight_dict_train, Env, num_iters_train,
                              save_path=save_folder + "variable_time_angle_shelfheight/", verbose=True)

    generate_environment_data(variable_time_angle_shelfheight_dict_test, Env, num_iters_test,
                              save_path=save_folder + "variable_time_angle_shelfheight/oos_", verbose=True)

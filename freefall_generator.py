import random

from environments.render_constants import SCREEN_HEIGHT as HEIGHT
from generated_environments.freefall.freefall import FreefallEnv
from utils.env_utils import RandomDict, pick_between_two_ranges, generate_environment_data
from utils.train_utils import set_seed
from gen_examples import test_iters, train_iters

if __name__ == "__main__":
    # Set the seed for reproducibility
    set_seed(6_345_789)
    # Wilson Pickett - 634-5789 https://www.youtube.com/watch?v=TSGuaVAufV0

    # Set the time limits for training and testing - We will have two test sets.
    # One in the training time limits and one in the test time limits
    time_limits_train = [5, 15]
    time_limits_test = [1, 4.999], [15.001, 20]
    fixed_time = 10

    height_limits_train = [0.25, 0.75]
    height_limits_test = [0.01, 0.24999], [0.75001, 1]
    fixed_height = round(0.5 * HEIGHT, 2)

    Env = FreefallEnv

    # Fixed Height Environment
    # Want to create a new random value each time the object is called
    fixed_height_dict_train = RandomDict(render=False, fixed_height=True, drop_height=fixed_height,
                                         time_limit=lambda: round(random.uniform(*time_limits_train), 1))

    fixed_height_dict_test = RandomDict(render=False, fixed_height=True, drop_height=fixed_height,
                                        time_limit=lambda: round(
                                            pick_between_two_ranges(time_limits_test[0], time_limits_test[1]), 1))

    # Variable Height Environment
    variable_height_dict_train = RandomDict(render=False, fixed_height=True,
                                            drop_height=lambda: round(random.uniform(*height_limits_train) * HEIGHT, 2),
                                            time_limit=fixed_time)

    variable_height_dict_test = RandomDict(render=False, fixed_height=True,
                                           drop_height=lambda: round(pick_between_two_ranges(height_limits_test[0],
                                                                                             height_limits_test[
                                                                                                 1]) * HEIGHT, 2),
                                           time_limit=fixed_time)

    # Generate Variable Height and Time Limit
    all_variable_dict_train = RandomDict(render=False, fixed_height=True,
                                         drop_height=lambda: round(random.uniform(*height_limits_train) * HEIGHT, 2),
                                         time_limit=lambda: round(
                                             pick_between_two_ranges(time_limits_test[0], time_limits_test[1]), 1))

    all_variable_dict_test = RandomDict(render=False, fixed_height=True,
                                        drop_height=lambda: round(pick_between_two_ranges(height_limits_test[0],
                                                                                          height_limits_test[
                                                                                              1]) * HEIGHT, 2),
                                        time_limit=lambda: round(
                                            pick_between_two_ranges(time_limits_test[0], time_limits_test[1]), 1))

    save_folder = 'data/freefall/'
    num_iters_train = train_iters
    num_iters_test = test_iters
    # Generate all files
    generate_environment_data(fixed_height_dict_train, Env, num_iters_train, save_path=save_folder + "variable_time/",
                              verbose=True)
    generate_environment_data(fixed_height_dict_test, Env, num_iters_test, save_path=save_folder + "variable_time/oos_",
                              verbose=True)
    generate_environment_data(variable_height_dict_train, Env, num_iters_train, save_path=save_folder + "variable_height/",
                              verbose=True)
    generate_environment_data(variable_height_dict_test, Env, num_iters_test, save_path=save_folder + "variable_height/oos_",
                              verbose=True)
    generate_environment_data(all_variable_dict_train, Env, num_iters_train, save_path=save_folder + "variable_height_time/",
                              verbose=True)
    generate_environment_data(all_variable_dict_test, Env, num_iters_test,
                              save_path=save_folder + "variable_height_time/oos_", verbose=True)

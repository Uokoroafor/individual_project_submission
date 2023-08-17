import csv
import random
from typing import Optional, Tuple, List, Dict

from environments.render_constants import SCREEN_HEIGHT as HEIGHT
from generated_environments.shelf_bounce.shelf_bounce import ShelfBounceEnv
from utils.env_utils import RandomDict, pick_between_two_ranges, generate_environment_data

from utils.train_utils import set_seed


# def main(env_dict: Dict, iters: int, save_path: Optional[str] = None, verbose: bool = False) -> Tuple[
#     List, List, List, List]:
#     """ Generates a number of freefall environments and saves the logs to a csv and txt file
#     Args:
#         env_dict (Dict): A dictionary of the environment parameters
#         iters (int): The number of environments to generate
#         save_path (Optional[str], optional): The path to save the logs to. Defaults to None.
#         verbose (bool, optional): Whether to print the number of unique logs generated. Defaults to False.
# 
#     Returns:
#         Tuple[List, List, List, List]: The numerical logs, text logs, minimal texts and descriptive texts
#         """
# 
#     numerical_logs = []
#     text_logs = []
#     minimal_texts = []
#     descriptive_texts = []
# 
#     for i in range(iters):
#         # fix the values in the env_dict
#         env_dict_copy = env_dict.copy()
#         for key, value in env_dict_copy.items():
#             if callable(value):
#                 env_dict_copy[key] = value()
# 
#         env = ShelfBounceEnv(**env_dict_copy)
#         numerical_logs.append(env.numerical_log)
#         text_logs.append(env.text_log)
#         minimal_texts.append(env.make_minimal_text())
#         descriptive_texts.append(env.make_descriptive_text())
# 
#     # Remove duplicates from one of the logs. Get the unique indices and use them to get the unique logs
#     # This is done because the numerical log is a dictionary and the text log is a list of tuples
# 
#     # Find unique indices of the numerical log
#     unique_indices = []
#     for i, log in enumerate(numerical_logs):
#         if log not in numerical_logs[:i]:
#             unique_indices.append(i)
#     # Get the unique logs
#     numerical_logs = [numerical_logs[i] for i in unique_indices]
#     text_logs = [text_logs[i] for i in unique_indices]
#     minimal_texts = [minimal_texts[i] for i in unique_indices]
#     descriptive_texts = [descriptive_texts[i] for i in unique_indices]
# 
# 
#     # Reconvert numerical logs to dicts
#     numerical_logs = [dict(log) for log in numerical_logs]
#     text_logs = [' '.join(list(log)) for log in text_logs]
# 
#     if save_path is not None:
#         # save numerical log as csv and the rest as txt
#         # All the fields in the numerical log are the same for each entry so save it as a csv
#         with open(save_path + "numerical_logs.csv", "w") as f:
#             writer = csv.DictWriter(f, fieldnames=numerical_logs[0].keys())
#             writer.writeheader()
#             for log in numerical_logs:
#                 writer.writerow(log)
# 
#         with open(save_path + "text_log.txt", "w") as f:
#             for log in text_logs:
#                 f.write(str(log) + "\n")
#         with open(save_path + "minimal_text.txt", "w") as f:
#             for text in minimal_texts:
#                 f.write(str(text) + "\n")
#         with open(save_path + "descriptive_text.txt", "w") as f:
#             for text in descriptive_texts:
#                 f.write(str(text) + "\n")
#     if len(numerical_logs) != len(text_logs) or len(numerical_logs) != len(minimal_texts) or len(
#             numerical_logs) != len(descriptive_texts):
#         print("Warning: The number of numerical logs, text logs, minimal texts and descriptive texts are not equal")
#         print(f"Number of unique numerical logs: {len(numerical_logs)}")
#         print(f"Number of unique text logs: {len(text_logs)}")
#         print(f"Number of unique minimal texts: {len(minimal_texts)}")
#         print(f"Number of unique descriptive texts: {len(descriptive_texts)}")
# 
#     if verbose:
#         print(f"Number of unique elements: {len(numerical_logs):,}")
#         print(f"Files saved to {save_path}\n")
# 
#     return numerical_logs, text_logs, minimal_texts, descriptive_texts


if __name__ == "__main__":
    # Set the seed for reproducibility
    set_seed(6_345_789)
    # Wilson Pickett - 634-5789 https://www.youtube.com/watch?v=TSGuaVAufV0

    # Set the time limits for training and testing - We will have two test sets.
    # One in the training time limits and one in the test time limits
    time_limits_train = [5, 15]
    time_limits_test = [1, 4.999], [15.001, 20]
    fixed_time = 10

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
                                           angle=lambda: round(random.uniform(*angle_limits_train), 1), time_limit=fixed_time)

    variable_angle_dict_test = RandomDict(render=False, fixed_shelf_y=True, shelf_y=shelf_fixed_height,
                                          fixed_ball_y=True, ball_y=ball_fixed_height, fixed_angle=True,
                                          angle=lambda: round(random.uniform(*angle_limits_test), 1), time_limit=fixed_time)

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
            random.uniform(*ball_height_limits_train) * HEIGHT, 2), fixed_angle=True, angle=fixed_angle, time_limit=fixed_time)

    variable_ballheight_dict_test = RandomDict(render=False, fixed_shelf_y=True, shelf_y=shelf_fixed_height,
                                                  fixed_ball_y=True, ball_y=lambda: round(
            pick_between_two_ranges(ball_height_limits_test[0], ball_height_limits_test[1]) * HEIGHT, 2),
                                                    fixed_angle=True, angle=fixed_angle, time_limit=fixed_time)


    # Fixed Shelf Height Environment - variable ball height and angle
    variable_ballheight_angle_dict_train = RandomDict(render=False, fixed_shelf_y=True, shelf_y=lambda: round(
        random.uniform(*shelf_height_limits_train) * HEIGHT, 2), fixed_ball_y=True, ball_y=ball_fixed_height,
                                                      fixed_angle=True,
                                                      angle=lambda: round(random.uniform(*angle_limits_train), 1), time_limit=fixed_time)

    variable_ballheight_angle_dict_test = RandomDict(render=False, fixed_shelf_y=True, shelf_y=lambda: round(
        pick_between_two_ranges(shelf_height_limits_test[0], shelf_height_limits_test[1]) * HEIGHT, 2),
                                                     fixed_ball_y=True, ball_y=ball_fixed_height, fixed_angle=True,
                                                     angle=lambda: round(random.uniform(*angle_limits_test), 1), time_limit=fixed_time)

    # Fixed Ball Height Environment - variable shelf height and angle
    variable_shelfheight_angle_dict_train = RandomDict(render=False, fixed_shelf_y=True, shelf_y=round(
        random.uniform(*shelf_height_limits_train) * HEIGHT, 2), fixed_ball_y=True, ball_y=ball_fixed_height,
                                                         fixed_angle=True,
                                                            angle=lambda: round(random.uniform(*angle_limits_train), 1), time_limit=fixed_time)

    variable_shelfheight_angle_dict_test = RandomDict(render=False, fixed_shelf_y=True, shelf_y=lambda: round(
        pick_between_two_ranges(shelf_height_limits_test[0], shelf_height_limits_test[1]) * HEIGHT, 2),
                                                         fixed_ball_y=True, ball_y=ball_fixed_height, fixed_angle=True,
                                                            angle=lambda: round(random.uniform(*angle_limits_test), 1), time_limit=fixed_time)

    # Fixed Angle Environment - variable shelf height and ball height
    variable_shelfheight_ballheight_dict_train = RandomDict(render=False, fixed_shelf_y=True, shelf_y=lambda: round(
        random.uniform(*shelf_height_limits_train) * HEIGHT, 2), fixed_ball_y=True, ball_y=lambda: round(
        random.uniform(*ball_height_limits_train) * HEIGHT, 2), fixed_angle=True, angle=fixed_angle, time_limit=fixed_time)

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
                                        angle=lambda: round(random.uniform(*angle_limits_test), 1), time_limit=fixed_time)

    num_iters = 5
    save_folder = '../../data/shelf_bounce/'

    # Generate all files
    generate_environment_data(variable_angle_dict_train,Env, num_iters, save_path=save_folder + "variable_angle/", verbose=True)
    generate_environment_data(variable_angle_dict_test,Env, num_iters, save_path=save_folder + "variable_angle/oos_", verbose=True)
    generate_environment_data(variable_shelfheight_dict_train,Env, num_iters, save_path=save_folder + "variable_shelfheight/", verbose=True)
    generate_environment_data(variable_shelfheight_dict_test,Env, num_iters, save_path=save_folder + "variable_shelfheight/oos_", verbose=True)
    generate_environment_data(variable_ballheight_dict_train,Env, num_iters, save_path=save_folder + "variable_ballheight/", verbose=True)
    generate_environment_data(variable_ballheight_dict_test,Env, num_iters, save_path=save_folder + "variable_ballheight/oos_", verbose=True)
    generate_environment_data(variable_ballheight_angle_dict_train,Env, num_iters, save_path=save_folder + "variable_ballheight_angle/",
            verbose=True)
    generate_environment_data(variable_ballheight_angle_dict_test,Env, num_iters, save_path=save_folder + "variable_ballheight_angle/oos_",
            verbose=True)
    generate_environment_data(variable_shelfheight_angle_dict_train,Env, num_iters, save_path=save_folder + "variable_shelfheight_angle/",
            verbose=True)
    generate_environment_data(variable_shelfheight_angle_dict_test,Env, num_iters, save_path=save_folder + "variable_shelfheight_angle/oos_",
            verbose=True)
    generate_environment_data(variable_shelfheight_ballheight_dict_train,Env, num_iters,
            save_path=save_folder + "variable_shelfheight_ballheight/", verbose=True)
    generate_environment_data(variable_shelfheight_ballheight_dict_test,Env, num_iters,
            save_path=save_folder + "variable_shelfheight_ballheight/oos_", verbose=True)
    generate_environment_data(all_variable_dict_train,Env, num_iters, save_path=save_folder + "variable_shelfheight_ballheight_angle/",verbose=True)
    generate_environment_data(all_variable_dict_test,Env, num_iters, save_path=save_folder + "variable_shelfheight_ballheight_angle/oos_",verbose=True)


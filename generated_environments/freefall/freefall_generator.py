import csv
import random
from typing import Optional, Tuple, List, Dict

from environments.render_constants import SCREEN_HEIGHT as HEIGHT
from generated_environments.freefall.freefall import FreefallEnv

from utils.train_utils import set_seed


def main(env_dict: Dict, iters: int, save_path: Optional[str] = None, verbose: bool = False) -> Tuple[
    List, List, List, List]:
    """ Generates a number of freefall environments and saves the logs to a csv and txt file
    Args:
        env_dict (Dict): A dictionary of the environment parameters
        iters (int): The number of environments to generate
        save_path (Optional[str], optional): The path to save the logs to. Defaults to None.
        verbose (bool, optional): Whether to print the number of unique logs generated. Defaults to False.
        Returns:
            """

    numerical_logs = []
    text_logs = []
    minimal_texts = []
    descriptive_texts = []

    for i in range(iters):
        # fix the values in the env_dict
        env_dict_copy = env_dict.copy()
        for key, value in env_dict_copy.items():
            if callable(value):
                env_dict_copy[key] = value()

        env = FreefallEnv(**env_dict_copy)
        numerical_logs.append(env.numerical_log)
        text_logs.append(env.text_log)
        minimal_texts.append(env.make_minimal_text())
        descriptive_texts.append(env.make_descriptive_text())

    # Remove duplicates
    numerical_logs = list({tuple(log.items()) for log in numerical_logs})
    text_logs = list({tuple(log) for log in text_logs})
    minimal_texts = list({text for text in minimal_texts})
    descriptive_texts = list({text for text in descriptive_texts})

    # Reconvert numerical logs to dicts
    numerical_logs = [dict(log) for log in numerical_logs]
    text_logs = [' '.join(list(log)) for log in text_logs]

    if save_path is not None:
        # save numerical log as csv and the rest as txt
        # All the fields in the numerical log are the same for each entry so save it as a csv
        with open(save_path + "numerical_logs.csv", "w") as f:
            writer = csv.DictWriter(f, fieldnames=numerical_logs[0].keys())
            writer.writeheader()
            for log in numerical_logs:
                writer.writerow(log)

        with open(save_path + "text_log.txt", "w") as f:
            for log in text_logs:
                f.write(str(log) + "\n")
        with open(save_path + "minimal_text.txt", "w") as f:
            for text in minimal_texts:
                f.write(str(text) + "\n")
        with open(save_path + "descriptive_text.txt", "w") as f:
            for text in descriptive_texts:
                f.write(str(text) + "\n")
    if len(numerical_logs) != len(text_logs) or len(numerical_logs) != len(minimal_texts) or len(
            numerical_logs) != len(descriptive_texts):
        print("Warning: The number of numerical logs, text logs, minimal texts and descriptive texts are not equal")
        print(f"Number of unique numerical logs: {len(numerical_logs)}")
        print(f"Number of unique text logs: {len(text_logs)}")
        print(f"Number of unique minimal texts: {len(minimal_texts)}")
        print(f"Number of unique descriptive texts: {len(descriptive_texts)}")

    if verbose:
        print(f"Number of unique elements: {len(numerical_logs):,}")
        print(f"Files saved to {save_path}\n")

    return numerical_logs, text_logs, minimal_texts, descriptive_texts


if __name__ == "__main__":
    # Set the seed for reproducibility
    set_seed(6_345_789)
    # Wilson Pickett - 634-5789 https://www.youtube.com/watch?v=TSGuaVAufV0

    # Set the time limits for training and testing - We will have two test sets.
    # One in the training time limits and one in the test time limits

    time_limits_train = [1, 20]
    time_limits_test = [21, 25]
    fixed_time_limit = 10

    height_limits_train = [0.5, 1]
    height_limits_test = [0.25, .49999]
    fixed_height_limit = round(0.75 * HEIGHT, 2)


    class RandomDict(dict):
        def __getitem__(self, key):
            return super().__getitem__(key)()


    # Fixed Height Environment
    # Want to create a new random value each time the object is called
    fixed_height_dict_train = RandomDict(render=False, fixed_height=True, drop_height=fixed_height_limit,
                                         time_limit=lambda: round(random.uniform(*time_limits_train), 1))

    fixed_height_dict_test = RandomDict(render=False, fixed_height=True, drop_height=fixed_height_limit,
                                        time_limit=lambda: round(random.uniform(*time_limits_test), 1))

    # Variable Height Environment
    variable_height_dict_train = RandomDict(render=False, fixed_height=True,
                                            drop_height=lambda: round(random.uniform(*height_limits_train) * HEIGHT, 2),
                                            time_limit=fixed_time_limit)

    variable_height_dict_test = RandomDict(render=False, fixed_height=True,
                                           drop_height=lambda: round(random.uniform(*height_limits_test) * HEIGHT, 2),
                                           time_limit=fixed_time_limit)

    # Generate Variable Height and Time Limit
    all_variable_dict_train = RandomDict(render=False, fixed_height=True,
                                         drop_height=lambda: round(random.uniform(*height_limits_train) * HEIGHT, 2),
                                         time_limit=lambda: lambda: round(random.uniform(*time_limits_train), 1))

    all_variable_dict_test = RandomDict(render=False, fixed_height=True,
                                        drop_height=lambda: round(random.uniform(*height_limits_test) * HEIGHT, 2),
                                        time_limit=lambda: lambda: round(random.uniform(*time_limits_test), 1))

    save_folder = '../../data/freefall/'
    num_iters = 200_000
    # Generate all files
    main(fixed_height_dict_train, num_iters, save_path=save_folder + "variable_time/", verbose=True)
    main(fixed_height_dict_test, num_iters, save_path=save_folder + "variable_time/oos_", verbose=True)
    main(variable_height_dict_train, num_iters, save_path=save_folder + "variable_height/", verbose=True)
    main(variable_height_dict_test, num_iters, save_path=save_folder + "variable_height/oos_", verbose=True)
    main(all_variable_dict_train, num_iters, save_path=save_folder + "variable_height_time/", verbose=True)
    main(all_variable_dict_test, num_iters, save_path=save_folder + "variable_height_time/oos_", verbose=True)

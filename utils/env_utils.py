import random
from typing import List, Dict, Optional, Callable, Tuple
import csv


class RandomDict(dict):
    """A dictionary that returns a new random value each time the key is called."""

    def __getitem__(self, key):
        return super().__getitem__(key)()


def pick_between_two_ranges(range1: List[float], range2: List[float]) -> float:
    """Picks a random value between two ranges.

    Args:
        range1 (List[float, float]): The first range.
        range2 (List[float, float]): The second range.

    Returns:
        float: A random value between the two ranges.
    """
    if random.random() < 0.5:
        return random.uniform(*range1)
    else:
        return random.uniform(*range2)


def generate_environment_data(env_dict: Dict, env_: Callable, iters: int, save_path: Optional[str] = None, verbose: bool = False) -> Tuple[
    List, List, List, List]:
    """ Generates a number of freefall environments and saves the logs to a csv and txt file
    Args:
        env_dict (Dict): A dictionary of the environment parameters
        env_ (Callable): The environment class
        iters (int): The number of environments to generate
        save_path (Optional[str], optional): The path to save the logs to. Defaults to None.
        verbose (bool, optional): Whether to print the number of unique logs generated. Defaults to False.

    Returns:
        Tuple[List, List, List, List]: The numerical logs, text logs, minimal texts and descriptive texts
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

        env = env_(**env_dict_copy)
        numerical_logs.append(env.numerical_log)
        text_logs.append(env.text_log)
        minimal_texts.append(env.make_minimal_text())
        descriptive_texts.append(env.make_descriptive_text())

    # Remove duplicates from one of the logs. Get the unique indices and use them to get the unique logs
    # This is done because the numerical log is a dictionary and the text log is a list of tuples

    # Find unique indices of the numerical log
    unique_indices = []
    for i, log in enumerate(numerical_logs):
        if log not in numerical_logs[:i]:
            unique_indices.append(i)
    # Get the unique logs
    numerical_logs = [numerical_logs[i] for i in unique_indices]
    text_logs = [text_logs[i] for i in unique_indices]
    minimal_texts = [minimal_texts[i] for i in unique_indices]
    descriptive_texts = [descriptive_texts[i] for i in unique_indices]

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

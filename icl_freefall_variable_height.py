import openai
import os
from dotenv import load_dotenv
from utils.icl_utils import ContextGenerator, LLMTrainer
from utils.train_utils import set_seed
from utils.data_utils import read_in_data
import pandas as pd
import torch
from torch import nn
import random
from models.gpt.models.do_transformer import DecodeOnlyTransformer
from utils.basic_tokeniser import BasicTokeniser
from utils.bpe import BPE
from utils.data_utils import read_in_data, make_data_loaders, make_data_loader
from utils.file_utils import load_config
from utils.train_utils import PhysicalTrainer, set_seed
from utils.plot_utils import plot_predictions, plot_errors

# Load variables from .env
load_dotenv()

openai.api_key = os.getenv("OPENAI_API_KEY")

# Set the seed
seed = 6_345_789
set_seed(seed)

data_folder = "data/freefall/variable_height/"
file_path = "descriptive_text.txt"  # Update this to the file containing the data
function_name = "Freefall Environment"  # Update this to the name of the dataset being trained (or the name of the function)

train_indices_path = "train_indices.csv"
val_indices_path = "val_indices.csv"
test_indices_path = "test_indices.csv"

oos_test_data_path = f"oos_descriptive_text.txt"

# Get train and test indices and convert to lists
train_indices = pd.read_csv(data_folder + train_indices_path).values.flatten().tolist()
val_indices = pd.read_csv(data_folder + val_indices_path).values.flatten().tolist()
test_indices = pd.read_csv(data_folder + test_indices_path).values.flatten().tolist()


# Load the datasets
data = read_in_data(data_folder + file_path, make_dict=False)
oos_data = read_in_data(data_folder + oos_test_data_path, make_dict=False)

# Context Generator Args
line_delimiter = "\n"
ans_delimiter = " ans: Ball is at y="
question = ', what is the value of y?'
answer_str = ' ans: '

model_name = "text-davinci-003"

# Create the context generator
context_generator = ContextGenerator(data=data, line_delimiter=line_delimiter, ans_delimiter=ans_delimiter,
                                     question=question, answer_str=answer_str,
                                     train_indices=train_indices, val_indices=val_indices, test_indices=test_indices)

# Create the trainer
trainer = LLMTrainer(cg=context_generator)

num_shots = list(range(1, 5))
errors = []
oos_errors = []

for num_shot in num_shots:
    # Evaluate the model on test data
    mse, _, _ = trainer.evaluate(num_shots=num_shot, model_name="text-davinci-003", plotting=False, save_preds=True,
                                 verbose=True)

    errors.append(mse)

for num_shot in num_shots:
    # Evaluate the model on test data

    oos_mse, _, _ = trainer.evaluate(num_shots=num_shot, model_name="text-davinci-003", plotting=False, save_preds=True,
                                     verbose=True, test_data=oos_data)

    oos_errors.append(oos_mse)

plot_errors(errors, num_shots, model_name='text-davinci-003',
            saved_path=f'{trainer.path}/training_logs/errors_vs_shots.png')
plot_errors(oos_errors, num_shots, model_name='text-davinci-003',
            saved_path=f'{trainer.path}/training_logs/oos_errors_vs_shots.png')


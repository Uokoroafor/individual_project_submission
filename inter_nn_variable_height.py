import pandas as pd
import torch
from torch import nn
import random
from models.nn_models.nn_model import Net
from utils.physical_dataset import PhysicalDataset
from utils.file_utils import load_config
from utils.train_utils import NNTrainer, set_seed
from utils.logging_utils import TrainingLogger

# Load the training hyperparameters from the txt file
training_hyperparams = load_config("config_structure_nn.txt")

# Set the random seed for reproducibility
set_seed(6_345_789)
# Wilson Pickett - 634-5789 https://www.youtube.com/watch?v=TSGuaVAufV0

# Create the logger
batch_logger = TrainingLogger("inter_logs_num.txt", verbose=False)

print("Using device: ", training_hyperparams["device"])
device = training_hyperparams["device"]
batch_size = training_hyperparams["batch_size"]
eval_iters = training_hyperparams["eval_every"]
max_iters = training_hyperparams["epochs"]
lr = training_hyperparams["learning_rate"]

data_folder = "data/freefall/variable_height/"
function_name = "Freefall Environment"  # Update this to the name of the dataset being trained (or the name of the function)
data_path = "numerical_logs.csv"
train_indices_path = "train_indices.csv"
val_indices_path = "val_indices.csv"
test_indices_path = "test_indices.csv"

oos_test_data_path = "oos_numerical_logs.csv"

data_amount = 200_000

num_input_features = 3
hidden_layers = [32, 32, 32]
num_output_features = 1

logging_intro = f"Training on {function_name} on {data_folder} data."

# Read in the data
data = pd.read_csv(data_folder + data_path)

train_indices = pd.read_csv(data_folder + train_indices_path).values.flatten()
val_indices = pd.read_csv(data_folder + val_indices_path).values.flatten()
# test_indices = pd.read_csv(data_folder + test_indices_path).values.flatten()

# Take subset of training data if required
# train_indices = train_indices[:int(data_amount)]

# # Scale the validation data by the same amount as the training data
# val_indices = val_indices[:int(data_amount * len(val_indices) / len(train_indices))]
print(len(train_indices))
print(len(val_indices))

train_data = data.iloc[train_indices]
val_data = data.iloc[val_indices]

print(len(train_data))
print(len(val_data))

# reset indices to 0 to len(data)
train_data.index = [None] * len(train_data)
val_data.index = [None] * len(val_data)


float_col = train_data.iloc[:, -1].astype(float)

y_min = float_col.min()
y_max = float_col.max()
y_mid = (y_max + y_min) / 2

# Take out the middle 20% of y values
y_range = y_max - y_min
y_min = round(y_mid - y_range * 0.025, 2)
y_max = round(y_mid + y_range * 0.025, 2)

print(f"y_min: {y_min}, y_max: {y_max}")

# Set the indices to range(0, len(data))
train_data.index = list(range(len(train_data)))
val_data.index = list(range(len(val_data)))

# Get the indices of the middle 20% of y values
test_indices = float_col[(float_col > y_min) & (float_col < y_max)].index
train_indices = float_col[(float_col <= y_min) | (float_col >= y_max)].index

print(f'len(test_indices): {test_indices[:10]}')
print(f'len(train_indices): {len(train_indices)}')




# Get the test data
test_data = train_data.iloc[test_indices]
train_data = train_data.iloc[train_indices]

print(f'len(test_data): {len(test_data)}')

# Reset the indices
train_data.index = [None] * len(train_data)
val_data.index = [None] * len(val_data)
test_data.index = [None] * len(test_data)

# Create the datasets
train_data = PhysicalDataset(train_data)
val_data = PhysicalDataset(val_data)
test_data = PhysicalDataset(test_data)


train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_data, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=True)

# Create the model, loss function and optimiser
loss_fn = nn.MSELoss()

model = Net(input_size=num_input_features, hidden_sizes=hidden_layers, output_size=num_output_features,
            activation=nn.ReLU())
optimiser = torch.optim.Adam(model.parameters(), lr=lr)
scheduler = None

device = torch.device(training_hyperparams["device"])

# Move the model and loss function to the device
model = model.to(device)
loss_fn = loss_fn.to(device)

trainer = NNTrainer(model=model, loss_fn=loss_fn, optimiser=optimiser, scheduler=scheduler)

model, _, _ = trainer.train(
    train_dataloader=train_loader,
    val_dataloader=val_loader,
    save_model=True,
    plotting=True,
    verbose=True,
    early_stopping=True,
    early_stopping_patience=10,
    logging_intro=logging_intro,
    epochs=max_iters,
)

test_error = trainer.evaluate(test_loader)
batch_logger.log_info(f"Training log is saved at {trainer.path}")
batch_logger.log_info(f"{function_name} on {data_folder} data with {len(train_data):,} training examples, {len(test_data):,} test examples ")

batch_logger.log_info(f"Test loss: {test_error:.4f} for values between {y_min:.2f} and {y_max:.2f}")

print("Finished_________________________________")

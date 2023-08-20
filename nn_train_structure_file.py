import pandas as pd
import torch
from torch import nn
import random
from models.nn_models.nn_model import Net
from utils.physical_dataset import PhysicalDataset
from utils.file_utils import load_config
from utils.train_utils import NNTrainer, set_seed

# Load the training hyperparameters from the txt file
training_hyperparams = load_config("config_structure_nn.txt")

# Set the random seed for reproducibility
set_seed(6_345_789)
# Wilson Pickett - 634-5789 https://www.youtube.com/watch?v=TSGuaVAufV0


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

# pooling = "cls"  # 'max', 'mean', 'cls', 'none' use none for text generation
data_portions = [0.01, 0.1, 0.25, 0.5, 1]

num_input_features = 3
hidden_layers = [32, 32]
num_output_features = 1

for data_portion in data_portions:
    logging_intro = f"Training on {function_name} on {data_folder} data."

    # Read in the data
    data = pd.read_csv(data_folder + data_path)

    train_indices = pd.read_csv(data_folder + train_indices_path).values.flatten()
    val_indices = pd.read_csv(data_folder + val_indices_path).values.flatten()
    test_indices = pd.read_csv(data_folder + test_indices_path).values.flatten()

    # Take subset of training data if required
    train_indices = train_indices[:int(data_portion * len(train_indices))]
    val_indices = val_indices[:int(data_portion * len(val_indices))]

    # Create the datasets
    train_data = PhysicalDataset(data.iloc[train_indices])
    val_data = PhysicalDataset(data.iloc[val_indices])
    test_data = PhysicalDataset(data.iloc[test_indices])

    oos_test_data = PhysicalDataset(pd.read_csv(data_folder + oos_test_data_path))

    print(f"Size of train dataset: {len(train_data)}")

    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_data, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=True)
    oos_test_loader = torch.utils.data.DataLoader(oos_test_data, batch_size=batch_size, shuffle=True)

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
        early_stopping_patience=20,
        logging_intro=logging_intro,
        epochs=max_iters,
    )

    test_error = trainer.evaluate(test_loader)
    print(f"Test error: {test_error: ,.4f} for data portion {data_portion:2%}")

    oos_test_error = trainer.evaluate(oos_test_loader)
    print(f"OOS Test error: {oos_test_error: ,.4f} for data portion {data_portion:2%}")

    print("Finished_________________________________")

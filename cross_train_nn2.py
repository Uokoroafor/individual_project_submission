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
batch_logger = TrainingLogger("nn_training_logs_3.txt", verbose=False)

print("Using device: ", training_hyperparams["device"])
device = training_hyperparams["device"]
batch_size = training_hyperparams["batch_size"]
eval_iters = training_hyperparams["eval_every"]
max_iters = training_hyperparams["epochs"]
lr = training_hyperparams["learning_rate"]

test_folders = ["variable_shelfheight_ballheight_angle"]
train_folders = [["variable_ballheight_angle", "variable_shelfheight_angle"]]


for train_folder, test_folder in zip(train_folders, test_folders):
    try:
        data_folders = [f"data/shelf_bounce/{folder}/" for folder in train_folder]

        test_folder = f"data/shelf_bounce/{test_folder}/"
        # data_folder = f"data/shelf_bounce/{folder}/"
        function_name = "Shelf Bounce Environment"  # Update this to the name of the dataset being trained (or the name of the function)
        data_path = "numerical_logs.csv"
        train_indices_path = "train_indices.csv"
        val_indices_path = "val_indices.csv"
        test_indices_path = "test_indices.csv"

        oos_test_data_path = "oos_numerical_logs.csv"

        data_amount = 200_000

        num_input_features = 10
        hidden_layers = [64, 64, 64]
        num_output_features = 1

        stop_training = False

        # for data_amount in data_amounts:
        #     if not stop_training:
        data_folder_str = ", ".join(train_folder)
        logging_intro = f"Training on {function_name} on {data_folder_str} data."
        train_data = []
        val_data = []
        for data_folder in data_folders:
            # Read in the data
            data = pd.read_csv(data_folder + data_path)

            train_indices = pd.read_csv(data_folder + train_indices_path).values.flatten()
            val_indices = pd.read_csv(data_folder + val_indices_path).values.flatten()
            test_indices = pd.read_csv(data_folder + test_indices_path).values.flatten()

            # Take subset of training data if required
            train_indices = train_indices[:int(data_amount)]

            # # Scale the validation data by the same amount as the training data
            # val_indices = val_indices[:int(data_amount * len(val_indices) / len(train_indices))]
            train_data.append(data.iloc[train_indices])
            val_data.append(data.iloc[val_indices])



        train_data = pd.concat(train_data)
        val_data = pd.concat(val_data)

        # Create the test data
        data = pd.read_csv(test_folder + data_path)
        test_indices = pd.read_csv(test_folder + test_indices_path).values.flatten()
        test_data = data.iloc[test_indices]

        # Create the datasets
        train_data = PhysicalDataset(train_data)
        val_data = PhysicalDataset(val_data)
        test_data = PhysicalDataset(test_data)

        train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
        val_loader = torch.utils.data.DataLoader(val_data, batch_size=batch_size, shuffle=True)
        test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=True)
        # oos_test_loader = torch.utils.data.DataLoader(oos_test_data, batch_size=batch_size, shuffle=True)

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
        batch_logger.log_info(f"{function_name} on {data_folder_str} data with {len(train_data):,} training examples")
        print(f"Test error: {test_error: ,.4f} for {len(train_data):,} training examples")

        batch_logger.log_info(f"Test error: {test_error: ,.4f}")

        print("Finished_________________________________")
    except Exception as e:
        batch_logger.log_info(f"Error during Training: {e}")
        continue

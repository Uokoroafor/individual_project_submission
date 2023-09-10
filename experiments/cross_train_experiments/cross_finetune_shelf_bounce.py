import pandas as pd
import torch
from torch import nn
from transformers import BertForSequenceClassification, BertConfig
from transformers import BertTokenizer
from transformers import get_linear_schedule_with_warmup

from utils.finetune_utils import FineTuneTrainer
from utils.finetune_utils import (
    make_finetune_dataloaders,
    freeze_bert_layers,
    count_frozen_bert_layers,
)
from utils.train_utils import set_seed
from utils.logging_utils import TrainingLogger

# Set the random seed for reproducibility
set_seed(6_345_789)
# Wilson Pickett - 634-5789 https://www.youtube.com/watch?v=TSGuaVAufV0

# Create the logger
batch_logger = TrainingLogger("cross_training_logs_finetune.txt", verbose=False)

# Preallocate variables defined in set_training_hyperparameters
training_params = dict(
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    epochs=500,
    batch_size=32,
    eval_every=1,
    eval_iters=1,
    max_seq_len=128,
    save_every=10000,
)

learning_params = dict(lr=1e-4, eps=1e-8)

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
max_length = training_params["max_seq_len"]
batch_size = training_params["batch_size"]

folder_loc = "data/shelf_bounce/"
function_names = [
    "variable_angle",
    "variable_ballheight",
    "variable_ballheight_angle",
    "variable_shelfheight",
    "variable_shelfheight_angle",
    "variable_shelfheight_ballheight",
    "variable_shelfheight_ballheight_angle",
]

folders = [
    "variable_angle",
    "variable_ballheight",
    "variable_ballheight_angle",
    "variable_shelfheight",
    "variable_shelfheight_angle",
    "variable_shelfheight_ballheight",
    "variable_shelfheight_ballheight_angle",
]

# Want to train on a concatenated set of two variables and train on the combined data
test_folders = [
    "variable_ballheight_angle",
    "variable_shelfheight_angle",
    "variable_shelfheight_ballheight",
    "variable_shelfheight_ballheight_angle",
]
# split each folder name by _, then train on the first two variables

train_folders = []

for x in test_folders:
    train_list = []
    x_ = x.split("_")[1:]
    # add variable_ to the start of each and append to train_folders
    for y in x_:
        train_list.append("variable_" + y)
    train_folders.append(train_list)

for train_folder, test_folder in zip(train_folders, test_folders):
    train_data = [
        pd.read_csv(folder_loc + folder + "/train_data.csv") for folder in train_folder
    ]
    val_data = [
        pd.read_csv(folder_loc + folder + "/val_data.csv") for folder in train_folder
    ]
    test_data = pd.read_csv(folder_loc + test_folder + "/test_data.csv")

    # Concatenate the dataframes
    train_data = pd.concat(train_data, ignore_index=True)
    val_data = pd.concat(val_data, ignore_index=True)

    # No OOStest data for this experiment and no concatenation of the test data is required

    model_name = "bert-base-uncased"

    # Create the data loaders
    train_dataloader, val_dataloader, test_dataloader = make_finetune_dataloaders(
        train_data=train_data,
        val_data=val_data,
        test_data=test_data,
        tokenizer=tokenizer,
        max_length=max_length,
        batch_size=batch_size,
        output_type="num",
    )

    config = BertConfig.from_pretrained(model_name)
    config.num_labels = 1
    output_type = "num"

    total_layers = config.num_hidden_layers
    flayers = 12

    try:
        model = BertForSequenceClassification(config)

        batch_logger.log_info(f"Model has Total number of layers: {total_layers}")

        model = freeze_bert_layers(model, flayers)
        batch_logger.log_info(
            f"Model has {count_frozen_bert_layers(model)} frozen layers"
        )

        # Unfreeze the classification layer
        for param in model.classifier.parameters():
            param.requires_grad = True

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        model.to(device)

        optimizer = torch.optim.Adam(
            model.parameters(), lr=learning_params["lr"], eps=learning_params["eps"]
        )
        epochs = training_params["epochs"]
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=0,
            num_training_steps=len(train_dataloader) * epochs,
        )

        loss_fn = nn.MSELoss()

        BertTrainer = FineTuneTrainer(
            model=model,
            optimiser=optimizer,
            scheduler=scheduler,
            loss_fn=loss_fn,
            training_hyperparameters=training_params,
            tokenizer=tokenizer,
        )

        batch_logger.log_info(f"Training log is saved at {BertTrainer.path}")

        model, _, _ = BertTrainer.train(
            train_dataloader=train_dataloader,
            val_dataloader=val_dataloader,
            save_model=True,
            plotting=True,
            verbose=True,
            early_stopping=True,
            early_stopping_patience=4,
        )

        test_loss = BertTrainer.log_numerical_outputs(
            test_dataloader, output_type=output_type
        )

        batch_logger.log_info(
            f"{test_folder} data with {output_type} output, {len(train_data)} training examples "
            f"and {flayers} frozen layers"
        )
        batch_logger.log_info(f"Test loss: {test_loss:,.4f}")

    except Exception as e:
        batch_logger.log_warning(f"Error during Training: {e}")
        continue

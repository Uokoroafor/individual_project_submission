import pandas as pd
import torch
from torch import nn
from transformers import BertForSequenceClassification, BertConfig
from transformers import BertTokenizer
from transformers import get_linear_schedule_with_warmup

from utils.finetune_utils import FineTuneTrainer
from utils.finetune_utils import (
    make_finetune_dataloaders,
    make_finetune_dataloader,
    freeze_bert_layers,
    count_frozen_bert_layers,
)
from utils.train_utils import set_seed
from utils.logging_utils import TrainingLogger

# Set the random seed for reproducibility
set_seed(6_345_789)
# Wilson Pickett - 634-5789 https://www.youtube.com/watch?v=TSGuaVAufV0

# Create the logger
batch_logger = TrainingLogger("../../inter_logs_num.txt", verbose=False)

# Preallocate variables defined in set_training_hyperparameters
training_params = dict(
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    epochs=500,
    batch_size=32,
    eval_every=5,
    eval_iters=1,
    max_seq_len=64,
    save_every=10000,
)

learning_params = dict(lr=5e-4, eps=1e-8)

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
max_length = training_params["max_seq_len"]
batch_size = training_params["batch_size"]

folder_loc = "data/freefall/"
function_name = "variable_height"

train_data = pd.read_csv(folder_loc + function_name + "/train_data.csv")
val_data = pd.read_csv(folder_loc + function_name + "/val_data.csv")
test_data = pd.read_csv(folder_loc + function_name + "/test_data.csv")
oos_test_data = pd.read_csv(folder_loc + function_name + "/oos_test_data.csv")

float_col = train_data.iloc[:, -1].astype(float)

y_min = float_col.min()
y_max = float_col.max()
y_mid = (y_max + y_min) / 2

# Take out the middle 20% of y values
y_range = y_max - y_min
y_min = round(y_mid - y_range * 0.025, 2)
y_max = round(y_mid + y_range * 0.025, 2)

# Get the indices of the middle 20% of y values
test_indices = float_col[(float_col > y_min) & (float_col < y_max)].index
train_indices = float_col[(float_col <= y_min) | (float_col >= y_max)].index

# Get the test data
test_data = train_data.iloc[test_indices]
train_data = train_data.iloc[train_indices]

# Reset the indices
train_data.index = [None] * len(train_data)
val_data.index = [None] * len(val_data)
test_data.index = [None] * len(test_data)

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

# oos_dataloader = make_finetune_dataloader(data=oos_test_data, tokenizer=tokenizer, max_length=max_length,
#                                           batch_size=batch_size, output_type='num')

config = BertConfig.from_pretrained(model_name)
config.num_labels = 1
output_type = "num"

total_layers = config.num_hidden_layers
flayers = 12

try:
    model = BertForSequenceClassification(config)

    batch_logger.log_info(f"Model has Total number of layers: {total_layers}")

    model = freeze_bert_layers(model, flayers)
    batch_logger.log_info(f"Model has {count_frozen_bert_layers(model)} frozen layers")

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
        optimizer, num_warmup_steps=0, num_training_steps=len(train_dataloader) * epochs
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
        f"{function_name} data with {output_type} output, {len(train_data)} training examples and"
        f" {len(test_data)} test examples"
        f"and {flayers} frozen layers"
    )
    batch_logger.log_info(
        f"Test loss: {test_loss:.4f} for values between {y_min:.2f} and {y_max:.2f}"
    )

except Exception as e:
    batch_logger.log_warning(f"Error: {e}")
    raise e

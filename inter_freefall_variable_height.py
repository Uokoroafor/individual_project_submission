import pandas as pd
import torch
from torch import nn
import random
from models.gpt.models.do_transformer import DecodeOnlyTransformer
from utils.basic_tokeniser import BasicTokeniser
from utils.bpe import BPE
from utils.data_utils import read_in_data, make_data_loaders
from utils.file_utils import load_config
from utils.train_utils import PhysicalTrainer, set_seed
from utils.logging_utils import TrainingLogger

# Load the training hyperparameters from the txt file
training_hyperparams = load_config("config_structure.txt")

# Set the random seed for reproducibility
set_seed(6_345_789)
# Wilson Pickett - 634-5789 https://www.youtube.com/watch?v=TSGuaVAufV0

# Create the logger
batch_logger = TrainingLogger("inter_logs_num.txt", verbose=False)

device = training_hyperparams["device"]
block_size = training_hyperparams["max_seq_len"]
batch_size = training_hyperparams["batch_size"]
eval_iters = training_hyperparams["eval_every"]
max_iters = training_hyperparams["epochs"]
lr = training_hyperparams["learning_rate"]

data_folder = "data/freefall/variable_height/"
file_path = "minimal_text.txt"  # Update this to the file containing the data
function_name = "Freefall Environment"  # Update this to the name of the dataset being trained (or the name of the function)
train_data_path = f"train_data.csv"
val_data_path = f"val_data.csv"
test_data_path = f"test_data.csv"

output_type = "num"  # 'num' or 'text'

# pooling = "cls"  # 'max', 'mean', 'cls', 'none' use none for text generation
data_portion = 200_000
pooling = "cls"

use_bpe = False  # Set to True to use BPE, False to use a character encoder/decoder

encoding_str = "bpe" if use_bpe else "char"

logging_intro = (f"Training on {function_name} with {output_type} output and {pooling} pooling on "
                 f"{data_folder + file_path} data. Using {encoding_str} encoding.")

# Read in the data
data = read_in_data(data_folder + file_path, make_dict=False)

# Create the tokeniser
if use_bpe:
    bpe = BPE(data)
    # Train for 50 iterations
    bpe.train(50)
    gpt_tokeniser = bpe
else:
    # Use BasicTokeniser for char-level encoding
    gpt_tokeniser = BasicTokeniser(data)

# Create the encoder and decoder dictionaries and the encode and decode functions
encoder_dict, decoder_dict, encode, decode = (
    gpt_tokeniser.lookup_table,
    gpt_tokeniser.reverse_lookup_table,
    gpt_tokeniser.encode,
    gpt_tokeniser.decode,
)

encoding_utils = dict(
    enc_dict=encoder_dict, dec_dict=decoder_dict, encode_fn=encode, decode_fn=decode
)

# Read in the data as pandas dataframes but dont't include the index
train_data = pd.read_csv(data_folder + train_data_path, dtype=str, index_col=None)
val_data = pd.read_csv(data_folder + val_data_path, dtype=str, index_col=None)
# test_data = pd.read_csv(data_folder + test_data_path, dtype=str)
# train_data.index = [None] * len(train_data)
# val_data.index = [None] * len(val_data)
# print(train_data.head())

# Remove the middle 20% of y values and make that the test data
# Find range of last column of the training data
# The values are strings so convert to float first

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

train_loader, val_loader, test_loader, max_seq_len = make_data_loaders(
    tokeniser=gpt_tokeniser,
    train_data=train_data,
    val_data=val_data,
    test_data=test_data,
    batch_size=batch_size,
    output=output_type,
    shuffle=True,
    max_seq_len=block_size,
)

# update block size to be the max sequence length
block_size = max_seq_len

# Create the model, loss function and optimiser
loss_fn = (
    nn.MSELoss()
    if output_type == "num"
    else nn.CrossEntropyLoss(ignore_index=encoder_dict[gpt_tokeniser.pad])
)

model = DecodeOnlyTransformer(
    src_pad=encoder_dict["<pad>"],
    src_sos=encoder_dict["<sos>"],
    vocab_size_enc=len(encoder_dict),
    output_size=1 if output_type == "num" else len(encoder_dict),
    pooling=pooling if output_type == "num" else "none",
    max_seq_len=block_size,
    num_heads=training_hyperparams["num_heads"],
    num_layers=training_hyperparams["num_layers"],
    d_model=training_hyperparams["d_model"],
    d_ff=training_hyperparams["d_ff"],
    dropout_prob=training_hyperparams["dropout_prob"],
    device=device,
)

optimiser = torch.optim.Adam(model.parameters(), lr=lr)
scheduler = None

device = torch.device(training_hyperparams["device"])

# Move the model and loss function to the device
model = model.to(device)
loss_fn = loss_fn.to(device)

trainer = PhysicalTrainer(
    model=model,
    optimiser=optimiser,
    loss_fn=loss_fn,
    training_hyperparameters=training_hyperparams,
    encoding_utils=encoding_utils,
    scheduler=scheduler,
)

model, _, _ = trainer.train(
    train_dataloader=train_loader,
    val_dataloader=val_loader,
    save_model=True,
    plotting=True,
    verbose=True,
    early_stopping=True,
    early_stopping_patience=20,
    logging_intro=logging_intro,
)

test_loss = trainer.log_numerical_outputs(
    test_loader, decode, "test_log.txt", output_type=output_type
)

batch_logger.log_info(f"Training log is saved at {trainer.path} for")
batch_logger.log_info(f"{function_name} on {data_folder} data with {output_type} "
                      f"output and {len(train_data)} training examples, {len(test_data)} test examples and ")
batch_logger.log_info(f"Test loss: {test_loss:.4f} for values between {y_min:.2f} and {y_max:.2f}")

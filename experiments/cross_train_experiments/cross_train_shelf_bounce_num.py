import pandas as pd
import torch
from torch import nn
import random
from models.gpt.models.eo_transformer import EncodeOnlyTransformer
from utils.basic_tokeniser import BasicTokeniser
from utils.bpe import BPE
from utils.data_utils import read_in_data, make_data_loaders, make_data_loader
from utils.file_utils import load_config
from utils.train_utils import PhysicalTrainer, set_seed
from utils.logging_utils import TrainingLogger

# Load the training hyperparameters from the txt file
training_hyperparams = load_config("../../config_files/cross_config_structure_num.txt")

# Set the random seed for reproducibility
set_seed(6_345_789)
# Wilson Pickett - 634-5789 https://www.youtube.com/watch?v=TSGuaVAufV0

# Create the logger
batch_logger = TrainingLogger("../../cross_training_logs.txt", verbose=False)

device = training_hyperparams["device"]
block_size = training_hyperparams["max_seq_len"]
batch_size = training_hyperparams["batch_size"]
eval_iters = training_hyperparams["eval_every"]
max_iters = training_hyperparams["epochs"]
lr = training_hyperparams["learning_rate"]

folders = ["variable_angle", "variable_ballheight", "variable_ballheight_angle", "variable_shelfheight",
           "variable_shelfheight_angle", "variable_shelfheight_ballheight", "variable_shelfheight_ballheight_angle"]

# Want to train on a concatenated set of two variables and train on the combined data
test_folders = ["variable_ballheight_angle", "variable_shelfheight_angle", "variable_shelfheight_ballheight",
                "variable_shelfheight_ballheight_angle"]
# split each folder name by _, then train on the first two variables

train_folders = []

for x in test_folders:
    train_list = []
    x_ = x.split("_")[1:]
    # add variable_ to the start of each and append to train_folders
    for y in x_:
        train_list.append("variable_" + y)
    train_folders.append(train_list)

print(train_folders)
print(test_folders)

for train_folder, test_folder in zip(train_folders, test_folders):
    try:
        data_folders = [f"data/shelf_bounce/{folder}/" for folder in train_folder]

        test_folder = f"data/shelf_bounce/{test_folder}/"
        # data_folder = f"data/shelf_bounce/{folder}/"
        file_path = "minimal_text.txt"  # Update this to the file containing the data
        function_name = "Shelf Bounce Environment"  # Update this to the name of the dataset being trained (or the name of the function)
        train_data_path = f"train_data.csv"
        val_data_path = f"val_data.csv"
        test_data_path = f"test_data.csv"
        oos_test_data_path = f"oos_test_data.csv"
        output_type = "num"  # 'num' or 'text'

        # pooling = "cls"  # 'max', 'mean', 'cls', 'none' use none for text generation
        # data_portion = 200_000
        pooling = "cls"  # Defaulting to cls now as it seems to be the best

        # # print(f"Training for pooling: {pooling}")
        # data_portion = 1

        use_bpe = False  # Set to True to use BPE, False to use a character encoder/decoder

        encoding_str = "bpe" if use_bpe else "char"

        data_folder_str = ", ".join(train_folder)

        logging_intro = (f"Training on {function_name} with {output_type} output and {pooling} pooling on "
                         f"({data_folder_str}) datasets.")

        # Train the encoder on the test dataset which is a combination of the two variables
        data = read_in_data(test_folder + file_path, make_dict=False)

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

        # Read in the data as pandas dataframes and combine them
        train_data = [pd.read_csv(data_folder + train_data_path, dtype=str) for data_folder in data_folders]
        val_data = [pd.read_csv(data_folder + val_data_path, dtype=str) for data_folder in data_folders]


        # concat the dataframes
        train_data = pd.concat(train_data, ignore_index=True)
        val_data = pd.concat(val_data, ignore_index=True)

        test_data = pd.read_csv(test_folder + test_data_path, dtype=str)
        # We will not do OOS testing for this experiment


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

        model = EncodeOnlyTransformer(
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
        batch_logger.log_info(f"{function_name} on ({data_folder_str}) data with {output_type} "
                              f"output, {pooling} pooling, {encoding_str} encoding and {len(train_data)} training  examples.")
        batch_logger.log_info(f"Test loss: {test_loss:.4f}")

    except Exception as e:
        batch_logger.log_info(f"Error during Training: {e}")
        continue

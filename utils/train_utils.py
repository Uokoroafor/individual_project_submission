import pickle
from typing import Dict, Optional, Tuple, Callable, List, Union
import torch
from torch import nn
from utils.file_utils import create_training_folder, save_losses, save_config
from utils.plot_utils import plot_losses, plot_predictions
from utils.time_utils import EpochTimer
from utils.logging_utils import TrainingLogger
import sys


class Trainer:
    def __init__(
            self,
            model: nn.Module,
            optimiser: torch.optim.Optimizer,
            loss_fn: torch.nn.modules.loss._Loss,
            training_hyperparameters: Dict,
            encoding_utils: Dict,
            scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
    ):
        """Constructor class for Trainer used to train a transformer model for language modelling and text generation
        Args:
            model (nn.Module): Model to train
            optimiser (torch.optim.Optimizer): Optimiser to use for training
            loss_fn (torch.nn.modules.loss._Loss): Loss function to use for training
            training_hyperparameters (Dict): Dictionary containing training hyperparameters
            encoding_utils (Dict): Dictionary containing encoder/decoder dictionaries and functions
            scheduler (Optional[torch.optim.lr_scheduler._LRScheduler], optional): Learning rate scheduler.
            Defaults to None.
        """
        self.train_data = None
        self.val_data = None
        self.model = model
        self.optimiser = optimiser
        self.loss_fn = loss_fn
        self.encoding_utils = encoding_utils
        self.scheduler = scheduler
        self.best_model_dict = None

        # Preallocate variables defined in set_training_hyperparameters
        self.device = None
        self.epochs = None
        self.batch_size = None
        self.eval_every = None
        self.eval_iters = None
        self.max_seq_len = None
        self.save_every = None

        # Create a folder to save the model and training losses
        self.path = create_training_folder()

        # Unpack training hyperparameters
        self.set_training_hyperparameters(**training_hyperparameters)

        # Move the model to the device
        self.model.to(self.device)

        # Save the training hyperparameters as a  txt file
        save_config(training_hyperparameters, f"{self.path}/config.txt")

        # Save the model architecture as a txt file
        with open(f"{self.path}/model.txt", "w") as f:
            f.write(str(self.model))

        # Save the parameters of the model as a txt file
        save_config(self.model.count_parameters(), f"{self.path}/model_parameters.txt")

        # Save the encoding_utils as a pickle file
        filename = f"{self.path}/encoding_utils.pkl"
        with open(filename, "wb") as file:
            pickle.dump(self.encoding_utils, file)

    def train(
            self,
            train_data: torch.Tensor,
            val_data: torch.Tensor,
            save_model: bool = True,
            save_model_path: Optional[str] = None,
            plotting: bool = True,
            verbose: bool = True,
            early_stopping: bool = False,
            logging_intro: Optional[str] = None,
    ):
        """Train the model
        Args:
            train_data (torch.Tensor): Training data
            val_data (torch.Tensor): Validation data
            save_model (bool, optional): Whether to save the model(s) and save the best model. Defaults to True.
            save_model_path (Optional[str], optional): Path to save the model. Defaults to None.
            plotting (bool, optional): Whether to plot the losses. Defaults to True.
            verbose (Optional[bool], optional): Whether to print the progress of training. Defaults to True.
            early_stopping (bool, optional): Whether to use early stopping. Defaults to False.
            logging_intro (Optional[str], optional): Introductory message to print in the training log. Defaults to None.

        """

        self.train_data = train_data
        self.val_data = val_data

        train_losses = []
        val_losses = []
        lowest_val_loss = float("inf")
        logger = TrainingLogger(
            self.path + "/training_logs/training_log.txt",
            name="training_log",
            verbose=verbose,
        )

        if logging_intro is not None:
            logger.log_info(logging_intro)
        logger.log_info(
            f"Training {type(self.model).__name__} for {self.epochs} iterations"
        )

        try:
            timer = EpochTimer()
            timer.start()
            decode = self.encoding_utils["decode_fn"]

            for i in range(self.epochs + 1):
                # Running for one extra epoch to get the final validation loss
                if i % self.eval_every == 0:
                    losses = self.estimate_loss()
                    logger.log_info(
                        f'At Iteration: {max(1, i)}/{self.epochs}, Train loss: {losses["train"]: ,.4f}, '
                        f'Val loss: {losses["val"]: ,.4f}'
                    )

                    timer.lap()
                    logger.log_info(
                        timer.print_last_epoch_time(
                            label=f"Time taken for last {self.eval_every} iterations: "
                        )
                    )
                    if verbose:
                        # Generate a sample from the model
                        chars = decode(
                            self.model.generate(
                                start_token=self.model.trg_sos * torch.ones((1, 1), dtype=torch.long),
                                max_length=30,
                                sampled=False,
                            )[0].tolist()
                        )
                        logger.log_info(
                            f"Generating 30 characters without sampling: {''.join(chars)}"
                        )

                    train_losses.append(losses["train"])
                    val_losses.append(losses["val"])

                    # Update the best model state dict and lowest validation loss
                    lowest_val_loss = self.update_best_model_dict(
                        losses["val"], lowest_val_loss
                    )

                    if (
                            early_stopping
                            and i > 0
                            and val_losses[-1] > val_losses[-2] > val_losses[-3]
                    ):
                        logger.log_info(f"Stopping early after {i} iterations")
                        break

                if self.save_every is not None and i % self.save_every == 0:
                    self.save_model(
                        f"{self.path}/saved_models/{type(self.model).__name__}_iter_{max(1, i)}.pt"
                    )

                if i == self.epochs:
                    break

                # Get a batch of data
                xb, yb = self.get_batch("train")

                # Zero the gradients
                self.optimiser.zero_grad()

                # Get the embeddings and the loss (Forward pass)
                embeds = self.model(trg=xb)

                loss = self.loss_fn(embeds.view(-1, embeds.size(-1)), yb.view(-1))

                # Back propagate the loss (Backward pass)
                loss.backward()

                # Take a step with the optimiser
                self.optimiser.step()

                if self.scheduler is not None:
                    self.scheduler.step()

            timer.lap()
            logger.log_info(timer.print_total_time(label="Total time taken: "))

            if save_model:
                # Load and save the best model
                self.model.load_state_dict(self.best_model_dict)
                save_model_path = self.save_best_model(save_model_path)
                logger.log_info(f"Saved best model at: {save_model_path}")

                # Save the losses
                save_losses(train_losses, val_losses, self.path)
                logger.log_info(
                    f"Saved losses at: {self.path}/training_logs/losses.csv"
                )

            else:
                # If we are not saving the model, load the best model
                self.model.load_state_dict(self.best_model_dict)

            if plotting:
                plot_save_path = (
                    f"{self.path}/training_logs/{type(self.model).__name__}_losses.png"
                    if save_model
                    else None
                )

                plot_losses(
                    train_losses,
                    val_losses,
                    model_name=type(self.model).__name__,
                    num_epochs=self.epochs,
                    saved_path=plot_save_path,
                )
        except Exception as e:
            logger.log_error(f"Error while training: {str(e)}")
            raise e

        except KeyboardInterrupt:
            logger.log_info("Training interrupted by the user")

        return self.model, train_losses, val_losses

    def evaluate(
            self,
            test_data: torch.Tensor,
            verbose: bool = True,
            num_iters: Optional[int] = None,
    ) -> float:
        """Evaluate the model
        Args:
            test_data (torch.Tensor): Test data
            verbose (bool, optional): Whether to print the progress of evaluation. Defaults to True.
            num_iters (Optional[int], optional): Number of iterations to evaluate. Defaults to None
            (Evaluate on the entire test data).
        Returns:
            float: Test loss
        """

        self.model.eval()
        if num_iters is None:
            test_loss = self.calculate_test_loss(test_data)
            if verbose:
                print(f"Test loss: {test_loss: ,.4f}")
        else:
            test_loss = self.estimate_test_loss(test_data, num_iters=num_iters)
            if verbose:
                print(f"Test loss: {test_loss: ,.4f}")
        return test_loss

    def set_training_hyperparameters(self, **kwargs):
        """Set the training hyperparameters which are passed as keyword arguments
        Args:
            **kwargs: Training hyperparameters
        """
        for key, value in kwargs.items():
            setattr(self, key, value)

    def save_model(self, model_path: str):
        """Save the model
        Args:
            model_path (str): Path to save the model
        """
        torch.save(self.model, model_path)

    def save_best_model(self, best_model_path: Optional[str]):
        """Save the best model
        Args:
            best_model_path (Optional[str]): Path to save the best model
        """
        if best_model_path is None:
            best_model_path = (
                f"{self.path}/saved_models/{type(self.model).__name__}_best.pt"
            )
        self.save_model(best_model_path)
        return best_model_path

    def update_best_model_dict(self, loss_val: float, lowest_val_loss: float) -> float:
        """Update the best model dictionary if the validation loss is the lowest so far
        Args:
            loss_val (float): Dictionary containing the training and validation losses
            lowest_val_loss (float): Lowest validation loss so far
        """
        if loss_val < lowest_val_loss:
            # Update the lowest validation loss
            lowest_val_loss = loss_val
            # Save the model state dict
            self.best_model_dict = self.model.state_dict()
        return lowest_val_loss

    def get_batch(
            self, split: Optional[str] = None, data: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get a batch of data from the train, validation or a provided data tensor
        Args:
            split (Optional[str], optional): Split to get the data from. Defaults to None.
            data (Optional[torch.Tensor], optional): Data tensor to get the batch from. Defaults to None.
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Batch of data
        """

        if data is not None:
            data = data
        else:
            if split == "train":
                data = self.train_data
            elif split == "val":
                data = self.val_data
            else:
                raise ValueError(f"Unknown split: '{split}'")
        ix = torch.randint(len(data) - self.max_seq_len, (self.batch_size,))
        x = torch.stack([data[i: i + self.max_seq_len] for i in ix])
        y = torch.stack([data[i + 1: i + self.max_seq_len + 1] for i in ix])
        x, y = x.to(self.device), y.to(
            self.device
        )  # Transfer the data to the GPU if we are using it
        return x, y

    @torch.no_grad()
    def estimate_loss(self) -> Dict[str, float]:
        """Estimate the loss for the data

        Returns:
            Dict[str, float]: Dictionary containing the training and validation losses
        """
        self.model.eval()  # Set the model to evaluation mode
        out = {}
        for split in ["train", "val"]:
            losses = []
            for i in range(self.eval_iters):
                x, y = self.get_batch(split)
                embeds = self.model(trg=x)
                loss = self.loss_fn(embeds.flatten(end_dim=1), y.flatten())
                losses.append(loss.item())
            out[split] = torch.tensor(losses).mean().item()
        return out

    def calculate_test_loss(self, test_data: torch.Tensor) -> float:
        """Calculate the loss on the full test data (without sampling)
        Args:
            test_data (torch.Tensor): Test data
        Returns:
            float: Loss on the test data
        """
        self.model.eval()
        test_loss = self.loss_fn(
            self.model(test_data).view(-1, test_data.size(-1)), test_data.view(-1)
        )
        return test_loss.item()

    def estimate_test_loss(
            self, test_data: torch.Tensor, num_iters: int = 100
    ) -> float:
        """Estimate the loss on the test data by sampling a number of batches
        Args:
            test_data (torch.Tensor): Test data
            num_iters (int, optional): Number of samples to estimate the loss. Defaults to 100.
        Returns:
            float: Loss on the test data
        """
        self.model.eval()
        losses = []
        for _ in range(num_iters):
            x, y = self.get_batch(data=test_data)
            embeds = self.model(trg=x)
            loss = self.loss_fn(embeds.flatten(end_dim=1), y.flatten())
            losses.append(loss.item())
        return torch.tensor(losses).mean().item()


class PhysicalTrainer(Trainer):
    def __init__(
            self,
            model: nn.Module,
            optimiser: torch.optim.Optimizer,
            loss_fn: torch.nn.modules.loss._Loss,
            training_hyperparameters: Dict,
            encoding_utils: Dict,
            scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
    ):
        """Initialize the trainer for the physical model. This is for training on the generated physical scenarios.
        No generation is done during training so this method is obviated. The training data is loaded from the
        generated data directory.

        Args:
            model (nn.Module): Model to train
            optimiser (torch.optim.Optimizer): Optimiser
            loss_fn (torch.nn.modules.loss._Loss): Loss function
            training_hyperparameters (Dict): Training hyperparameters which include the batch size, number of epochs
            defined in the config file
            encoding_utils (Dict): Encoding utilities
            scheduler (Optional[torch.optim.lr_scheduler._LRScheduler], optional): Scheduler. Defaults to None.
        """
        super().__init__(
            model,
            optimiser,
            loss_fn,
            training_hyperparameters,
            encoding_utils,
            scheduler,
        )
        self.logger = None

    def train(
            self,
            train_dataloader: torch.utils.data.DataLoader,
            val_dataloader: torch.utils.data.DataLoader,
            save_model: bool = True,
            save_model_path: Optional[str] = None,
            plotting: bool = True,
            verbose: bool = True,
            early_stopping: bool = False,
            early_stopping_patience: int = 10,
            logging_intro: Optional[str] = None,

    ):
        """Train the model

        Args:
            train_dataloader (torch.utils.data.DataLoader): Training dataloader
            val_dataloader (torch.utils.data.DataLoader): Validation dataloader
            save_model (bool, optional): Whether to save the model. Defaults to True.
            save_model_path (Optional[str], optional): Path to save the model. Defaults to None.
            plotting (bool, optional): Whether to plot the training and validation losses. Defaults to True.
            verbose (Optional[bool], optional): Whether to print the training and validation losses. Defaults to True.
            early_stopping (bool, optional): Whether to use early stopping. Defaults to False.
            early_stopping_patience (int, optional): Patience for early stopping. Defaults to 10.
            logging_intro (Optional[str], optional): Introductory message to print in the training log. Defaults to None.
        """
        train_losses = []
        val_losses = []
        lowest_val_loss = float("inf")
        stop_training = False
        logger = TrainingLogger(
            self.path + "/training_logs/training_log.txt",
            name="training_log",
            verbose=verbose,
        )
        self.logger = logger

        if logging_intro is not None:
            logger.log_info(logging_intro)

        logger.log_info(
            f"Training {type(self.model).__name__} for {self.epochs} iterations"
        )

        try:
            timer = EpochTimer()
            timer.start()

            count = 0

            for i in range(self.epochs):
                train_loss = self.training_loop(train_dataloader, method="train")
                train_losses.append(train_loss)

                val_loss = self.training_loop(val_dataloader, method="val")
                val_losses.append(val_loss)

                if i % self.eval_every == 0:
                    logger.log_info(
                        f"At Iteration: {i + 1}/{self.epochs}, Train loss: {train_loss: ,.4f}, "
                        f"Val loss: {val_loss: ,.4f}"
                    )

                    timer.lap()
                    logger.log_info(
                        timer.print_last_epoch_time(
                            label=f"Time taken for last {self.eval_every} iteration(s): "
                        )
                    )

                # Update the best model state dict and lowest validation loss
                lowest_val_loss, count = self.update_best_model_dict_(
                    val_loss, lowest_val_loss, count
                )

                if early_stopping and i > 0 and count >= early_stopping_patience:
                    logger.log_info(f"Stopping early after {i + 1} iterations")
                    stop_training = True

                if self.save_every is not None and i % self.save_every == 0:
                    self.save_model(
                        f"{self.path}/saved_models/{type(self.model).__name__}_iter_{i + 1}.pt"
                    )

                if save_model and count == 0:
                    save_model_path = self.save_best_model(save_model_path)

                if stop_training:
                    break

            timer.lap()
            logger.log_info(timer.print_total_time(label="Total time taken: "))

            if save_model:
                # Save the losses
                save_losses(train_losses, val_losses, self.path)
                logger.log_info(
                    f"Saved losses at: {self.path}/training_logs/losses.csv"
                )

            self.model.load_state_dict(self.best_model_dict)
            logger.log_info(f"Saved best model at: {save_model_path}")

            if plotting:
                plot_save_path = (
                    f"{self.path}/training_logs/{type(self.model).__name__}_losses.png"
                    if save_model
                    else None
                )

                plot_losses(
                    train_losses,
                    val_losses,
                    model_name=type(self.model).__name__,
                    saved_path=plot_save_path,
                )
        except Exception as e:
            logger.log_error(f"Error while training: {str(e)}")
            raise e

        except KeyboardInterrupt:
            logger.log_info("Training interrupted by the user")
            # Exit the program
            sys.exit()

        return self.model, train_losses, val_losses

    def training_loop(
            self, dataloader: torch.utils.data.DataLoader, method: str = "train"
    ) -> float:
        """Training loop for the model

        Args:
            dataloader (torch.utils.data.DataLoader): DataLoader for the training data
            method (str, optional): Whether to train or validate. Defaults to 'train'.

        Returns:
            float: Average loss over the training loop
        """
        if method == "train":
            self.model.train()
        elif method == "val":
            self.model.eval()
        else:
            raise ValueError(f"method must be either 'train' or 'val' not {method}")

        total_loss = 0
        # Only want to loop through a subset of the data_loader as it is too large

        for batch_idx, (inputs, targets) in enumerate(dataloader):
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)

            if method == "train":
                self.optimiser.zero_grad()

            outputs = self.model(inputs)

            # want to reshape the outputs and targets to be 2D with the same number of columns
            if self.model.output_size == 1:
                outputs = self.model(inputs).squeeze(-1)
                targets = targets.squeeze(-1)
                loss = self.loss_fn(outputs, targets)
            else:
                loss = self.loss_fn(
                    outputs.view(-1, outputs.size(-1)), targets.view(-1)
                )

            # only want to backpropagate if we are training
            if method == "train":
                loss.backward()
                self.optimiser.step()

                # step the scheduler if it is not None
                if self.scheduler is not None:
                    self.scheduler.step()

            total_loss += loss.item()

        return total_loss / len(dataloader)

    def update_best_model_dict_(
            self, loss_val: float, lowest_val_loss: float, count: int
    ) -> Tuple[float, int]:
        """Update the best model dictionary if the validation loss is the lowest so far
        Args:
            loss_val (float): Dictionary containing the training and validation losses
            lowest_val_loss (float): Lowest validation loss so far
            count (int): Number of times the validation loss has not been lower than the lowest validation loss.
            If this exceeds the early stopping patience, training will stop
        Returns:
            float: The updated lowest validation loss
        """
        if loss_val < lowest_val_loss:
            # Update the lowest validation loss
            lowest_val_loss = loss_val
            # Save the model state dict
            self.best_model_dict = self.model.state_dict()
            count = 0

        else:
            count += 1

        return lowest_val_loss, count

    def calculate_test_loss(self, test_data: torch.Tensor) -> float:
        """Calculate the loss on the full test data (without sampling)
        Args:
            test_data (torch.Tensor): Test data
        Returns:
            float: Loss on the test data
        """
        self.model.eval()
        # Reshape according to the model output size
        if self.model.output_size == 1:
            test_loss = self.loss_fn(self.model(test_data).squeeze(-1), test_data)
        else:
            test_loss = self.loss_fn(
                self.model(test_data).view(-1, test_data.size(-1)), test_data.view(-1)
            )
        return test_loss.item()

    def log_numerical_outputs(
            self,
            dataloader: torch.utils.data.DataLoader,
            decode: Callable,
            log_name: Optional[str] = None,
            output_type: str = "num",
    ):
        """Log the numerical outputs of the model to a file. It also plots the predictions vs the targets
        Args:
            dataloader (DataLoader): DataLoader for the data
            decode (Callable): Function to decode the numerical outputs
            log_name (Optional[str], optional): Name of the log file. Defaults to None.
            output_type (str): Type of output. Either 'num' or 'text'

        """
        self.model.eval()
        if log_name is None:
            log_name = "float_predictions.txt"
        file_name = f"{self.path}/training_logs/{log_name}"
        with torch.no_grad():
            test_loss = 0
            predictions = []
            targets = []
            for batch_idx, (inputs, target) in enumerate(dataloader):
                inputs = inputs.to(self.device)

                if output_type == "num":
                    output = self.model(inputs).squeeze(-1)
                    target = target.to(self.device).squeeze(-1)
                    loss = self.loss_fn(output, target)
                else:
                    output = self.model(inputs)
                    target = target.to(self.device)
                    # print(output.shape, target.shape)
                    loss = self.loss_fn(
                        output.view(-1, output.size(-1)), target.view(-1)
                    )

                test_loss += loss.item()

                if output_type == "text":
                    # We want to append argmax of the output
                    predictions.extend(output.argmax(-1).tolist())
                    targets.extend(target.tolist())
                else:
                    predictions.extend(output.view(-1).tolist())
                    targets.extend(target.view(-1).tolist())
                if batch_idx % 5:
                    for i in range(len(inputs)):
                        with open(file_name, "a+") as f:
                            f.write(
                                "Question is "
                                + "".join(decode(inputs[i].tolist(), True))
                                + "\n"
                            )
                            if output_type == "text":
                                f.write(
                                    "Target is "
                                    + "".join(decode(target[i, :].tolist(), True))
                                    + "\n"
                                )
                                pred = "".join(decode(output[i, :].argmax(-1).tolist()))
                                f.write(
                                    f"Prediction is {pred.split('<eos>')[0].replace('<sos>', '')}"
                                    + "\n\n"
                                )
                            else:
                                f.write(
                                    "Target is "
                                    + str(round(target[i].item(), 4))
                                    + "\n"
                                )
                                pred = output[i].tolist()
                                f.write(f"Prediction is {pred:,.4f}" + "\n\n")

            test_loss /= len(dataloader)
            self.logger.log_info(f"Test loss was {test_loss :,.4f}")

            plot_save_path = (
                f"{self.path}/training_logs/{type(self.model).__name__}_predictions.png"
            )

            if output_type == "text":
                predictions, targets, count, error_log = self.convert_string_to_float(
                    predictions, targets, decode
                )
                if count > 0:
                    self.logger.log_warning(
                        f"Could not convert {count} predictions to floats"
                    )
                    self.logger.log_warning(error_log)
                # log MSE error
                self.logger.log_info(f"MSE Error on converted numerical outputs "
                                     f"is {nn.MSELoss()(torch.tensor(predictions), torch.tensor(targets)) :,.4f}")

            plot_predictions(
                predictions=predictions,
                targets=targets,
                model_name=type(self.model).__name__,
                saved_path=plot_save_path,
            )

    @staticmethod
    def convert_string_to_float(
            predictions: List[str], targets: List[str], decode: Callable
    ) -> Tuple[List[float], List[float], int, str]:
        """Convert the predictions and targets from strings to floats
        Args:
            predictions (List[str]): List of predicted tokens
            targets (List[str]): List of targets
            decode (Callable): Function to decode the numerical outputs
        Returns:
            Tuple[List[float],List[float], int, str]: Tuple containing the converted predictions and targets, the number of errors and the error log
        """
        # convert each element in both lists to numbers. Decode and convert to float. If there is an error
        # Then remove that index from both lists and raise a warning
        pred_out = []
        target_out = []
        count = 0
        error_log = ''

        for i in range(len(predictions)):
            try:
                pred = "".join(decode(predictions[i]))
                pred = pred.split('<eos>')[0].replace('<sos>', '')
                pred_out.append(float(pred))
                target_out.append(float("".join(decode(targets[i], True))))

            except ValueError:
                count += 1
                if count <= 20:
                    # Only want to log the first 20 errors otherwise the log file gets too big
                    error_log += f"Could not convert Prediction: {pred} to float.\n"
                    error_log += f"Target was {float(''.join(decode(targets[i], True)))}\n\n"
                continue
        return pred_out, target_out, count, error_log


class NNTrainer:
    def __init__(
            self,
            model: nn.Module,
            optimiser: torch.optim.Optimizer,
            loss_fn: torch.nn.modules.loss._Loss,
            scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None):
        """Constructor class for Trainer used to train a standard neural network model
        Args:
            model (nn.Module): Model to train
            optimiser (torch.optim.Optimizer): Optimiser to use for training
            loss_fn (torch.nn.modules.loss._Loss): Loss function to use for training
          """
        self.train_data = None
        self.val_data = None
        self.model = model
        self.optimiser = optimiser
        self.loss_fn = loss_fn
        self.scheduler = scheduler

        self.best_model_dict = None

        # Preallocate variables defined in set_training_hyperparameters
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Create a folder to save the model and training losses
        self.path = create_training_folder()

        # Move the model to the device
        self.model.to(self.device)

        # Save the model architecture as a txt file
        with open(f"{self.path}/model.txt", "w") as f:
            f.write(str(self.model))

    def train(
            self,
            train_dataloader: torch.utils.data.DataLoader,
            val_dataloader: torch.utils.data.DataLoader,
            epochs: int,
            eval_every: int = 1,
            save_model: bool = True,
            save_model_path: Optional[str] = None,
            plotting: bool = True,
            verbose: bool = True,
            early_stopping: bool = False,
            early_stopping_patience: int = 10,
            logging_intro: Optional[str] = None,
    ):
        """Train the model
        Args:
            train_dataloader (torch.utils.data.DataLoader): Training dataloader
            val_dataloader (torch.utils.data.DataLoader): Validation dataloader
            epochs (int): Number of epochs to train for
            eval_every (int, optional): Evaluate the model every eval_every epochs. Defaults to 1.
            save_model (bool, optional): Whether to save the model(s) and save the best model. Defaults to True.
            save_model_path (Optional[str], optional): Path to save the model. Defaults to None.
            plotting (bool, optional): Whether to plot the losses. Defaults to True.
            verbose (Optional[bool], optional): Whether to print the progress of training. Defaults to True.
            early_stopping (bool, optional): Whether to use early stopping. Defaults to False.
            early_stopping_patience (int, optional): Number of iterations to wait before stopping early. Defaults to 10.
            logging_intro (Optional[str], optional): Introductory message to print in the training log. Defaults to None.

        """

        train_losses = []
        val_losses = []
        lowest_val_loss = float("inf")
        logger = TrainingLogger(
            self.path + "/training_logs/training_log.txt",
            name="training_log",
            verbose=verbose,
        )

        if logging_intro is not None:
            logger.log_info(logging_intro)

        logger.log_info(
            f"Training {type(self.model).__name__} for {epochs} iterations"
        )
        count = 0

        try:
            for i in range(epochs):
                # Running for one extra epoch to get the final validation loss
                if i % eval_every == 0:
                    train_loss, val_loss = self.train_loop(
                        train_dataloader, val_dataloader)

                    logger.log_info(
                        f'At Iteration: {max(1, i)}/{epochs}, Train loss: {train_loss:.4f}, Val loss: {val_loss:.4f}'
                    )

                    train_losses.append(train_loss)
                    val_losses.append(val_loss)

                    # Update the best model state dict and lowest validation loss
                    lowest_val_loss, count = self.update_best_model_dict(
                        val_loss, lowest_val_loss, count
                    )

                    if early_stopping and count >= early_stopping_patience:
                        logger.log_info(f"Stopping early after {i} iterations")
                        break

                else:
                    train_loss, _ = self.train_loop(train_dataloader)
                    train_losses.append(train_loss)

            if save_model:
                # Load and save the best model
                self.model.load_state_dict(self.best_model_dict)
                save_model_path = self.save_best_model(save_model_path)
                logger.log_info(f"Saved best model at: {save_model_path}")

                # Save the losses
                save_losses(train_losses, val_losses, self.path)
                logger.log_info(
                    f"Saved losses at: {self.path}/training_logs/losses.csv"
                )

            else:
                # If we are not saving the model, load the best model
                self.model.load_state_dict(self.best_model_dict)

            if plotting:
                plot_save_path = (
                    f"{self.path}/training_logs/{type(self.model).__name__}_losses.png"
                    if save_model
                    else None
                )

                plot_losses(
                    train_losses,
                    val_losses,
                    model_name=type(self.model).__name__,
                    saved_path=plot_save_path,
                )
        except Exception as e:
            logger.log_error(f"Error while training: {str(e)}")
            raise e

        except KeyboardInterrupt:
            logger.log_info("Training interrupted by the user")
            # Exit the program
            sys.exit()

        return self.model, train_losses, val_losses

    def save_model(self, model_path: str):
        """Save the model
        Args:
            model_path (str): Path to save the model
        """
        torch.save(self.model, model_path)

    def save_best_model(self, best_model_path: Optional[str]):
        """Save the best model
        Args:
            best_model_path (Optional[str]): Path to save the best model
        """
        if best_model_path is None:
            best_model_path = (
                f"{self.path}/saved_models/{type(self.model).__name__}_best.pt"
            )
        self.save_model(best_model_path)
        return best_model_path

    def update_best_model_dict(self, loss_val: float, lowest_val_loss: float, count: int) -> Tuple[float, int]:
        """Update the best model dictionary if the validation loss is the lowest so far
        Args:
            loss_val (float): Dictionary containing the training and validation losses
            lowest_val_loss (float): Lowest validation loss so far
            count (int): Number of iterations since the lowest validation loss was updated
        """
        if loss_val < lowest_val_loss:
            # Update the lowest validation loss
            lowest_val_loss = loss_val
            # Save the model state dict
            self.best_model_dict = self.model.state_dict()
            # Reset the count
            count = 0
        else:
            # Increment the count
            count += 1
        return lowest_val_loss, count

    def train_loop(self, dataloader_train: torch.utils.data.DataLoader,
                   dataloader_val: Optional[torch.utils.data.DataLoader] = None) -> Tuple[float, Optional[float]]:
        """Train the model for one epoch and return the train and validation loss.

        Args:
            dataloader_train: The training data
            dataloader_val: The validation data it is optional because we might not want to validate after every epoch

        Returns:
            Tuple[float, Optional[float]]: The train and validation loss if validate is True, otherwise just the train loss

        """
        train_loss = 0

        for batch, (X_train, y_train) in enumerate(dataloader_train):
            # Move X_train and y_train to the device
            X_train, y_train = X_train.to(self.device), y_train.to(self.device)

            train_pred = self.model(X_train).squeeze(-1)
            loss = self.loss_fn(train_pred, y_train)

            # Backpropagation
            self.optimiser.zero_grad()
            loss.backward()
            self.optimiser.step()

            if self.scheduler is not None:
                self.scheduler.step()

            train_loss += loss.item()
        train_loss /= len(dataloader_train.dataset)
        if dataloader_val is not None:
            with torch.no_grad():
                val_loss = 0
                for batch, (X_val, y_val) in enumerate(dataloader_val):
                    X_val, y_val = X_val.to(self.device), y_val.to(self.device)
                    val_pred = self.model(X_val).squeeze(-1)
                    loss = self.loss_fn(val_pred, y_val)
                    val_loss += loss.item()
                val_loss /= len(dataloader_val.dataset)
        else:
            val_loss = None

        return train_loss, val_loss

    def evaluate(self, testdata: Union[torch.utils.data.DataLoader, torch.Tensor]) -> float:
        """Test the model and return the test loss and accuracy.

        Args:
            testdata: The data to test on

        Returns:
            The average test loss and accuracy
        """
        self.model.eval()
        test_loss = 0
        with torch.no_grad():
            if isinstance(testdata, torch.Tensor):
                test_pred = self.model(testdata).squeeze(-1)
                loss = self.loss_fn(test_pred, testdata)
                test_loss += loss.item()
            else:
                for batch, (X_test, y_test) in enumerate(testdata):
                    test_pred = self.model(X_test).squeeze(-1)
                    loss = self.loss_fn(test_pred, y_test)
                    test_loss += loss.item()
        test_loss /= len(testdata.dataset)
        return test_loss


def set_seed(seed: int = 0):
    """Set the random seed for reproducibility
    Args:
        seed (Optional[int], optional): Random seed. Defaults to 0.
    """
    if "torch" in sys.modules:
        sys.modules["torch"].manual_seed(seed)
        sys.modules["torch"].cuda.manual_seed_all(seed)
        sys.modules["torch"].backends.cudnn.deterministic = True
        sys.modules["torch"].backends.cudnn.benchmark = False

    if "numpy" in sys.modules:
        sys.modules["numpy"].random.seed(seed)

    if "random" in sys.modules:
        sys.modules["random"].seed(seed)

from typing import List, Optional
import matplotlib.pyplot as plt
import torch


def plot_losses(
        train_losses: List[float],
        val_losses: List[float],
        model_name: Optional[str] = None,
        num_epochs: Optional[int] = None,
        saved_path: Optional[str] = None,
) -> None:
    """Plot the training and validation losses
    Args:

        train_losses (List[float]): Training losses
        val_losses (List[float]): Validation losses
        model_name (Optional[str], optional): Name of the model. Defaults to None.
        num_epochs (Optional[int], optional): Number of epochs. Defaults to None.
        saved_path: (Optional[str], optional): Path to save the plot. Defaults to None.
    """
    if num_epochs is not None:
        steps = num_epochs
        x = torch.arange(0, num_epochs + 1, num_epochs // (len(train_losses) - 1))
        # Make the x-axis start at 1
        x[0] = 1
    else:
        steps = len(train_losses)
        x = torch.arange(1, len(train_losses) + 1)
    plt.plot(x, train_losses, label="train")
    plt.plot(x, val_losses, label="val", linestyle="dashed")
    if model_name is not None:
        plt.title(f"Losses for the {model_name} model over {steps} iterations")
    else:
        plt.title(f"Losses over {len(train_losses)} steps")
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.legend()
    if saved_path is not None:
        plt.savefig(saved_path)
    plt.show()
    plt.close()  # Close the figure to save memory and prevent plots from overlapping


def plot_predictions(
        predictions: List[float],
        targets: List[float],
        model_name: Optional[str] = None,
        saved_path: Optional[str] = None,
) -> None:
    """Plot the predictions and targets
    Args:
        predictions (List[float]): Predictions
        targets (List[float]): Targets
        model_name (Optional[str], optional): Name of the model. Defaults to None.
        saved_path: (Optional[str], optional): Path to save the plot. Defaults to None.
    """

    # Scatter of predictions vs targets with a diagonal line
    plt.scatter(predictions, targets, s=1)
    plt.plot(targets, targets, color="red", label="Perfect Predictions", linestyle="dashed")
    plt.title(f"Predictions vs Targets for the {model_name} model")
    plt.xlabel("Predictions")
    plt.ylabel("Targets")
    plt.legend()
    if saved_path is not None:
        plt.savefig(saved_path)
    plt.show()
    plt.close()  # Close the figure to save memory and prevent plots from overlapping

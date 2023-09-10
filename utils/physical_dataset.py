from typing import Optional, List, Tuple

import pandas as pd
import torch


class PhysicalDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        output_col: Optional[int] = None,
        col_names: Optional[List[str]] = None,
    ):
        """This class is used to create a dataset from a pandas dataframe. The dataframe should have the following
        format:
        x1, x2, x3, ..., xn, y
        If the col_names argument is not specified, then the column names are assumed to be the same as the column.
        If output_col is not specified, then the last column is assumed to be the output column.

        Args:
            df (pd.DataFrame): The dataframe to be used to create the dataset
            output_col (Optional[int], optional): The index of the output column. Defaults to None.
            col_names (Optional[List[str]], optional): The names of the columns in the dataframe. Defaults to None.
        """
        self.df = df
        if col_names is None:
            self.col_names = df.columns
        else:
            self.col_names = col_names

        if output_col is None:
            self.output_col = len(self.col_names) - 1
        else:
            self.output_col = output_col

        self.x = df[self.col_names[: self.output_col]].values
        self.y = df[self.col_names[self.output_col]].values

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        # Fetching directly from numpy arrays
        x = self.x[idx]
        y = self.y[idx]

        # Converting to tensor
        return torch.tensor(x, dtype=torch.float32), torch.tensor(
            y, dtype=torch.float32
        )

    def get_batch(self, indices: List[int]) -> Tuple[torch.Tensor, torch.Tensor]:
        """This method is used to fetch multiple indices from the dataset. This is useful for creating batches.

        Args:
            indices (List[int]): The indices to be fetched from the dataset

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: A tuple containing the input and output tensors
        """
        x = self.x[indices]
        y = self.y[indices]
        return torch.tensor(x, dtype=torch.float32), torch.tensor(
            y, dtype=torch.float32
        )

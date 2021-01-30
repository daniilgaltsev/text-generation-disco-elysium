import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader
from text_generation.data.lines_dataset import LinesDataset
from typing import Optional, List, Callable, Dict, Any
import os


class LinesDataModule(pl.LightningDataModule):
    """
    A data module class handling lines dataset for splitting train/validation and preprocessing.

    Args:
        batch_size: A size of a batch to use in DataLoaders.
        tokenize: A function that encodes lines during preprocessing.
        tokenize_args: A dictionary of args for tokenizer function in addition to batch.
        bos_token (optional): A string to use as bos token.
        eos_token (optional): A string to use as eos token.
        val_size (optional): A float between 0.0 and 1.0 that represents the proportion of the data to use for validation.
        path (optional): A path to the file containing the lines.
        num_workers (optional): Number of parallel workers to use in DataLoaders.
        seed (optional): A seed used for random splitting.
    """

    PATH: str = os.path.join(os.path.dirname(__file__), "texts_extracted.txt")

    def __init__(
        self,
        batch_size: int,
        tokenize: Callable[..., Dict[str, torch.Tensor]],
        tokenize_args: Dict[str, Any],
        bos_token: str = "<|BOS|>",
        eos_token: str = "<|EOS|>",
        val_size: float = 0.1,
        path: Optional[str] = None,
        num_workers: int = 8,
        seed: int = 0,
    ):
        super().__init__()

        self.batch_size = batch_size
        self.tokenize = tokenize
        self.tokenize_args = tokenize_args
        self.bos_token = bos_token
        self.eos_token = eos_token
        if not (0.0 <= val_size <= 1.0):
            raise ValueError("val_size should be between 0.0 and 1.0 not {:.6f}".format(val_size))
        self.val_size = val_size
        self.path = path
        self.num_workers = num_workers
        self.seed = seed

    def setup(self, stage: Optional[str]):
        """
        Loads the lines dataset and splits it into train and validation
        """

        self.full_dataset = LinesDataset(path=self.path, bos_token=self.bos_token, eos_token=self.eos_token)
        val_size = int(len(self.full_dataset) * self.val_size)
        train_size = len(self.full_dataset) - val_size

        datasets = torch.utils.data.random_split(self.full_dataset, [train_size, val_size], torch.Generator().manual_seed(self.seed))
        self.train_dataset = datasets[0]
        self.val_dataset = datasets[1]

    def _preprocess_batch(self, batch: List[str]) -> Dict[str, torch.Tensor]:
        """
        Preprocesses a batch of lines.
        """

        encoded = self.tokenize(batch, **self.tokenize_args)
        encoded["labels"] = encoded["input_ids"]
        return encoded

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset, 
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            collate_fn=self._preprocess_batch,
            drop_last=True,
            pin_memory=True
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset, 
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            collate_fn=self._preprocess_batch,
            drop_last=False,
            pin_memory=True
        )

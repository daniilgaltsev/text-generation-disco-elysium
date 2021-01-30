from typing import List, Dict, Any
import torch
import pytorch_lightning as pl
from text_generation.models.gpt2_lm import GPT2LMHeadModel


class GPT2LMHeadModelFinetune(pl.LightningModule):
    """
    The class contains a gpt2 model with a language modelling head and the fine-tuning setup for it.

    Args:
        model: An initialized gpt2 model with a language modelling head.
        warmup_steps (optional): Number of warmup_steps during training.
        lr (optional): Learning rate used for optimzer.
    """

    def __init__(
        self,
        model: GPT2LMHeadModel,
        warmup_steps: int = 500,
        lr: float = 0.001
    ):
        super().__init__()

        self.warmup_steps = warmup_steps
        self.lr = lr
        self.model = model

    def generate(self, prompts: List[str], max_length: int = 40, clean_special_tokens: bool = True) -> List[str]:
        """
        Generates seqeunces given a list of prompts

        Args:
            prompts: A list of string, which will be used as prompts for generation
            max_length (optional): Maximum number of tokens to output (includes length of the prompt)
            clean_special_tokens (optional): If True will remove special tokens from generated strings.

        Returns:
            A list of generated strings corresponding to given prompts.
        """

        return self.model.generate(prompts, max_length, clean_special_tokens)
    
    def forward(self, x: Dict[str, torch.Tensor]) -> Any:
        return self.model(x)

    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        batch["labels"] = torch.where(batch["labels"] != self.model.tokenizer.pad_token_id, batch["labels"], -100)
        output = self(batch)
        loss = output["loss"]

        return loss

    def on_validation_epoch_end(self) -> None:
        print("___SAMPLE ON EPOCH {}___".format(self.current_epoch))
        print(*self.generate([""], max_length=80, clean_special_tokens=False))
        print("________________________")

    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        output = self(batch)
        loss = output["loss"]

        self.log("val_loss", loss, prog_bar=True, on_step=False, on_epoch=True)

    def _calculate_number_of_steps(self) -> int:
        """
        Calculates number of steps over the whole training process.
        """

        if self.trainer.max_steps is None and self.trainer.max_epochs is None:
            return len(self.train_dataloader())
        
        train_steps = self.trainer.max_steps
        if self.trainer.max_epochs is not None:
            epochs_steps = self.trainer.max_epochs * len(self.train_dataloader())
        else:
            epochs_steps = train_steps
        
        if train_steps is None:
            return epochs_steps
        
        return min(train_steps, epochs_steps)


    def setup(self, stage: str):
        if stage == "fit":
            self.number_of_steps = self._calculate_number_of_steps()
        
            print("GPTLMHead.setup: number_of_steps={}".format(self.number_of_steps))


    def configure_optimizers(self):
        optimizers = [
            torch.optim.AdamW(self.model.parameters(), lr=self.lr)
        ]
        schedulers = [
            {
                'scheduler': torch.optim.lr_scheduler.OneCycleLR(
                    optimizers[0],
                    max_lr=self.lr,
                    anneal_strategy='linear',
                    total_steps=self.number_of_steps,
                    pct_start=min(self.warmup_steps/self.number_of_steps, 0.5),
                    cycle_momentum=False,
                ),
                'interval': 'step',
                'frequency': 1,
            }
        ]
        return optimizers, schedulers

    def save(self, save_directory: str) -> None:
        """
        Saves the model at specified directory.

        Args:
            save_directory: A directory where to save the model.
        """

        self.model.save(save_directory)
from text_generation.models.gpt2_lm import GPT2LMHeadModel
from text_generation.lit_models.gpt2_lm_finetune import GPT2LMHeadModelFinetune
from text_generation.data.lines_data_module import LinesDataModule
import text_generation.utils
import pytorch_lightning as pl
from typing import Optional
import os

SAVE_PATH = os.path.realpath(os.path.join(os.path.dirname(__file__), "..", "weights"))

def finetune(
    batch_size: int,
    gpus: int,
    tpu_cores: Optional[int],
    epochs: int,
    num_workers: int,
    save_model: bool,
    max_sequence_length: int = 64,
    save_path: str = SAVE_PATH
) -> GPT2LMHeadModel:
    """
    Downloads pretrained model and tokenizer from HuggingFace and finetunes the model on a 'texts_extraced.txt' file in 'data' folder.

    Args:
        batch_size: A batch size to use during training.
        gpus: A number of gpus to use (-1 for all, 0 to not use gpus).
        tpu_cores: A number of tpu cores to use (None to not use tpu).
        epochs: A number of epochs to train for.
        num_workers: A number of parallel processes to use for preprocessing data for training (0 to not create additional processes).
        save_model: If True, will save tokenizer and model at save_path.
        max_sequence_length (optional): The maximum length of a token sequence to use in training (if longer will be truncated).
        save_path (optional): A save path where to save the tokenizer and model. By default, it is 'weights' folder.
    """

    tokenizer = text_generation.utils.get_tokenizer("gpt2")
    tokenize_args = {"tokenizer": tokenizer, "max_length": max_sequence_length}
    
    data_module = LinesDataModule(
        batch_size=batch_size,
        tokenize=text_generation.utils.tokenize,
        tokenize_args=tokenize_args,
        num_workers=num_workers,
        bos_token=tokenizer.bos_token,
        eos_token=tokenizer.eos_token
    )

    model_base = GPT2LMHeadModel(tokenizer)
    model = GPT2LMHeadModelFinetune(model_base)
    model.train()

    precision = 16
    if gpus == 0 and tpu_cores is None:
        precision = 32
    trainer = pl.Trainer(
        gpus=gpus,
        tpu_cores=tpu_cores,
        log_gpu_memory=None,
        precision=precision,
        max_epochs=epochs,
        checkpoint_callback=False
    )
    trainer.fit(model, data_module)

    if save_model:
        model.save(save_path)
        tokenizer.save_pretrained(save_directory=save_path, legacy_format=False)

    return model_base

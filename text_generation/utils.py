import transformers
import random
import torch
import numpy as np


def set_seed(seed: int) -> None:
    """
    Seeds various random generators.

    Args:
        seed: Seed to use.
    """

    random.seed(seed)
    np.random.seed(seed)
    torch.seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_tokenizer(path: str = "gpt2") -> transformers.PreTrainedTokenizerBase:
    """
    Returns a pretrained tokenizer with correctly set settings.

    Args:
        path (optional): A path to a directory containing vocabulary files.
    """

    tokenizer = transformers.GPT2TokenizerFast.from_pretrained(
        path,
        eos_token="<|EOS|>",
        bos_token="<|BOS|>",
        pad_token="<|PAD|>",
        padding_side="right"
    )

    return tokenizer


def tokenize(
    batch,
    tokenizer: transformers.PreTrainedTokenizerBase,
    max_length: int
):
    """
    Encodes 

    Args:
        batch:
        tokenizer:  A tokenizer to use for encoding.
        max_length: Maximum length of encoded sequences.
    """

    encoded = tokenizer(
        batch,
        add_special_tokens=True,
        padding=True,
        return_tensors="pt",
        return_attention_mask=True,
        return_special_tokens_mask=False,
        truncation=True,
        max_length=max_length
    )

    return encoded
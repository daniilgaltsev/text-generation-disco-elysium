from text_generation.models.gpt2_lm import GPT2LMHeadModel
from text_generation.lit_models.gpt2_lm_finetune import GPT2LMHeadModelFinetune
from text_generation.data.lines_data_module import LinesDataModule
from text_generation.data.lines_dataset import LinesDataset
import text_generation.training.finetune
import text_generation.utils
import numpy as np
import matplotlib.pyplot as plt

import ipdb
import transformers
import pytorch_lightning as pl
import os



if __name__ == "__main__":
    prompts = ["Disco ", "Kim, ", "Calculus", "Rafael ", "Hello, I'm", "Cuno!", "Hardcore"]


    model = text_generation.training.finetune.finetune(
        batch_size=8,
        gpus=1,
        tpu_cores=None,
        epochs=5,
        num_workers=0,
        save_model=True
    )

    model.eval()
    generated = model.generate(prompts, max_length=60, clean_special_tokens=False)

    for i in range(len(generated)):
        print('\n____________________')
        print('PROMPT: ', prompts[i])
        print('_____________________')
        print(generated[i])
        print('____________________')

    generated = model.generate([""] * 10, max_length=60, clean_special_tokens=False)
    for i in range(10):
        print('\n_________{}__________'.format(i))
        print(generated[i])
        print('____________________')

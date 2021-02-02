# Text Generation

Implementation of finetuning GPT-2 to generate stylized text using PyTorch Lightning and Hugging Face transformers.

## Setup

To run the project for generation only you need to install all dependencies specified in `requirement.txt`. For finetuning -- dependencies are in `requirements-dev.txt`. You can also set up a conda environment.

If you need to update requirements, edit the `requirements.in` file, then run:

```sh
pip-compile requirements.in -v --find-links=https://download.pytorch.org/whl/torch_stable.html
```

### Set up the conda environment

With conda first you need to create an environment, as defined in `environment.yml`:

```sh
conda env create -f environment.yml
```

After running `conda env create`, activate the new environment and install the requirements (on Windows might need to add `wincertstore` to requirements, on Mac might need to remove CUDA dependicies):

```sh
conda activate text-generation-disco-elysium
pip-sync requirements.txt
```

## Running the Project

To run generation you need to specify `-g` flag and optionally the number of sequences and their lengths. For example, this command will generate 5 sequences with a maximum of 200 characters:

```sh
python -m text_generation.main -g --n_sequences 5 --max_length_char 200
```

To finetune you need to specify `-f` flag and optionally the training hyperparameters. For example, this command will finetune the model for 5 epochs with a batch size of 64 using 1 gpu (this will take ~40-45 minutes to run in Colab):

```sh
python -m text_generation.main -f --epochs 5 --batch_size 64 --gpus 1
```

To see additional options add `--help` flag.

## Requirements

* PyTorch
* transformers
* PyTorch Lightning (for finetuning)


## References

* [Language Models are Unsupervised Multitask Learners](https://paperswithcode.com/paper/language-models-are-unsupervised-multitask)
* [Hugging Face GPT-2 documentation](https://huggingface.co/transformers/model_doc/gpt2.html)

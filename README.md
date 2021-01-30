# Text Generation


## Setup

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



## Requirements

* 


## References

* 

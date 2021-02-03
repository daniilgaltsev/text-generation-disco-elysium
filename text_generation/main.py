import importlib
from text_generation.models.gpt2_lm import GPT2LMHeadModel
import text_generation.utils
import transformers
import argparse
import os
from typing import Tuple, List



def finetune(args: argparse.Namespace) -> GPT2LMHeadModel:
    finetune_module = importlib.import_module("text_generation.training.finetune")
    model = finetune_module.finetune(
        batch_size=args.batch_size,
        gpus=args.gpus,
        tpu_cores=args.tpu_cores,
        epochs=args.epochs,
        num_workers=args.num_workers,
        save_model=(not args.not_save_model)
    )

    return model


def print_test_generations(model: GPT2LMHeadModel, bos_token: str = "<|BOS|>") -> None:
    prompts = ["Disco ", "Kim, ", "Calculus", "Rafael ", "Hello, I'm", "Cuno!", "Hardcore"]
    for i, prompt in enumerate(prompts):
        prompts[i] = bos_token + prompt

    model.eval()
    generated = model.generate(prompts, max_length=60, clean_special_tokens=True)

    for i in range(len(generated)):
        print('\n____________________')
        print('PROMPT: ', prompts[i])
        print('_____________________')
        print(generated[i])
        print('____________________')

    generated = model.generate([""] * 10, max_length=60, clean_special_tokens=True)
    for i in range(10):
        print('\n_________{}__________'.format(i))
        print(generated[i])
        print('____________________')


def debug_print():
    tokenizer, model = load_model()
    print_test_generations(model)


def load_model() -> Tuple[transformers.PreTrainedTokenizerBase, GPT2LMHeadModel]:
    path = text_generation.utils.WEIGHTS_PATH
    tokenizer = text_generation.utils.get_tokenizer(path)
    model = GPT2LMHeadModel(tokenizer, path)
    return tokenizer, model


def generate(args: argparse.Namespace, write_to_file: bool = True) -> List[str]:
    tokenizer, model = load_model()

    prompts = [""] * args.n_sequences
    generated = model.generate(prompts, max_length=args.max_length_token)

    for idx, seq in enumerate(generated):
        generated[idx] = seq[:args.max_length_char]

    if args.print:
        for i in range(len(generated)):
            print('\n_________{}__________'.format(i))
            print(generated[i])
            print('____________________')

    if write_to_file:
        path = os.path.join(os.path.dirname(__file__), "data", "generated.txt")
        fd = open(path, "w+", encoding="utf-8")
        for seq in generated:
            fd.write(seq + '\n')
        fd.close()
    
    return generated


def parse_args() -> argparse.Namespace:
    """
    Adds arguments to the parser and parses them from the input.
    """

    parser = argparse.ArgumentParser("Performs text generation given a model or finetunes GPT-2 with the given data.")

    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "-f", "--finetune",
        action="store_true",
        help="If present, will perform finetuning of GPT-2"
    )
    group.add_argument(
        "-g", "--generate",
        action="store_true",
        help="If present, will generate text and save it to data/generated.txt"
    )

    group = parser.add_argument_group("finetuning")
    group.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="A batch size to use during finetuning."
    )
    group.add_argument(
        "--gpus",
        type=int,
        default=0,
        help="A number of gpus to use during finetuning."
    )
    group.add_argument(
        "--tpu_cores",
        type=int,
        default=0,
        help="A number of tpu cores to use during finetuning."
    )
    group.add_argument(
        "--epochs",
        type=int,
        default=5,
        help="A number of epochs of finetuning."
    )
    group.add_argument(
        "--num_workers",
        type=int,
        default=0,
        help="A number of workers to use for data preprocessing (recommend 0, because tokenizer is already parallel and is fast enough)."
    )
    group.add_argument(
        "--not_save_model",
        action="store_true",
        help="If present, will not save model after finetuning."
    )
    group.add_argument(
        "--print_test_generations",
        action="store_true",
        help="If present, will print test generations."
    )
    
    group = parser.add_argument_group("generating")
    group.add_argument(
        "--n_sequences",
        type=int,
        default=1,
        help="A number of sequences to generate."
    )
    group.add_argument(
        "--max_length_token",
        type=int,
        default=60,
        help="A maximum number of tokens in a generated sequence."
    )
    group.add_argument(
        "--max_length_char",
        type=int,
        default=250,
        help="A maximum number of chars in a generated sequence."
    )
    group.add_argument(
        "--print",
        action="store_true",
        help="If present, will print generated sequences."
    )


    args = parser.parse_args()
    if args.tpu_cores == 0:
        args.tpu_cores = None

    return args


if __name__ == "__main__":
    args = parse_args()

    if args.finetune:
        model = finetune(args)
        if args.print_test_generations:
            print_test_generations(model)
    elif args.generate:
        generate(args)
